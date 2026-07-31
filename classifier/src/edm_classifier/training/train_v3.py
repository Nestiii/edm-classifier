"""Training loop for the v3 late-fusion model (mel + tempograms).

Mirrors :mod:`edm_classifier.training.train` but feeds three inputs per batch
(mel, Fourier tempogram, autocorrelation tempogram) to the fusion model. Reuses
the same device selection, SpecAugment, AMP, early stopping and track-level
aggregation.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from edm_classifier import SUBGENRES
from edm_classifier.data.splits import load_split
from edm_classifier.data.torch_dataset import (
    CachedMultiFeatureDataset,
    build_multifeature_split_subsets,
)
from edm_classifier.models.fusion import build_fusion_model
from edm_classifier.training.evaluation import evaluate
from edm_classifier.training.train import (
    TrainConfig,
    _track_level_result,
    select_device,
    spec_augment,
)


@torch.no_grad()
def _gather_probs_mf(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    on_cuda = device.type == "cuda"
    probs_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    for mel, fourier, autocorr, y in loader:
        mel = mel.to(device, non_blocking=on_cuda)
        fourier = fourier.to(device, non_blocking=on_cuda)
        autocorr = autocorr.to(device, non_blocking=on_cuda)
        with torch.autocast(device_type=device.type, enabled=on_cuda):
            logits = model(mel, fourier, autocorr)
        probs_chunks.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        label_chunks.append(y.numpy())
    return np.concatenate(probs_chunks), np.concatenate(label_chunks)


def train_fusion_model(
    cache_dir: str | Path,
    splits_path: str | Path,
    out_dir: str | Path,
    config: TrainConfig | None = None,
    device: str = "auto",
) -> dict:
    """Train the late-fusion model and save the best checkpoint + metrics report."""
    config = config or TrainConfig()
    torch.manual_seed(config.seed)
    dev = select_device(device)
    on_cuda = dev.type == "cuda"
    use_amp = config.use_amp and on_cuda
    if on_cuda:
        torch.backends.cudnn.benchmark = True

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split = load_split(splits_path)
    train_set, val_set, test_set = build_multifeature_split_subsets(cache_dir, split)
    dataset: CachedMultiFeatureDataset = train_set.dataset

    loader_kwargs = {
        "num_workers": config.num_workers,
        "pin_memory": on_cuda,
        "persistent_workers": config.num_workers > 0,
    }
    train_loader = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True, drop_last=True, **loader_kwargs
    )
    val_loader = DataLoader(val_set, batch_size=config.batch_size, **loader_kwargs)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, **loader_kwargs)

    model = build_fusion_model(n_channels=config.n_channels, dropout=config.dropout).to(dev)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    lr_scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
        if config.scheduler == "cosine"
        else None
    )

    monitor_acc = config.monitor == "val_acc"
    best_metric = -float("inf") if monitor_acc else float("inf")
    best_state: dict | None = None
    epochs_without_improvement = 0
    history: list[dict] = []

    for epoch in range(config.epochs):
        model.train()
        running_loss, n_batches = 0.0, 0
        for mel, fourier, autocorr, y in train_loader:
            mel = mel.to(dev, non_blocking=on_cuda)
            fourier = fourier.to(dev, non_blocking=on_cuda)
            autocorr = autocorr.to(dev, non_blocking=on_cuda)
            y = y.to(dev, non_blocking=on_cuda)
            if config.use_spec_augment:
                mel = spec_augment(mel, config)  # augment the mel branch only
            optimizer.zero_grad()
            with torch.autocast(device_type=dev.type, enabled=use_amp):
                loss = criterion(model(mel, fourier, autocorr), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item())
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)

        val_probs, val_labels = _gather_probs_mf(model, val_loader, dev)
        if len(val_labels):
            true_probs = val_probs[np.arange(len(val_labels)), val_labels]
            val_loss = float(-np.log(np.clip(true_probs, 1e-9, 1.0)).mean())
            val_acc = float((val_probs.argmax(axis=1) == val_labels).mean())
        else:
            val_loss, val_acc = float("inf"), 0.0
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc}
        )

        current = val_acc if monitor_acc else val_loss
        improved = (current > best_metric + 1e-4) if monitor_acc else (current < best_metric - 1e-4)
        if improved:
            best_metric = current
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if config.verbose:
            marker = "  <- best" if improved else ""
            print(
                f"[epoch {epoch + 1:>3}/{config.epochs}] train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}{marker}",
                flush=True,
            )

        if lr_scheduler is not None:
            lr_scheduler.step()

        if not improved and epochs_without_improvement >= config.patience:
            if config.verbose:
                print(f"Early stopping at epoch {epoch + 1} (no val improvement).", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_probs, test_labels = _gather_probs_mf(model, test_loader, dev)
    segment_result = evaluate(test_labels, test_probs)
    track_result = _track_level_result(dataset, test_set, test_probs)

    torch.save(
        {
            "model": model.state_dict(),
            "model_kwargs": {"n_channels": config.n_channels, "dropout": config.dropout},
            "config": asdict(config),
        },
        out_dir / "model.pt",
    )
    report = {
        "history": history,
        "epochs_trained": len(history),
        "subgenres": list(SUBGENRES),
        "test_segment": segment_result.summary(),
        "test_track": track_result.to_dict(),
        "meets_targets": track_result.meets_targets(),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if config.verbose:
        print(track_result.format_confusion(), flush=True)
    return report
