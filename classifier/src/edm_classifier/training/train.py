"""Training loop for the Short-chunk CNN (WBS 4.4).

Reads the precomputed feature cache and the persisted split, trains with
validation-based early stopping and light SpecAugment data augmentation, saves
the best checkpoint, and evaluates on the test set at both segment and (properly
aggregated) track level. Device-agnostic: CUDA (Colab) > MPS (Mac) > CPU.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from edm_classifier import SUBGENRES
from edm_classifier.data.splits import load_split
from edm_classifier.data.torch_dataset import CachedSegmentDataset, build_split_subsets
from edm_classifier.models.short_chunk_cnn import build_model
from edm_classifier.training.evaluation import EvaluationResult, aggregate_segment_probs, evaluate


def select_device(prefer: str = "auto") -> torch.device:
    """Pick the best available device unless one is explicitly requested."""
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class TrainConfig:
    """Hyperparameters for a training run."""

    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8  # early-stopping patience
    monitor: str = "val_acc"  # metric for checkpoint/early-stop: "val_acc" | "val_loss"
    num_workers: int = 0
    n_channels: int = 128
    dropout: float = 0.5
    use_spec_augment: bool = True
    freq_mask: int = 16
    time_mask: int = 16
    seed: int = 42
    verbose: bool = True  # print a line per epoch during training
    use_amp: bool = True  # mixed precision (only effective on CUDA)
    label_smoothing: float = 0.0  # softens targets; curbs overconfidence
    scheduler: str = "none"  # LR schedule: "none" | "cosine"


def spec_augment(x: torch.Tensor, cfg: TrainConfig) -> torch.Tensor:
    """Mask a random frequency band and time span per sample (SpecAugment)."""
    x = x.clone()
    _, _, n_mels, n_frames = x.shape
    fill = x.mean()
    if cfg.freq_mask > 0:
        f = int(torch.randint(0, min(cfg.freq_mask, n_mels) + 1, (1,)).item())
        if f > 0:
            f0 = int(torch.randint(0, n_mels - f + 1, (1,)).item())
            x[:, :, f0 : f0 + f, :] = fill
    if cfg.time_mask > 0:
        t = int(torch.randint(0, min(cfg.time_mask, n_frames) + 1, (1,)).item())
        if t > 0:
            t0 = int(torch.randint(0, n_frames - t + 1, (1,)).item())
            x[:, :, :, t0 : t0 + t] = fill
    return x


@torch.no_grad()
def _gather_probs(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over a loader; return (probs, labels) in loader order."""
    model.eval()
    on_cuda = device.type == "cuda"
    probs_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    for x, y in loader:
        x = x.to(device, non_blocking=on_cuda)
        with torch.autocast(device_type=device.type, enabled=on_cuda):
            logits = model(x)
        probs_chunks.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        label_chunks.append(y.numpy())
    return np.concatenate(probs_chunks), np.concatenate(label_chunks)


def _track_level_result(
    dataset: CachedSegmentDataset, subset: Subset, seg_probs: np.ndarray
) -> EvaluationResult:
    """Aggregate segment probabilities to track level and evaluate."""
    seg_track_ids = dataset.track_ids[np.asarray(subset.indices)]
    track_probs, unique_ids = aggregate_segment_probs(seg_probs, seg_track_ids)
    track_id_to_label = {int(t["track_id"]): int(t["label"]) for t in dataset.index["tracks"]}
    track_true = np.array([track_id_to_label[int(tid)] for tid in unique_ids])
    return evaluate(track_true, track_probs)


def train_model(
    cache_dir: str | Path,
    splits_path: str | Path,
    out_dir: str | Path,
    config: TrainConfig | None = None,
    device: str = "auto",
) -> dict:
    """Train the model and save the best checkpoint plus a metrics report.

    Returns a dict with the training history and test-set metrics (segment and
    track level).
    """
    config = config or TrainConfig()
    torch.manual_seed(config.seed)
    dev = select_device(device)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    on_cuda = dev.type == "cuda"
    use_amp = config.use_amp and on_cuda
    if on_cuda:
        torch.backends.cudnn.benchmark = True  # autotune conv kernels for fixed shapes

    split = load_split(splits_path)
    train_set, val_set, test_set = build_split_subsets(cache_dir, split)
    dataset: CachedSegmentDataset = train_set.dataset  # shared underlying cache

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,  # keeps BatchNorm happy (no size-1 batches)
        num_workers=config.num_workers,
        pin_memory=on_cuda,
        persistent_workers=config.num_workers > 0,
    )
    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": on_cuda,
        "persistent_workers": config.num_workers > 0,
    }
    val_loader = DataLoader(val_set, **loader_kwargs)
    test_loader = DataLoader(test_set, **loader_kwargs)

    model = build_model(n_channels=config.n_channels, dropout=config.dropout).to(dev)
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

    # Checkpoint / early-stop on the monitored metric. Default is val accuracy
    # (the target metric): with small track counts, val loss can rise from
    # overconfidence while accuracy still improves, so monitoring loss would save
    # an under-trained checkpoint.
    monitor_acc = config.monitor == "val_acc"
    best_metric = -float("inf") if monitor_acc else float("inf")
    best_state: dict | None = None
    epochs_without_improvement = 0
    history: list[dict] = []

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for x, y in train_loader:
            x, y = x.to(dev, non_blocking=on_cuda), y.to(dev, non_blocking=on_cuda)
            if config.use_spec_augment:
                x = spec_augment(x, config)
            optimizer.zero_grad()
            with torch.autocast(device_type=dev.type, enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item())
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)

        # Validation. Cross-entropy computed directly from the probabilities:
        # mean negative log-likelihood of the true class.
        val_probs, val_labels = _gather_probs(model, val_loader, dev)
        if len(val_labels):
            true_probs = val_probs[np.arange(len(val_labels)), val_labels]
            val_loss = float(-np.log(np.clip(true_probs, 1e-9, 1.0)).mean())
            val_acc = float((val_probs.argmax(axis=1) == val_labels).mean())
        else:
            val_loss, val_acc = float("inf"), 0.0
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
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
                f"[epoch {epoch + 1:>3}/{config.epochs}] "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_acc={val_acc:.3f}{marker}",
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

    # Final test evaluation: segment level and aggregated track level.
    test_probs, test_labels = _gather_probs(model, test_loader, dev)
    segment_result = evaluate(test_labels, test_probs)
    track_result = _track_level_result(dataset, test_set, test_probs)

    checkpoint = {
        "model": model.state_dict(),
        "model_kwargs": {"n_channels": config.n_channels, "dropout": config.dropout},
        "config": asdict(config),
    }
    torch.save(checkpoint, out_dir / "model.pt")

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


def evaluate_checkpoint(
    cache_dir: str | Path,
    splits_path: str | Path,
    checkpoint_path: str | Path,
    partition: str = "test",
    device: str = "auto",
) -> dict:
    """Evaluate a saved checkpoint on a partition without retraining.

    Returns segment- and track-level metrics (including the confusion matrix and
    per-class F1) for the chosen partition ("train" | "val" | "test").
    """
    dev = select_device(device)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model_kwargs = ckpt.get("model_kwargs", {}) if isinstance(ckpt, dict) else {}
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model = build_model(**model_kwargs).to(dev)
    model.load_state_dict(state)

    split = load_split(splits_path)
    names = ("train", "val", "test")
    subsets = dict(zip(names, build_split_subsets(cache_dir, split), strict=True))
    subset = subsets[partition]
    dataset: CachedSegmentDataset = subset.dataset

    loader = DataLoader(subset, batch_size=128, num_workers=0, pin_memory=dev.type == "cuda")
    probs, labels = _gather_probs(model, loader, dev)
    segment_result = evaluate(labels, probs)
    track_result = _track_level_result(dataset, subset, probs)

    return {
        "partition": partition,
        "subgenres": list(SUBGENRES),
        "segment": segment_result.to_dict(),
        "track": track_result.to_dict(),
        "meets_targets": track_result.meets_targets(),
    }
