"""Train / validation / test splitting.

Splits happen at the **track** level, never at the segment level. Because each
track is later expanded into many 2-second segments, splitting whole tracks is
what prevents segments of the same song leaking across train/val/test — the key
lesson carried over from the earlier prototypes. The split is stratified so the
class balance (Req 3.4) is preserved in every partition.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sklearn.model_selection import train_test_split

from edm_classifier.config import settings
from edm_classifier.data.dataset import TrackRecord


@dataclass(frozen=True)
class DataSplit:
    """A train/validation/test partition of track records."""

    train: list[TrackRecord]
    val: list[TrackRecord]
    test: list[TrackRecord]

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def _record_to_dict(r: TrackRecord) -> dict[str, object]:
    return {"path": str(r.path), "subgenre": r.subgenre, "label": r.label}


def _record_from_dict(d: dict[str, object]) -> TrackRecord:
    return TrackRecord(
        path=Path(str(d["path"])),
        subgenre=str(d["subgenre"]),
        label=int(d["label"]),
    )


def save_split(split: DataSplit, path: str | Path, seed: int | None = None) -> None:
    """Persist a split to JSON so train/val/test stay fixed across runs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed if seed is not None else settings.dataset.seed,
        "train": [_record_to_dict(r) for r in split.train],
        "val": [_record_to_dict(r) for r in split.val],
        "test": [_record_to_dict(r) for r in split.test],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_split(path: str | Path) -> DataSplit:
    """Load a persisted split from JSON."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return DataSplit(
        train=[_record_from_dict(d) for d in payload["train"]],
        val=[_record_from_dict(d) for d in payload["val"]],
        test=[_record_from_dict(d) for d in payload["test"]],
    )


def stratified_split(
    records: list[TrackRecord],
    train_ratio: float | None = None,
    val_ratio: float | None = None,
    test_ratio: float | None = None,
    seed: int | None = None,
) -> DataSplit:
    """Split records into stratified train/val/test partitions.

    Ratios default to the dataset config (70/15/15). They must sum to 1.

    Args:
        records: All labeled tracks.
        train_ratio, val_ratio, test_ratio: Partition fractions.
        seed: RNG seed for reproducibility.

    Returns:
        A :class:`DataSplit` with class balance preserved across partitions.
    """
    cfg = settings.dataset
    train_ratio = train_ratio if train_ratio is not None else cfg.train_ratio
    val_ratio = val_ratio if val_ratio is not None else cfg.val_ratio
    test_ratio = test_ratio if test_ratio is not None else cfg.test_ratio
    seed = seed if seed is not None else cfg.seed

    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1, got {total}.")
    if not records:
        raise ValueError("Cannot split an empty record list.")

    labels = [r.label for r in records]

    # First split off the training set; then divide the remainder into val/test.
    train, remainder = train_test_split(
        records,
        train_size=train_ratio,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )
    rem_labels = [r.label for r in remainder]
    # Fraction of the remainder that should become the test set.
    test_fraction_of_rem = test_ratio / (val_ratio + test_ratio)
    val, test = train_test_split(
        remainder,
        test_size=test_fraction_of_rem,
        random_state=seed,
        shuffle=True,
        stratify=rem_labels,
    )
    return DataSplit(train=train, val=val, test=test)
