"""File organization: move/copy classified tracks into per-subgenre subfolders.

Implements the desktop app's auto-organization (Req 2.3/2.4): a predicted track
is placed under ``<base>/<subgenre_dirname>/``. The classifier never keeps a copy
of the user's audio — it only relocates the originals (move) or duplicates them at
the user's request (copy). Name collisions are resolved by suffixing.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from edm_classifier.config import REVIEW_DIRNAME, dirname_for


def unique_destination(dest_dir: Path, filename: str) -> Path:
    """Return a non-colliding path in ``dest_dir`` for ``filename``."""
    candidate = dest_dir / filename
    if not candidate.exists():
        return candidate
    stem, suffix = Path(filename).stem, Path(filename).suffix
    i = 1
    while True:
        candidate = dest_dir / f"{stem} ({i}){suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def place_file(source: str | Path, dest_dir: str | Path, move: bool = True) -> Path:
    """Move/copy ``source`` into ``dest_dir`` (created if needed), collision-safe."""
    source = Path(source)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = unique_destination(dest_dir, source.name)

    # A file already inside its correct target folder needs no action.
    if source.resolve() == dest.resolve():
        return source

    if move:
        shutil.move(str(source), str(dest))
    else:
        shutil.copy2(str(source), str(dest))
    return dest


def organize_file(
    source: str | Path,
    base_dir: str | Path,
    subgenre: str,
    move: bool = True,
) -> Path:
    """Place ``source`` under ``<base_dir>/<subgenre_dirname>/``."""
    return place_file(source, Path(base_dir) / dirname_for(subgenre), move=move)


def review_file(source: str | Path, base_dir: str | Path, move: bool = True) -> Path:
    """Place a low-confidence ``source`` under ``<base_dir>/Revisar/``."""
    return place_file(source, Path(base_dir) / REVIEW_DIRNAME, move=move)
