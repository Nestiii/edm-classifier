"""File organization: move/copy classified tracks into per-subgenre subfolders.

Implements the desktop app's auto-organization (Req 2.3/2.4): a predicted track
is placed under ``<base>/<subgenre_dirname>/``. The classifier never keeps a copy
of the user's audio — it only relocates the originals (move) or duplicates them at
the user's request (copy). Name collisions are resolved by suffixing.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from edm_classifier.config import dirname_for


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


def organize_file(
    source: str | Path,
    base_dir: str | Path,
    subgenre: str,
    move: bool = True,
) -> Path:
    """Place ``source`` under ``<base_dir>/<subgenre_dirname>/``.

    Args:
        source: The audio file to organize.
        base_dir: Root under which per-subgenre folders are created.
        subgenre: Canonical subgenre label (its dirname is looked up).
        move: Move the file when True, copy it when False.

    Returns:
        The destination path.
    """
    source = Path(source)
    dest_dir = Path(base_dir) / dirname_for(subgenre)
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
