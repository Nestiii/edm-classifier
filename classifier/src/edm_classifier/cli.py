"""Command-line entry point for the classifier service.

Lets the Python module run fully standalone (independent of the desktop UI),
as required by the design spec. Subcommands are wired up as the corresponding
features land.
"""

from __future__ import annotations

import argparse

from edm_classifier import SUBGENRES, __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="edm-classifier",
        description="Automatic subgenre classification of electronic dance music.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--list-subgenres",
        action="store_true",
        help="Print the eight target subgenres and exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_subgenres:
        for i, name in enumerate(SUBGENRES):
            print(f"{i}: {name}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
