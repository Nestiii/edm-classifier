"""Command-line entry point for the classifier service.

Lets the Python module run fully standalone (independent of the desktop UI).
Exposes the ordered data pipeline as subcommands so the same package drives both
local development and cloud (Colab) runs:

    edm-classifier manifest   --root data/raw --out data/manifest.csv
    edm-classifier validate   --manifest data/manifest.csv
    edm-classifier split      --manifest data/manifest.csv --out data/splits.json
    edm-classifier preprocess --manifest data/manifest.csv --cache data/cache
"""

from __future__ import annotations

import argparse

from edm_classifier import SUBGENRES, __version__


def _cmd_list_subgenres(_: argparse.Namespace) -> int:
    for i, name in enumerate(SUBGENRES):
        print(f"{i}: {name}")
    return 0


def _cmd_manifest(args: argparse.Namespace) -> int:
    from edm_classifier.data.manifest import build_manifest, save_manifest

    entries = build_manifest(args.root)
    save_manifest(entries, args.out)
    print(f"Wrote manifest with {len(entries)} tracks to {args.out}")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    from edm_classifier.data.manifest import load_manifest
    from edm_classifier.data.validate import validate_manifest

    entries = load_manifest(args.manifest)
    report = validate_manifest(entries, require_dual_source=args.require_dual_source)
    print(report.summary())
    return 0 if report.ok else 1


def _cmd_split(args: argparse.Namespace) -> int:
    from edm_classifier.data.manifest import load_manifest, to_track_records
    from edm_classifier.data.splits import save_split, stratified_split

    records = to_track_records(load_manifest(args.manifest))
    split = stratified_split(records, seed=args.seed)
    save_split(split, args.out, seed=args.seed)
    print(f"Wrote split to {args.out}: {split.sizes()}")
    return 0


def _cmd_preprocess(args: argparse.Namespace) -> int:
    from edm_classifier.data.manifest import load_manifest, to_track_records
    from edm_classifier.data.preprocess import preprocess_dataset

    records = to_track_records(load_manifest(args.manifest))

    def progress(done: int, total: int) -> None:
        if done % 25 == 0 or done == total:
            print(f"  preprocessed {done}/{total} tracks", flush=True)

    result = preprocess_dataset(records, args.cache, progress=progress)
    print(
        f"Cached {result.n_segments} segments from {result.n_tracks} tracks "
        f"({result.n_mels}x{result.n_frames}) to {result.cache_dir}"
    )
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    import json

    from edm_classifier.training.train import TrainConfig, train_model

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        n_channels=args.n_channels,
    )
    report = train_model(args.cache, args.splits, args.out, config=config, device=args.device)
    print(json.dumps(report, indent=2))
    return 0


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
    sub = parser.add_subparsers(dest="command")

    p_manifest = sub.add_parser("manifest", help="Scan a dataset root and write a manifest CSV.")
    p_manifest.add_argument("--root", required=True, help="Dataset root (one folder per subgenre).")
    p_manifest.add_argument("--out", required=True, help="Output manifest CSV path.")
    p_manifest.set_defaults(func=_cmd_manifest)

    p_validate = sub.add_parser("validate", help="Validate a manifest against the requirements.")
    p_validate.add_argument("--manifest", required=True, help="Manifest CSV path.")
    p_validate.add_argument(
        "--require-dual-source",
        action="store_true",
        help="Treat missing dual-source validation as an error (Req 3.2).",
    )
    p_validate.set_defaults(func=_cmd_validate)

    p_split = sub.add_parser("split", help="Write a persisted train/val/test split.")
    p_split.add_argument("--manifest", required=True, help="Manifest CSV path.")
    p_split.add_argument("--out", required=True, help="Output splits.json path.")
    p_split.add_argument("--seed", type=int, default=None, help="RNG seed (default: config).")
    p_split.set_defaults(func=_cmd_split)

    p_pre = sub.add_parser("preprocess", help="Precompute the mel-spectrogram feature cache.")
    p_pre.add_argument("--manifest", required=True, help="Manifest CSV path.")
    p_pre.add_argument("--cache", required=True, help="Output cache directory.")
    p_pre.set_defaults(func=_cmd_preprocess)

    p_train = sub.add_parser("train", help="Train the model from the feature cache.")
    p_train.add_argument("--cache", required=True, help="Feature cache directory.")
    p_train.add_argument("--splits", required=True, help="Persisted splits.json path.")
    p_train.add_argument("--out", required=True, help="Output directory for checkpoint + report.")
    p_train.add_argument("--epochs", type=int, default=50)
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--n-channels", type=int, default=128)
    p_train.add_argument("--device", default="auto", help="auto|cuda|mps|cpu")
    p_train.set_defaults(func=_cmd_train)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_subgenres:
        return _cmd_list_subgenres(args)

    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 0
    return func(args)


if __name__ == "__main__":
    raise SystemExit(main())
