import json
from pathlib import Path


def make_train_val_split(
    input_dir: str | Path,
    output_path: str | Path,
    train_fraction: float = 2.0 / 3.0,
) -> dict[str, list[str]]:
    """Create deterministic train/val split over TFRecord files.

    Args:
        input_dir: Directory containing Waymo ``*.tfrecord`` files.
        output_path: JSON path to write.
        train_fraction: Fraction of sorted files assigned to train.

    Returns:
        Dict with ``train`` and ``val`` file lists.
    """

    files = sorted(str(p) for p in Path(input_dir).glob("*.tfrecord*"))
    split_idx = int(round(len(files) * train_fraction))
    split = {"train": files[:split_idx], "val": files[split_idx:]}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(split, indent=2), encoding="utf-8")
    return split


def make_official_train_val_split(
    train_dir: str | Path,
    val_dir: str | Path,
    output_path: str | Path,
) -> dict[str, list[str]]:
    """Create a split JSON from separate official Waymo train/validation dirs."""

    split = {
        "train": sorted(str(p) for p in Path(train_dir).glob("*.tfrecord*")),
        "val": sorted(str(p) for p in Path(val_dir).glob("*.tfrecord*")),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(split, indent=2), encoding="utf-8")
    return split


def load_split(path: str | Path) -> dict[str, list[str]]:
    """Load a split JSON created by ``make_train_val_split``."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--train-dir", default=None)
    parser.add_argument("--val-dir", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.train_dir is not None or args.val_dir is not None:
        if args.train_dir is None or args.val_dir is None:
            raise SystemExit("--train-dir and --val-dir must be provided together")
        split = make_official_train_val_split(args.train_dir, args.val_dir, args.output)
    elif args.input_dir is not None:
        split = make_train_val_split(args.input_dir, args.output)
    else:
        raise SystemExit("Provide either --input-dir or both --train-dir/--val-dir")
    print(f"train={len(split['train'])} val={len(split['val'])}")
