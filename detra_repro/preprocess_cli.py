import time
import json
from pathlib import Path

from detra_repro.config import DataConfig
from detra_repro.data.waymo import make_scene_sample, read_waymo_frames
from detra_repro.data.cache_v2 import (
    CACHE_SCHEMA_VERSION,
    MANIFEST_NAME,
    atomic_save_sample,
    finalize_cache_manifest,
    sha256_file,
    validate_v2_sample,
    write_segment_manifest,
)
from detra_repro.splits import load_split


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds as a compact human-readable string."""

    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def preprocess_files(
    files: list[str],
    output_dir: str | Path,
    cfg: DataConfig,
    max_segments: int | None = None,
    max_frames_per_segment: int | None = None,
    sample_stride: int = 5,
    skip_existing: bool = True,
    hash_sources: bool = False,
) -> int:
    """Cache Waymo segment samples in schema v2.

    Args:
        files: TFRecord paths.
        output_dir: Cache root.
        cfg: Data config.
        max_segments: Optional segment cap for pilot runs.
        max_frames_per_segment: Optional frame cap.
        sample_stride: Save every N-th usable frame.
        skip_existing: If true, do not rewrite samples that are already cached.

    Returns:
        Number of cached samples.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    selected_files = files[:max_segments] if max_segments is not None else files
    start_time = time.monotonic()

    for segment_idx, tfrecord in enumerate(selected_files):
        segment_start = time.monotonic()
        segment_name = Path(tfrecord).stem.replace("_with_camera_labels", "")
        segment_dir = output_dir / segment_name
        segment_dir.mkdir(parents=True, exist_ok=True)
        elapsed = time.monotonic() - start_time
        done = max(segment_idx, 1)
        eta = elapsed / done * (len(selected_files) - segment_idx) if segment_idx else 0.0
        print(
            f"[{segment_idx + 1}/{len(selected_files)}] reading {Path(tfrecord).name} "
            f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta)}",
            flush=True,
        )
        frames = read_waymo_frames(tfrecord, max_frames=max_frames_per_segment)
        last_usable = len(frames) - cfg.future_steps * cfg.frame_stride
        if last_usable <= 0:
            print(f"  skipped: only {len(frames)} frames for requested horizon", flush=True)
            continue

        segment_written = 0
        segment_skipped = 0
        first_usable = cfg.history_sweeps - 1
        segment_stats = {
            "samples": 0,
            "expected_samples": len(range(first_usable, last_usable, sample_stride)),
            "raw_frames": len(frames),
            "points": 0,
            "gt": 0,
            "paper_vehicle_gt": 0,
            "bytes": 0,
            "point_min": 0,
            "point_max": 0,
        }
        point_counts: list[int] = []
        for current_index in range(first_usable, last_usable, sample_stride):
            out = segment_dir / f"sample_{current_index:06d}.npz"
            if skip_existing and out.exists():
                try:
                    stats = validate_v2_sample(out)
                    segment_skipped += 1
                    segment_stats["samples"] += 1
                    segment_stats["points"] += stats["num_points"]
                    segment_stats["gt"] += stats["num_gt"]
                    segment_stats["paper_vehicle_gt"] += stats["num_paper_vehicle_gt"]
                    segment_stats["bytes"] += out.stat().st_size
                    point_counts.append(stats["num_points"])
                    continue
                except (OSError, ValueError, KeyError):
                    print(f"  rebuilding invalid cache-v2 sample: {out}", flush=True)
            sample = make_scene_sample(frames, current_index, cfg)
            atomic_save_sample(out, sample, Path(tfrecord).name)
            validate_v2_sample(out)
            total += 1
            segment_written += 1
            segment_stats["samples"] += 1
            segment_stats["points"] += len(sample.points)
            segment_stats["gt"] += len(sample.gt_boxes)
            segment_stats["paper_vehicle_gt"] += int(sample.paper_vehicle_eligible.sum())
            segment_stats["bytes"] += out.stat().st_size
            point_counts.append(len(sample.points))
        if point_counts:
            segment_stats["point_min"] = min(point_counts)
            segment_stats["point_max"] = max(point_counts)
        write_segment_manifest(
            segment_dir,
            source_tfrecord=str(tfrecord),
            source_sha256=sha256_file(tfrecord) if hash_sources else None,
            cfg=cfg,
            sample_stride=sample_stride,
            stats=segment_stats,
        )
        segment_elapsed = time.monotonic() - segment_start
        elapsed = time.monotonic() - start_time
        done = segment_idx + 1
        eta = elapsed / done * (len(selected_files) - done)
        print(
            f"  wrote={segment_written} skipped_existing={segment_skipped} "
            f"segment_time={_format_duration(segment_elapsed)} cached_new={total} "
            f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta)}",
            flush=True,
        )

    return total


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-segments", type=int, default=None)
    parser.add_argument("--start-segment", type=int, default=0)
    parser.add_argument("--max-frames-per-segment", type=int, default=None)
    parser.add_argument("--sample-stride", type=int, default=5)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument("--history-sweeps", type=int, default=5)
    parser.add_argument("--future-steps", type=int, default=10)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--roi", type=float, nargs=4, default=(-80.0, -80.0, 80.0, 80.0))
    parser.add_argument("--voxel-size", type=float, default=0.4)
    parser.add_argument("--hash-sources", action="store_true")
    parser.add_argument("--finalize-manifest-only", action="store_true")
    args = parser.parse_args()

    cfg = DataConfig(
        history_sweeps=args.history_sweeps,
        future_steps=args.future_steps,
        frame_stride=args.frame_stride,
        roi_m=tuple(args.roi),
        voxel_size_m=args.voxel_size,
    )
    split = load_split(args.split_json)
    output_dir = Path(args.output_dir)
    existing_manifest = output_dir / MANIFEST_NAME
    if existing_manifest.exists():
        existing_version = int(
            json.loads(existing_manifest.read_text(encoding="utf-8")).get(
                "cache_schema_version", 0
            )
        )
        if existing_version != CACHE_SCHEMA_VERSION:
            raise SystemExit(
                f"{output_dir} already contains cache schema {existing_version}; "
                f"required schema {CACHE_SCHEMA_VERSION}"
            )
    if args.finalize_manifest_only:
        payload = finalize_cache_manifest(
            output_dir,
            split=args.split,
            split_json=args.split_json,
            cfg=cfg,
            sample_stride=args.sample_stride,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        raise SystemExit(0)
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise SystemExit("--shard-index must be in [0, num_shards)")
    files = split[args.split]
    if args.start_segment < 0:
        raise SystemExit("--start-segment must be >= 0")
    if args.max_segments is not None:
        files = files[: args.max_segments]
    if args.start_segment:
        files = files[args.start_segment :]
    if args.num_shards > 1:
        files = files[args.shard_index :: args.num_shards]
        print(
            f"shard {args.shard_index + 1}/{args.num_shards}: "
            f"{len(files)} {args.split} segments after start={args.start_segment}",
            flush=True,
        )
    count = preprocess_files(
        files,
        args.output_dir,
        cfg,
        max_segments=None,
        max_frames_per_segment=args.max_frames_per_segment,
        sample_stride=args.sample_stride,
        skip_existing=not args.overwrite_existing,
        hash_sources=args.hash_sources,
    )
    print(f"cached {count} samples")
