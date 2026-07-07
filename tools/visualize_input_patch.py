from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon


MAP_TOKEN_TYPES = (
    "lane",
    "road_line",
    "road_edge",
    "crosswalk",
    "speed_bump",
    "stop_sign",
    "driveway",
)

MAP_COLORS = {
    "lane": "#2ca02c",
    "road_line": "#ff7f0e",
    "road_edge": "#8c564b",
    "crosswalk": "#9467bd",
    "speed_bump": "#d62728",
    "stop_sign": "#e377c2",
    "driveway": "#17becf",
}


def _box_corners(box: np.ndarray) -> np.ndarray:
    """Return oriented BEV box corners for ``[x,y,z,l,w,h,heading]``."""

    x, y, _, length, width, _, heading = box.astype(np.float32)
    local = np.array(
        [
            [length / 2, width / 2],
            [length / 2, -width / 2],
            [-length / 2, -width / 2],
            [-length / 2, width / 2],
        ],
        dtype=np.float32,
    )
    c, s = np.cos(heading), np.sin(heading)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return local @ rot.T + np.array([x, y], dtype=np.float32)


def _draw_points_by_sweep(ax: plt.Axes, points: np.ndarray, max_points_per_sweep: int) -> None:
    """Draw LiDAR points with one color per sweep timestamp.

    Args:
        ax: Matplotlib axes.
        points: ``[P,4]`` cached points with ``x,y,z,dt`` columns.
        max_points_per_sweep: Plot cap per unique ``dt`` value.
    """

    if len(points) == 0:
        return
    sweep_times = np.unique(np.round(points[:, 3], decimals=3))
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, max(len(sweep_times), 1)))
    rng = np.random.default_rng(0)
    for color, dt in zip(colors, sweep_times):
        mask = np.isclose(np.round(points[:, 3], decimals=3), dt)
        sweep_points = points[mask]
        if len(sweep_points) > max_points_per_sweep:
            idx = rng.choice(len(sweep_points), max_points_per_sweep, replace=False)
            sweep_points = sweep_points[idx]
        label = f"dt={dt:.1f}s"
        ax.scatter(
            sweep_points[:, 0],
            sweep_points[:, 1],
            s=0.08,
            color=color,
            alpha=0.55,
            linewidths=0,
            label=label,
            rasterized=True,
        )


def _draw_map(ax: plt.Axes, map_xy: np.ndarray, map_features: np.ndarray) -> dict[str, int]:
    """Draw typed map tokens and return per-type counts."""

    counts: dict[str, int] = {}
    if len(map_xy) == 0 or len(map_features) == 0:
        return counts
    type_idx = np.argmax(map_features[:, : len(MAP_TOKEN_TYPES)], axis=1)
    for idx, name in enumerate(MAP_TOKEN_TYPES):
        mask = type_idx == idx
        count = int(mask.sum())
        counts[name] = count
        if count == 0:
            continue
        xy = map_xy[mask]
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=4.0,
            color=MAP_COLORS[name],
            marker=".",
            alpha=0.85,
            linewidths=0,
            label=name,
        )
    return counts


def _draw_gt(ax: plt.Axes, boxes: np.ndarray, future: np.ndarray, future_mask: np.ndarray) -> None:
    """Draw GT current boxes and future center trajectories."""

    for obj_idx, box in enumerate(boxes):
        ax.add_patch(
            Polygon(
                _box_corners(box),
                closed=True,
                fill=False,
                edgecolor="#0057b8",
                linewidth=1.2,
                alpha=0.95,
            )
        )
        ax.scatter(box[0], box[1], s=14, color="#0057b8", marker="x", linewidths=1.0)
        if obj_idx >= len(future):
            continue
        valid = future_mask[obj_idx].astype(bool)
        if not valid.any():
            continue
        xy = future[obj_idx, valid]
        ax.plot(xy[:, 0], xy[:, 1], color="#0047ab", linewidth=1.2, alpha=0.9)
        ax.scatter(xy[-1, 0], xy[-1, 1], s=18, color="#0047ab", marker="o", linewidths=0)


def _load_sample(path: Path) -> dict[str, np.ndarray]:
    """Load one cached ``sample_*.npz`` into numpy arrays."""

    with np.load(path, allow_pickle=True) as data:
        return {
            "points": data["points"].astype(np.float32),
            "gt_boxes": data["gt_boxes"].astype(np.float32),
            "gt_future": data["gt_future"].astype(np.float32),
            "gt_future_mask": data["gt_future_mask"].astype(bool),
            "map_xy": data["map_xy"].astype(np.float32) if "map_xy" in data else np.zeros((0, 2), np.float32),
            "map_features": data["map_features"].astype(np.float32)
            if "map_features" in data
            else np.zeros((0, len(MAP_TOKEN_TYPES) + 5), np.float32),
        }


def plot_sample(
    sample_path: Path,
    output_path: Path,
    roi: tuple[float, float, float, float],
    max_points_per_sweep: int,
) -> None:
    """Create one BEV input-patch plot.

    The plot contains:
    - LiDAR points colored by sweep timestamp ``dt``.
    - Current GT boxes and GT future center trajectories.
    - Cached map tokens colored by semantic map type.
    """

    sample = _load_sample(sample_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 9), dpi=180)

    _draw_points_by_sweep(ax, sample["points"], max_points_per_sweep=max_points_per_sweep)
    map_counts = _draw_map(ax, sample["map_xy"], sample["map_features"])
    _draw_gt(ax, sample["gt_boxes"], sample["gt_future"], sample["gt_future_mask"])

    ego_box = np.array([0.0, 0.0, 0.0, 4.8, 2.0, 1.6, 0.0], dtype=np.float32)
    ax.add_patch(
        Polygon(
            _box_corners(ego_box),
            closed=True,
            fill=False,
            edgecolor="black",
            linewidth=1.5,
            linestyle="--",
        )
    )
    ax.scatter(0.0, 0.0, s=30, color="black", marker="+", linewidths=1.2)

    x_min, y_min, x_max, y_max = roi
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.25, alpha=0.35)
    ax.set_xlabel("x forward from current ego frame [m]")
    ax.set_ylabel("y left from current ego frame [m]")
    ax.set_title(
        "\n".join(
            [
                sample_path.parent.name,
                f"{sample_path.name} | points={len(sample['points'])} gt={len(sample['gt_boxes'])} map={len(sample['map_xy'])}",
            ]
        ),
        fontsize=9,
    )

    handles = [
        Line2D([0], [0], color="#0057b8", lw=1.5, label="GT box/future"),
        Line2D([0], [0], color="black", lw=1.5, linestyle="--", label="ego"),
    ]
    for name, count in map_counts.items():
        if count:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=MAP_COLORS[name],
                    marker=".",
                    linestyle="None",
                    label=f"{name} ({count})",
                )
            )
    ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(output_path)


def _select_samples(cache_root: Path, count: int) -> list[Path]:
    """Select samples preferring high map-token counts plus one zero-map sample."""

    files = sorted(cache_root.glob("**/sample_*.npz"))
    scored: list[tuple[int, Path]] = []
    for path in files:
        with np.load(path, allow_pickle=True) as data:
            scored.append((len(data["map_xy"]) if "map_xy" in data else 0, path))
    selected = [path for _, path in sorted(scored, key=lambda item: item[0], reverse=True)[:count]]
    zero = next((path for score, path in scored if score == 0), None)
    if zero is not None and zero not in selected:
        selected.append(zero)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output-dir", default="outputs/input_patch")
    parser.add_argument("--sample", action="append", default=[])
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--roi", type=float, nargs=4, default=(-25.0, -25.0, 25.0, 25.0))
    parser.add_argument("--max-points-per-sweep", type=int, default=45000)
    args = parser.parse_args()

    cache_root = Path(args.cache_root)
    output_dir = Path(args.output_dir)
    sample_paths = [Path(p) for p in args.sample] if args.sample else _select_samples(cache_root, args.count)
    for path in sample_paths:
        rel = path.relative_to(cache_root)
        output_name = "__".join(rel.with_suffix(".png").parts)
        plot_sample(
            path,
            output_dir / output_name,
            roi=tuple(args.roi),
            max_points_per_sweep=args.max_points_per_sweep,
        )


if __name__ == "__main__":
    main()
