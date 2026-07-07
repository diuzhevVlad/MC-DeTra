#!/usr/bin/env python3
"""Create dark BEV GIFs for diverse validation scenarios."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from detra_repro.data.dataset import CachedWaymoDataset, collate_cached_waymo
from detra_repro.eval.paper_detection_ap import vehicle_confidence
from detra_repro.evaluate_detection_pr import _build_configs, _load_checkpoint, _load_models, _predict_batch
from detra_repro.losses import current_boxes_from_pred


@dataclass
class ScenarioPick:
    label: str
    rel_path: str
    segment: str
    sample_id: int
    num_vehicles: int
    moving_count: int
    static_count: int
    mean_moving_disp: float
    max_disp: float
    max_turn_deg: float


SCENARIOS = [
    "high_flow",
    "high_flow_dense",
    "high_flow_fast",
    "turning",
    "turning_dense",
    "hard_turn",
    "low_flow",
    "low_flow_sparse",
    "stationary_dense",
    "crowded",
    "crowded_moving",
    "sparse_motion",
    "fast_motion",
    "fast_turn",
    "mixed_static_moving",
    "balanced",
    "many_movers",
    "few_movers",
    "long_future",
    "medium_flow",
]


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an < 1.5 or bn < 1.5:
        return 0.0
    dot = float(np.clip(np.dot(a, b) / (an * bn), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def _stats_for_file(path: Path, cache_root: Path) -> dict | None:
    with np.load(path, allow_pickle=False) as data:
        gt_boxes = data["gt_boxes"].astype(np.float32)
        gt_future = data["gt_future"].astype(np.float32)
        gt_future_mask = data["gt_future_mask"].astype(bool)
        gt_types = data["gt_types"].astype(np.int64) if "gt_types" in data.files else np.ones(len(gt_boxes), dtype=np.int64)
        eligible = data["paper_vehicle_eligible"].astype(bool) if "paper_vehicle_eligible" in data.files else gt_types == 1
        sample_id = int(data["sample_id"]) if "sample_id" in data.files else -1

    mask = eligible & (gt_types == 1)
    if int(mask.sum()) == 0:
        return None

    boxes = gt_boxes[mask]
    future = gt_future[mask]
    future_mask = gt_future_mask[mask]
    disps = []
    turns = []
    for box, traj, valid in zip(boxes, future, future_mask, strict=False):
        valid_idx = np.flatnonzero(valid)
        if len(valid_idx) == 0:
            disps.append(0.0)
            turns.append(0.0)
            continue
        last = traj[valid_idx[-1], :2]
        start = box[:2]
        disp = float(np.linalg.norm(last - start))
        disps.append(disp)
        if len(valid_idx) >= 4:
            mid = traj[valid_idx[len(valid_idx) // 2], :2]
            turns.append(_angle_deg(mid - start, last - mid))
        else:
            turns.append(0.0)

    disp_arr = np.asarray(disps, dtype=np.float32)
    moving = disp_arr > 2.0
    rel = path.relative_to(cache_root).as_posix()
    return {
        "rel_path": rel,
        "segment": path.parent.name,
        "sample_id": sample_id,
        "num_vehicles": int(mask.sum()),
        "moving_count": int(moving.sum()),
        "static_count": int((~moving).sum()),
        "mean_moving_disp": float(disp_arr[moving].mean()) if bool(moving.any()) else 0.0,
        "max_disp": float(disp_arr.max()) if len(disp_arr) else 0.0,
        "max_turn_deg": float(max(turns) if turns else 0.0),
    }


def _scenario_score(label: str, stats: dict) -> float:
    n = stats["num_vehicles"]
    moving = stats["moving_count"]
    static = stats["static_count"]
    mean_disp = stats["mean_moving_disp"]
    max_disp = stats["max_disp"]
    turn = stats["max_turn_deg"]

    scores = {
        "high_flow": moving * 3.0 + mean_disp + 0.05 * n,
        "high_flow_dense": moving * 2.5 + 0.25 * n + mean_disp,
        "high_flow_fast": moving * 2.0 + max_disp * 1.8,
        "turning": turn * 2.5 + moving,
        "turning_dense": turn * 2.0 + 0.2 * n + moving,
        "hard_turn": turn * 3.5 + max_disp,
        "low_flow": static * 2.0 - moving * 4.0 - max_disp,
        "low_flow_sparse": -moving * 5.0 - max_disp - 0.05 * n,
        "stationary_dense": static * 2.5 + 0.2 * n - moving * 3.0,
        "crowded": n * 1.5 + moving * 0.5,
        "crowded_moving": n + moving * 2.0 + mean_disp,
        "sparse_motion": moving * 3.0 + mean_disp - 0.25 * n,
        "fast_motion": max_disp * 2.5 + mean_disp,
        "fast_turn": max_disp * 1.5 + turn * 2.0,
        "mixed_static_moving": min(moving, static) * 3.0 + 0.1 * n,
        "balanced": min(moving, static) * 2.0 + mean_disp - abs(moving - static),
        "many_movers": moving * 4.0 + 0.5 * mean_disp,
        "few_movers": (2.0 if 1 <= moving <= 3 else -20.0) + mean_disp - 0.1 * n,
        "long_future": mean_disp * 2.0 + max_disp,
        "medium_flow": -abs(moving - 8) * 1.5 + mean_disp + 0.05 * n,
    }
    return float(scores[label])


def select_scenarios(cache_root: Path, num: int, limit: int | None) -> list[ScenarioPick]:
    files = sorted(cache_root.glob("*/*.npz"))
    if limit:
        files = files[:limit]
    stats = []
    for idx, path in enumerate(files):
        item = _stats_for_file(path, cache_root)
        if item is not None:
            stats.append(item)
        if (idx + 1) % 2000 == 0:
            print(f"scanned {idx + 1}/{len(files)} samples", flush=True)

    picks: list[ScenarioPick] = []
    used_segments: set[str] = set()
    used_paths: set[str] = set()
    labels = SCENARIOS[:num]
    for label in labels:
        ranked = sorted(stats, key=lambda item: _scenario_score(label, item), reverse=True)
        chosen = None
        for item in ranked:
            if item["rel_path"] not in used_paths and item["segment"] not in used_segments:
                chosen = item
                break
        if chosen is None:
            for item in ranked:
                if item["rel_path"] not in used_paths:
                    chosen = item
                    break
        if chosen is None:
            continue
        used_paths.add(chosen["rel_path"])
        used_segments.add(chosen["segment"])
        picks.append(ScenarioPick(label=label, **chosen))
    return picks


def _world_to_px(xy: np.ndarray, roi: tuple[float, float, float, float], size: int) -> tuple[float, float]:
    xmin, xmax, ymin, ymax = roi
    x = (float(xy[0]) - xmin) / (xmax - xmin) * size
    y = size - (float(xy[1]) - ymin) / (ymax - ymin) * size
    return x, y


def _draw_box(draw: ImageDraw.ImageDraw, box: np.ndarray, roi: tuple[float, float, float, float], size: int, color: tuple[int, int, int], width: int = 1) -> None:
    x, y, dx, dy, yaw = [float(v) for v in box[:5]]
    corners = np.array([[dx / 2, dy / 2], [dx / 2, -dy / 2], [-dx / 2, -dy / 2], [-dx / 2, dy / 2]], dtype=np.float32)
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    pts = corners @ rot.T + np.array([x, y], dtype=np.float32)
    poly = [_world_to_px(pt, roi, size) for pt in pts]
    draw.line(poly + [poly[0]], fill=color, width=width)


def _draw_polyline(draw: ImageDraw.ImageDraw, points: np.ndarray, roi: tuple[float, float, float, float], size: int, color: tuple[int, int, int], width: int = 2) -> None:
    if len(points) < 2:
        return
    pts = [_world_to_px(pt[:2], roi, size) for pt in points]
    draw.line(pts, fill=color, width=width)


def _render_panel(
    sample: dict,
    frame_idx: int,
    roi: tuple[float, float, float, float],
    panel_size: int,
    title: str,
    pred_boxes: np.ndarray | None,
    pred_trajs: np.ndarray | None,
    pred_scores: np.ndarray | None,
) -> Image.Image:
    img = Image.new("RGB", (panel_size, panel_size + 34), (9, 12, 17))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    draw.text((10, 8), title, fill=(226, 232, 240), font=font)
    origin_y = 34
    crop = Image.new("RGB", (panel_size, panel_size), (7, 10, 15))
    cdraw = ImageDraw.Draw(crop)

    points = sample.get("points")
    if points is not None and len(points):
        xy = points[:, :2]
        keep = (xy[:, 0] >= roi[0]) & (xy[:, 0] <= roi[1]) & (xy[:, 1] >= roi[2]) & (xy[:, 1] <= roi[3])
        xy = xy[keep]
        if len(xy) > 120_000:
            stride = max(1, len(xy) // 120_000)
            xy = xy[::stride]
        for pt in xy:
            px, py = _world_to_px(pt, roi, panel_size)
            cdraw.point((px, py), fill=(42, 51, 64))

    gt_types = sample.get("gt_types", np.ones(len(sample["gt_boxes"]), dtype=np.int64))
    eligible = sample.get("paper_vehicle_eligible", gt_types == 1).astype(bool)
    gt_mask = eligible & (gt_types == 1)
    for box, future, valid in zip(sample["gt_boxes"][gt_mask], sample["gt_future"][gt_mask], sample["gt_future_mask"][gt_mask], strict=False):
        _draw_box(cdraw, box, roi, panel_size, (120, 144, 178), width=1)
        valid_idx = np.flatnonzero(valid[: frame_idx + 1])
        if len(valid_idx) >= 2:
            _draw_polyline(cdraw, future[valid_idx, :2], roi, panel_size, (74, 222, 128), width=2)

    if pred_boxes is not None and pred_trajs is not None and pred_scores is not None:
        order = np.argsort(pred_scores)
        for idx in order:
            box = pred_boxes[idx]
            score = float(pred_scores[idx])
            shade = int(120 + 120 * min(1.0, score))
            _draw_box(cdraw, box, roi, panel_size, (252, shade, 94), width=1)
            horizon = min(frame_idx + 1, pred_trajs.shape[1])
            if horizon >= 2:
                _draw_polyline(cdraw, pred_trajs[idx, :horizon, :2], roi, panel_size, (248, 113, 113), width=2)

    img.paste(crop, (0, origin_y))
    return img


def _sample_to_plain(sample: dict) -> dict:
    out = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.detach().cpu().numpy()
        else:
            out[key] = value
    return out


def render_gifs(args: argparse.Namespace, picks: list[ScenarioPick]) -> None:
    dataset = CachedWaymoDataset(args.cache)
    files = [Path(path).relative_to(args.cache).as_posix() for path in dataset.files]
    index_by_rel = {rel: idx for idx, rel in enumerate(files)}

    ckpt = _load_checkpoint(args.checkpoint, args.device)
    data_cfg, model_cfg = _build_configs(ckpt, {}, dataset)
    model_cfg = replace(model_cfg, num_queries=args.num_queries)
    lidar_encoder, head, map_encoder, model = _load_models(ckpt, data_cfg, model_cfg, args.norm_type, args.device)
    lidar_encoder.eval()
    model.eval()
    head.eval()
    if map_encoder is not None:
        map_encoder.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = []
    roi = tuple(float(v) for v in args.roi)
    for pick in picks:
        ds_idx = index_by_rel.get(pick.rel_path)
        if ds_idx is None:
            print(f"skip missing sample {pick.rel_path}", flush=True)
            continue
        raw_sample = dataset[ds_idx]
        batch = collate_cached_waymo([raw_sample], max_points=args.max_points)
        with torch.inference_mode():
            pred = _predict_batch(
                batch,
                lidar_encoder,
                head,
                map_encoder,
                model,
                data_cfg,
                model_cfg,
                args.device,
                args.proposal_nms_radius_m,
                args.proposal_nms_mode,
                args.proposal_center_mode,
                args.proposal_peak_filter,
            )
        object_logits = pred["object_logits"].detach().float().cpu().numpy()
        class_logits = pred["class_logits"].detach().float().cpu().numpy()
        scores = vehicle_confidence(object_logits, class_logits)[0]
        boxes = current_boxes_from_pred(pred)[0].detach().cpu().numpy()
        poses = pred["poses"][0].detach().cpu().numpy()
        mode_logits = pred["mode_logits"][0].detach().cpu().numpy()
        modes = mode_logits.argmax(axis=1)
        trajs = poses[np.arange(len(poses)), modes, :, :2]
        keep = np.flatnonzero(scores >= args.score_threshold)
        if len(keep) == 0:
            keep = np.argsort(scores)[-args.max_predictions :]
        if len(keep) > args.max_predictions:
            keep = keep[np.argsort(scores[keep])[-args.max_predictions :]]
        keep = keep[np.argsort(scores[keep])[::-1]]

        sample = _sample_to_plain(raw_sample)
        frames = []
        for frame_idx in range(args.frames):
            gt_panel = _render_panel(
                sample,
                frame_idx,
                roi,
                args.panel_size,
                f"{pick.label} | GT future",
                None,
                None,
                None,
            )
            pred_panel = _render_panel(
                sample,
                frame_idx,
                roi,
                args.panel_size,
                f"Predictions | top {len(keep)} vehicle-scored",
                boxes[keep],
                trajs[keep],
                scores[keep],
            )
            canvas = Image.new("RGB", (args.panel_size * 2 + 8, args.panel_size + 34), (3, 6, 12))
            canvas.paste(gt_panel, (0, 0))
            canvas.paste(pred_panel, (args.panel_size + 8, 0))
            frames.append(canvas)

        out_name = f"{len(metadata):02d}_{pick.label}_{pick.segment}_{pick.sample_id}.gif"
        out_path = args.output_dir / out_name
        imageio.mimsave(out_path, frames, duration=1.0 / args.fps)
        item = asdict(pick)
        item["gif"] = out_name
        item["kept_predictions"] = int(len(keep))
        item["score_threshold"] = float(args.score_threshold)
        metadata.append(item)
        print(f"wrote {out_path}", flush=True)

    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _load_picks(path: Path) -> list[ScenarioPick]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [ScenarioPick(**item) for item in payload]


def _write_picks(path: Path, picks: Iterable[ScenarioPick]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(item) for item in picks], indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/scenario_gifs"))
    parser.add_argument("--selected-json", type=Path)
    parser.add_argument("--write-selected-json", type=Path)
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--num-gifs", type=int, default=20)
    parser.add_argument("--scan-limit", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-queries", type=int, default=600)
    parser.add_argument("--max-points", type=int, default=750000)
    parser.add_argument("--norm-type", choices=["batch", "group"], default="group")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--proposal-center-mode", choices=["anchor", "regression"], default="regression")
    parser.add_argument("--proposal-nms-mode", choices=["center", "rotated"], default="rotated")
    parser.add_argument("--proposal-nms-radius-m", type=float, default=0.1)
    parser.add_argument("--proposal-peak-filter", action="store_true")
    parser.add_argument("--score-threshold", type=float, default=0.30)
    parser.add_argument("--max-predictions", type=int, default=40)
    parser.add_argument("--panel-size", type=int, default=620)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--roi", nargs=4, type=float, default=(-75.0, 75.0, -75.0, 75.0))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    picks = _load_picks(args.selected_json) if args.selected_json else select_scenarios(args.cache, args.num_gifs, args.scan_limit)
    if args.write_selected_json:
        _write_picks(args.write_selected_json, picks)
    if args.select_only:
        return
    if args.checkpoint is None:
        raise SystemExit("--checkpoint is required unless --select-only is used")
    render_gifs(args, picks[: args.num_gifs])


if __name__ == "__main__":
    main()
