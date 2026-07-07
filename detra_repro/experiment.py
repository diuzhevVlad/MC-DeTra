"""Small experiment config and logging utilities."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Sequence
from typing import Any

from detra_repro.secrets import ensure_comet_api_key


DEFAULT_COMET_PROJECT = "mc-detra"
DEFAULT_COMET_WORKSPACE = "your-comet-workspace"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def flatten_experiment_config(payload: dict[str, Any]) -> dict[str, Any]:
    """Accept either flat JSON or grouped train/model/logging sections."""

    if "config" in payload and isinstance(payload["config"], dict):
        payload = payload["config"]
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, dict) and key in {
            "paths",
            "data",
            "training",
            "model",
            "losses",
            "evaluation",
            "logging",
            "runtime",
            "checkpointing",
        }:
            flat.update(value)
        else:
            flat[key] = value
    return flat


def load_experiment_config(
    path: str | Path | None,
    *,
    _seen: frozenset[Path] = frozenset(),
) -> dict[str, Any]:
    """Load a flattened experiment config, resolving ``extends`` bases.

    A config may set ``"extends": "base.json"`` or ``"extends": ["a.json",
    "b.json"]`` to inherit from one or more base configs. Bases are resolved
    relative to the referencing file, applied left to right, and the current
    file's own keys win on conflict. Inheritance is recursive; cycles raise.
    """

    if path is None:
        return {}
    resolved = Path(path).resolve()
    if resolved in _seen:
        chain = " -> ".join(str(p) for p in (*_seen, resolved))
        raise ValueError(f"Cyclic experiment-config extends: {chain}")
    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    bases = payload.pop("extends", None)
    if isinstance(bases, (str, Path)):
        bases = [bases]

    merged: dict[str, Any] = {}
    for base in bases or []:
        base_path = resolved.parent / base
        merged.update(load_experiment_config(base_path, _seen=_seen | {resolved}))
    merged.update(flatten_experiment_config(payload))
    return merged


class ExperimentLogger:
    """Local JSONL logger with optional Comet mirroring."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        config: dict[str, Any],
        comet_enabled: bool = False,
        comet_project: str | None = None,
        comet_workspace: str | None = None,
        comet_run_name: str | None = None,
        comet_tags: list[str] | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.config_path = self.run_dir / "config.json"
        self.config = to_jsonable(config)
        self.comet = None

        self.config_path.write_text(
            json.dumps(
                {
                    "created_utc": utc_now_iso(),
                    "command": sys.argv,
                    "config": self.config,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        if comet_enabled:
            credential_source = ensure_comet_api_key()
            if credential_source is None:
                raise RuntimeError(
                    "Comet logging was requested, but no API key was found. "
                    "Set COMET_API_KEY or add COMET_API_KEY=... to .env."
                )
            try:
                from comet_ml import Experiment
            except ImportError as exc:
                raise RuntimeError(
                    "Comet logging was requested, but comet_ml is not installed"
                ) from exc
            project_name = comet_project or os.environ.get("COMET_PROJECT_NAME") or DEFAULT_COMET_PROJECT
            workspace = comet_workspace or os.environ.get("COMET_WORKSPACE") or DEFAULT_COMET_WORKSPACE
            self.comet = Experiment(project_name=project_name, workspace=workspace)
            if comet_run_name:
                self.comet.set_name(comet_run_name)
            for tag in comet_tags or []:
                self.comet.add_tag(tag)
            self.comet.log_parameters(self.config)
            self.log_event(
                "comet_enabled",
                {
                    "project_name": project_name,
                    "workspace": workspace,
                    "credential_source": credential_source,
                },
            )

        self.log_event("run_start", {"run_dir": str(self.run_dir)})

    def log_event(self, event: str, payload: dict[str, Any] | None = None) -> None:
        record = {
            "time_utc": utc_now_iso(),
            "event": event,
            "payload": to_jsonable(payload or {}),
        }
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def log_metrics(
        self,
        split: str,
        step: int,
        metrics: dict[str, Any],
        comet_keys: Sequence[str] | None = None,
    ) -> None:
        clean = {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        }
        # Local JSONL always keeps the full payload.
        self.log_event(f"{split}_metrics", {"step": step, "metrics": clean})
        if self.comet is not None:
            # ``comet_keys`` restricts the noisy per-step train series to a curated
            # headline set; other splits (val) mirror everything.
            comet_clean = (
                {k: v for k, v in clean.items() if k in set(comet_keys)}
                if comet_keys is not None
                else clean
            )
            self.comet.log_metrics({f"{split}/{k}": v for k, v in comet_clean.items()}, step=step)

    def write_summary(self, payload: dict[str, Any]) -> None:
        self.summary_path.write_text(
            json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.log_event("run_summary", payload)

    def close(self) -> None:
        if self.comet is not None:
            self.comet.end()
