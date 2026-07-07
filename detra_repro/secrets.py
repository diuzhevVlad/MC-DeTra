"""Minimal local secret loading."""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> str | None:
    """Load simple KEY=VALUE pairs from a local .env file without overwriting env."""

    dotenv = Path(path)
    if not dotenv.exists():
        return None
    for raw_line in dotenv.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
    return str(dotenv)


def ensure_comet_api_key() -> str | None:
    """Load .env if needed and return where COMET_API_KEY came from."""

    if os.environ.get("COMET_API_KEY"):
        return "env:COMET_API_KEY"
    source = load_dotenv()
    if os.environ.get("COMET_API_KEY"):
        return source or ".env"
    return None
