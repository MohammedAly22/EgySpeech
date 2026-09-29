"""Configuration: YAML with ${VAR} / ${VAR:-default} expansion, attribute access."""

import os
import re
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "config.yaml"

_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand(value: str) -> str:
    def repl(m: re.Match[str]) -> str:
        name, default = m.group(1), m.group(2)
        env = os.environ.get(name)
        if env:
            return env
        if default is not None:
            return default
        raise ValueError(f"environment variable {name} is not set (used in config value {value!r})")

    return _VAR.sub(repl, value)


def _walk(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _walk(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk(v) for v in obj]
    if isinstance(obj, str):
        return _expand(obj)
    return obj


class Section(dict):
    """dict with attribute access (cfg.segmentation.min_sec)."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(f"config has no field {key!r}") from exc
        return Section(value) if isinstance(value, dict) and not isinstance(value, Section) else value


def load_config(path: str | Path | None = None) -> Section:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = Section(_walk(raw))
    cfg["_path"] = str(Path(path).resolve())
    return cfg


def resolve_repo_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p
