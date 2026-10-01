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


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _load_raw(path: Path) -> dict:
    """YAML file; `extends: other.yaml` (relative to this file) is loaded first and overridden."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    parent = raw.pop("extends", None)
    return _merge(_load_raw((path.parent / parent).resolve()), raw) if parent else raw


def load_config(path: str | Path | None = None) -> Section:
    """--config, else $EGYSPEECH_CONFIG, else configs/config.yaml (relative paths: cwd, then repo root)."""
    path = Path(path or os.environ.get("EGYSPEECH_CONFIG") or DEFAULT_CONFIG)
    if not path.is_absolute() and not path.exists():
        path = REPO_ROOT / path
    cfg = Section(_walk(_load_raw(Path(path))))
    cfg["_path"] = str(Path(path).resolve())
    return cfg


def resolve_repo_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p
