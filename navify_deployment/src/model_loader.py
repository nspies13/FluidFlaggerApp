"""
Local-only BMP model loader for the Navify Algorithm Suite container.

The Navify image must be self-contained: no Hugging Face Hub fallback, no user
uploads, and no runtime model downloads. Readiness is true only when all baked
BMP model artifacts have been loaded.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import joblib

BMP_FLUIDS = (
    "NS",
    "LR",
    "D5NS",
    "D5LR",
    "D5W",
    "D5halfNSwK",
    "D5halfNS",
    "halfNS",
    "Water",
)
BMP_TIMINGS = ("Realtime", "Retrospective")


def bmp_classification_key(fluid: str, timing: str) -> str:
    return f"bmp_{fluid}_{timing}"


def bmp_mix_ratio_key(fluid: str) -> str:
    return f"bmp_{fluid}_mix_ratio"


EXPECTED_BMP_MODEL_KEYS = tuple(
    [bmp_classification_key(fluid, timing) for fluid in BMP_FLUIDS for timing in BMP_TIMINGS]
    + [bmp_mix_ratio_key(fluid) for fluid in BMP_FLUIDS]
)
EXPECTED_BMP_MODEL_COUNT = len(EXPECTED_BMP_MODEL_KEYS)

_cache: dict[str, dict] = {}


def _load_joblib(path: Path) -> dict:
    """Load a joblib artifact while suppressing sklearn pickle-version noise."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        return joblib.load(path)


def filename_for_key(key: str) -> str:
    return f"{key}.joblib"


def get_model(key: str) -> Optional[dict]:
    """Return a previously loaded local model, or None if it was not loaded."""
    return _cache.get(key)


def cache_model(key: str, model_dict: dict) -> None:
    """Insert a model dict into the in-process cache. Used by tests only."""
    _cache[key] = model_dict


def clear_cache() -> None:
    _cache.clear()


def model_key(model_dict: dict) -> str:
    panel = model_dict["panel"]
    fluid = model_dict["fluid"]
    typ = model_dict["type"]
    task = model_dict["task"]
    if task == "mix_ratio":
        return f"{panel}_{fluid}_mix_ratio"
    return f"{panel}_{fluid}_{typ}"


def load_models_from_dir(directory: str | Path) -> dict[str, dict]:
    """Load all expected BMP .joblib files from a local directory."""
    directory = Path(directory)
    loaded: dict[str, dict] = {}
    for key in EXPECTED_BMP_MODEL_KEYS:
        path = directory / filename_for_key(key)
        if not path.exists():
            continue
        model = _load_joblib(path)
        _cache[key] = model
        loaded[key] = model
    return loaded


def missing_model_keys() -> list[str]:
    return [key for key in EXPECTED_BMP_MODEL_KEYS if key not in _cache]


def load_required_models_from_dir(directory: str | Path) -> dict[str, dict]:
    """Load BMP models and raise if any expected artifact is absent or unloadable."""
    loaded = load_models_from_dir(directory)
    missing = missing_model_keys()
    if missing:
        raise RuntimeError(
            f"Missing required BMP model artifacts: {', '.join(missing)}"
        )
    if len(loaded) != EXPECTED_BMP_MODEL_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_BMP_MODEL_COUNT} BMP models, loaded {len(loaded)}"
        )
    return loaded
