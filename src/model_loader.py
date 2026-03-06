"""
Lazy model loader with HuggingFace Hub backend.

Models are downloaded from HF Hub on first request and cached in a
module-level dict so subsequent calls in the same process are instant.
Also supports loading custom models from uploaded .joblib files.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Optional

import joblib


def _load_joblib(path):
    """Load a joblib file, suppressing sklearn version-mismatch warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        return joblib.load(path)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HF_REPO_ID = os.environ.get("HF_MODEL_REPO", "nspies13/fluidflagger-models")

# Module-level cache: model_key → loaded model dict
_cache: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------

def bmp_classification_key(fluid: str, timing: str) -> str:
    """e.g. 'bmp_NS_Realtime'"""
    return f"bmp_{fluid}_{timing}"


def bmp_mix_ratio_key(fluid: str) -> str:
    return f"bmp_{fluid}_mix_ratio"


def cbc_classification_key(timing: str) -> str:
    return f"cbc_CBC_{timing}"


def cbc_mix_ratio_key() -> str:
    return "cbc_CBC_mix_ratio"


def filename_for_key(key: str) -> str:
    return f"{key}.joblib"

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _download_from_hub(key: str) -> Optional[dict]:
    """Download a model from HF Hub and return the loaded dict, or None if unavailable."""
    try:
        from huggingface_hub import hf_hub_download
        filename = filename_for_key(key)
        local_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            repo_type="model",
        )
        return _load_joblib(local_path)
    except Exception:
        return None


def get_model(key: str) -> Optional[dict]:
    """
    Return a loaded model dict for the given key, using the module-level cache.
    Downloads from HF Hub on first miss. Returns None if unavailable.
    """
    if key not in _cache:
        model = _download_from_hub(key)
        if model is not None:
            _cache[key] = model
    return _cache.get(key)


def load_from_file(path: str | Path) -> dict:
    """Load a model dict directly from a local .joblib file (e.g. user upload)."""
    return _load_joblib(path)


def cache_model(key: str, model_dict: dict) -> None:
    """Insert a model dict into the cache under the given key."""
    _cache[key] = model_dict


def clear_cache() -> None:
    _cache.clear()


def model_key(model_dict: dict) -> str:
    """Canonical filename stem for a model, e.g. 'bmp_NS_Realtime'."""
    panel = model_dict["panel"]
    fluid = model_dict["fluid"]
    typ = model_dict["type"]
    task = model_dict["task"]
    if task == "mix_ratio":
        return f"{panel}_{fluid}_mix_ratio"
    return f"{panel}_{fluid}_{typ}"

def load_models_from_dir(directory: str | Path) -> dict[str, dict]:
    """
    Load all .joblib files from a directory into the cache.
    Returns a dict of key → model_dict for newly loaded models.
    """
    directory = Path(directory)
    loaded = {}
    for path in sorted(directory.glob("*.joblib")):
        key = path.stem
        try:
            model = _load_joblib(path)
            _cache[key] = model
            loaded[key] = model
        except Exception as e:
            print(f"Warning: could not load {path}: {e}")
    return loaded
