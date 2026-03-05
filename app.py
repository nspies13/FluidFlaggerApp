"""
FluidFlagger entrypoint for Hugging Face Spaces.

HF Spaces expects app.py as the entry point.
This file:
  1. Builds the Gradio UI (src/ui.py)
  2. Mounts the FastAPI sub-app (src/api.py) at /api/
  3. Launches on the port HF Spaces expects (7860)

Local dev:
    python app.py
    # or
    uvicorn app:app --reload --port 7860
"""

from pathlib import Path

import gradio as gr

from src.api import api as fastapi_app
from src.model_loader import (
    bmp_classification_key,
    bmp_mix_ratio_key,
    cbc_classification_key,
    cbc_mix_ratio_key,
    get_model,
    load_models_from_dir,
)
from src.simulate import get_fluid_names
from src.ui import build_ui

_ALL_FLUIDS = get_fluid_names()
_TIMINGS = ("Realtime", "Retrospective")

_ALL_MODEL_KEYS = (
    [bmp_classification_key(f, t) for f in _ALL_FLUIDS for t in _TIMINGS]
    + [bmp_mix_ratio_key(f) for f in _ALL_FLUIDS]
    + [cbc_classification_key(t) for t in _TIMINGS]
    + [cbc_mix_ratio_key()]
)


def _prefetch_models():
    """Download all models from HF Hub into the in-process cache."""
    ok, failed = 0, 0
    for key in _ALL_MODEL_KEYS:
        m = get_model(key)
        if m is not None:
            ok += 1
        else:
            failed += 1
    print(f"Model prefetch complete: {ok} loaded, {failed} unavailable.")


# Fall back to local models when running outside HF Spaces (dev mode)
_local_models = Path("models")
if _local_models.is_dir():
    _n = len(load_models_from_dir(_local_models))
    print(f"Loaded {_n} local model(s) into cache.")
else:
    # On HF Spaces there are no local files — prefetch from Hub at startup
    _prefetch_models()

# Build the Gradio Blocks UI; register _prefetch_models as the load handler
# inside the Blocks context so it fires on every new browser session.
demo = build_ui(on_load=_prefetch_models)

# Mount FastAPI inside Gradio so both share one process and one port.
# Routes:  /api/health, /api/bmp/predict, /api/cbc/predict, ...
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    demo.launch(server_port=7860, server_name="0.0.0.0")
