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

import gradio as gr

from src.api import api as fastapi_app
from src.ui import build_ui

# Build the Gradio Blocks UI
demo = build_ui()

# Mount FastAPI inside Gradio so both share one process and one port.
# Routes:  /api/health, /api/bmp/predict, /api/cbc/predict, ...
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    demo.launch(server_port=7860, server_name="0.0.0.0")
