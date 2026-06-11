from __future__ import annotations

import gradio as gr

_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#f0f4f8",
    body_background_fill_dark="#0f172a",
    block_background_fill="#ffffff",
    block_background_fill_dark="#1e293b",
    block_border_width="1px",
    block_border_color="#e2e8f0",
    block_border_color_dark="#334155",
    block_label_background_fill="#f8fafc",
    block_label_background_fill_dark="#1e293b",
    block_label_text_color="#475569",
    block_label_text_color_dark="#94a3b8",
    block_shadow="0 1px 3px 0 rgb(0 0 0 / 0.08), 0 1px 2px -1px rgb(0 0 0 / 0.08)",
    block_title_text_weight="600",
    input_background_fill="#f8fafc",
    input_background_fill_dark="#0f172a",
    input_border_color="#e2e8f0",
    input_border_color_dark="#334155",
    button_primary_background_fill="linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%)",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#f1f5f9",
    button_secondary_background_fill_hover="#e2e8f0",
    button_secondary_text_color="#1e293b",
    button_secondary_border_color="#e2e8f0",
)
