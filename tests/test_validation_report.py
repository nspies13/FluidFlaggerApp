"""Tests for exportable Validate-tab reports."""

from pathlib import Path

import pandas as pd

from src.validation import build_validation_payload
from src.validation_report import build_validation_report_files


def test_build_validation_report_files_creates_self_contained_html_and_pdf(tmp_path):
    dataframe = pd.DataFrame(
        {
            "human_label": ["Real", "Contaminated", "Real", "Contaminated"],
            "max_retrospective_prob": [0.08, 0.92, 0.36, 0.78],
        }
    )
    payload = build_validation_payload(
        dataframe,
        score_column="max_retrospective_prob",
        label_column="human_label",
        threshold=0.5,
    )

    html_path, pdf_path = build_validation_report_files(payload, output_dir=tmp_path)

    html_file = Path(html_path)
    pdf_file = Path(pdf_path)
    assert html_file.is_file()
    assert pdf_file.is_file()
    assert pdf_file.read_bytes().startswith(b"%PDF-")

    report_html = html_file.read_text(encoding="utf-8")
    assert "FluidFlagger Validation Report" in report_html
    assert "2 x 2 classification table" in report_html
    assert "data:image/png;base64," in report_html
    assert "max_retrospective_prob" in report_html
    assert "Performance at threshold 0.500" in report_html
    assert len(list(tmp_path.glob("*.png"))) == 3
