"""Focused checks for SHAP helper safeguards."""

from src.shap_tab import _UPLOAD_SENTINEL, _compute_shap_png


def test_custom_shap_model_requires_a_model_file_before_reading_data():
    result = _compute_shap_png(_UPLOAD_SENTINEL, None, "does-not-need-to-exist.csv")

    assert result == (None, None, "⚠️ Please upload a custom .joblib file.")
