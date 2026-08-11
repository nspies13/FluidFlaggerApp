"""Regression tests for Validate's integrated SHAP layout."""

from src.ui import build_ui


def test_validate_places_shap_after_the_dashboard_without_a_second_csv_upload():
    app = build_ui()
    config = app.get_config_file()
    components = config["components"]

    image_labels = [
        component.get("props", {}).get("label")
        for component in components
        if component.get("type") == "image"
    ]
    file_labels = [
        component.get("props", {}).get("label")
        for component in components
        if component.get("type") == "file"
    ]
    html_values = [
        str(component.get("props", {}).get("value", ""))
        for component in components
        if component.get("type") == "html"
    ]

    assert "SHAP Feature Importance" in image_labels
    assert "Upload CSV" not in file_labels
    assert not any("Explain model behavior" in value for value in html_values)


def test_review_and_validate_do_not_expose_equivocal_controls():
    app = build_ui()
    components = app.get_config_file()["components"]

    button_values = [
        component.get("props", {}).get("value")
        for component in components
        if component.get("type") == "button"
    ]
    radio_labels = [
        component.get("props", {}).get("label")
        for component in components
        if component.get("type") == "radio"
    ]

    assert "Equivocal" not in button_values
    assert "Equivocal labels" not in radio_labels
