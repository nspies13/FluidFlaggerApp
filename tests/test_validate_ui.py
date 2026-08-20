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


def test_validate_column_changes_refresh_only_after_user_input():
    """Loading detected dropdown defaults must not hide the new dashboard."""
    app = build_ui()
    config = app.get_config_file()
    dropdown_ids = {
        component["props"].get("label"): component["id"]
        for component in config["components"]
        if component.get("type") == "dropdown"
    }
    column_dropdown_ids = {
        dropdown_ids["Ground-truth label"],
        dropdown_ids["Prediction probability"],
    }
    targets = {
        tuple(target)
        for dependency in config["dependencies"]
        for target in dependency.get("targets", [])
    }

    for component_id in column_dropdown_ids:
        assert (component_id, "input") in targets
        assert (component_id, "change") not in targets
