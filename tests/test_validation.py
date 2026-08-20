"""Tests for reviewed-prediction validation helpers."""

import pandas as pd
import pytest

from src.validation import (
    DEFAULT_THRESHOLD,
    ValidationDataError,
    _threshold_curve,
    build_validation_payload,
    find_label_columns,
    find_score_columns,
    operating_metrics,
    prepare_validation_data,
)


def test_reviewed_predictions_are_detected_and_mapped():
    df = pd.DataFrame(
        {
            "max_retrospective_prob": [0.91, 0.12, 0.72, 0.48],
            "prob_NS_Realtime": [0.82, 0.14, 0.55, 0.31],
            "human_label": ["Contaminated", "Real", "Equivocal", "Real"],
            "label_timestamp": ["2026-08-11T10:00:00"] * 4,
        }
    )

    assert find_label_columns(df)[0] == "human_label"
    assert "label_timestamp" not in find_label_columns(df)
    assert find_score_columns(df)[0] == "max_retrospective_prob"

    prepared = prepare_validation_data(
        df,
        score_column="max_retrospective_prob",
        label_column="human_label",
    )
    assert prepared.labels.tolist() == [1, 0, 0]
    assert prepared.scores.tolist() == [0.91, 0.12, 0.48]
    assert prepared.excluded_rows == 1


def test_equivocal_policy_can_map_to_positive_or_negative():
    df = pd.DataFrame(
        {
            "prob_CBC_Retrospective": [0.9, 0.1, 0.7],
            "human_label": ["Contaminated", "Real", "Equivocal"],
        }
    )

    positive = prepare_validation_data(
        df, "prob_CBC_Retrospective", "human_label", "Treat equivocal as contaminated"
    )
    negative = prepare_validation_data(
        df, "prob_CBC_Retrospective", "human_label", "Treat equivocal as real"
    )

    assert positive.labels.tolist() == [1, 0, 1]
    assert negative.labels.tolist() == [1, 0, 0]


def test_operating_metrics_calculates_all_requested_measures():
    labels = pd.Series([1, 1, 0, 0]).to_numpy()
    scores = pd.Series([0.9, 0.4, 0.8, 0.3]).to_numpy()

    metrics = operating_metrics(labels, scores, threshold=0.50)

    assert {key: metrics[key] for key in ("tp", "fp", "tn", "fn")} == {
        "tp": 1,
        "fp": 1,
        "tn": 1,
        "fn": 1,
    }
    assert metrics["sensitivity"] == pytest.approx(0.5)
    assert metrics["specificity"] == pytest.approx(0.5)
    assert metrics["ppv"] == pytest.approx(0.5)
    assert metrics["npv"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)


def test_operating_metrics_uses_inclusive_default_threshold_boundary():
    labels = pd.Series([1, 1, 0]).to_numpy()
    scores = pd.Series([0.26, 0.25, 0.24]).to_numpy()

    metrics = operating_metrics(labels, scores)

    assert DEFAULT_THRESHOLD == pytest.approx(0.25)
    # A probability exactly at the Validate operating point is positive.
    assert {key: metrics[key] for key in ("tp", "fp", "tn", "fn")} == {
        "tp": 2,
        "fp": 0,
        "tn": 1,
        "fn": 0,
    }


def test_payload_has_curves_auc_calibration_and_binary_table():
    df = pd.DataFrame(
        {
            "max_realtime_prob": [0.95, 0.83, 0.20, 0.08],
            "human_label": ["T", 1, "F", 0],
        }
    )

    payload = build_validation_payload(df, "max_realtime_prob", "human_label")

    assert payload["auc"]["roc"] == pytest.approx(1.0)
    assert payload["auc"]["pr"] == pytest.approx(1.0)
    assert payload["metrics"]["tp"] == 2
    assert payload["metrics"]["tn"] == 2
    assert payload["roc"]
    assert payload["pr"]
    assert payload["calibration"]
    assert all("threshold" in point for point in payload["roc"])
    assert payload["threshold"] == pytest.approx(0.25)
    assert payload["threshold_rule"] == "score >= threshold"

    # The compact operating arrays power browser-side threshold updates.
    thresholds = payload["operating"]["thresholds"]
    selected = min(index for index, value in enumerate(thresholds) if value >= 0.25)
    assert payload["operating"]["tp"][selected] == 2
    assert payload["operating"]["fp"][selected] == 0


def test_compact_operating_points_match_every_slider_interval_and_boundary():
    df = pd.DataFrame(
        {
            "max_realtime_prob": [0.95, 0.83, 0.50, 0.25, 0.08],
            "human_label": ["Contaminated", "Real", "Contaminated", "Real", "Real"],
        }
    )
    payload = build_validation_payload(df, "max_realtime_prob", "human_label")
    prepared = prepare_validation_data(df, "max_realtime_prob", "human_label")
    operating = payload["operating"]

    # These values cover each side of an observed score as well as the
    # inclusive 0.250 boundary a 0.001-step slider can produce.
    for threshold in (0.0, 0.079, 0.08, 0.081, 0.249, 0.25, 0.251, 0.429, 0.43, 0.499, 0.50, 0.501, 0.829, 0.83, 0.831, 0.95, 0.951, 1.0):
        selected = min(
            index
            for index, cached_threshold in enumerate(operating["thresholds"])
            if cached_threshold >= threshold
        )
        actual = operating_metrics(prepared.labels, prepared.scores, threshold)
        assert operating["tp"][selected] == actual["tp"]
        assert operating["fp"][selected] == actual["fp"]


def test_payload_requires_both_ground_truth_classes():
    df = pd.DataFrame(
        {
            "max_realtime_prob": [0.9, 0.8],
            "human_label": ["Contaminated", "Contaminated"],
        }
    )

    with pytest.raises(ValidationDataError, match="at least one Real"):
        build_validation_payload(df, "max_realtime_prob", "human_label")


def test_threshold_curve_keeps_inclusive_zero_score_endpoint_selectable():
    labels = pd.Series([0, 1]).to_numpy()
    scores = pd.Series([0.0, 0.0]).to_numpy()

    roc, pr = _threshold_curve(labels, scores)

    assert roc[0] == {"fpr": 0.0, "tpr": 0.0, "threshold": 1.0}
    assert roc[-1] == {"fpr": 1.0, "tpr": 1.0, "threshold": 0.0}
    assert pr[-1] == {"recall": 1.0, "precision": 0.5}


def test_threshold_curve_uses_nonselectable_origin_when_a_score_is_one():
    labels = pd.Series([0, 1]).to_numpy()
    scores = pd.Series([1.0, 1.0]).to_numpy()

    roc, pr = _threshold_curve(labels, scores)

    assert roc[0] == {"fpr": 0.0, "tpr": 0.0, "threshold": None}
    assert roc[1] == {"fpr": 1.0, "tpr": 1.0, "threshold": 1.0}
    assert roc[-1] == {"fpr": 1.0, "tpr": 1.0, "threshold": 0.0}
    assert pr[-1] == {"recall": 1.0, "precision": 0.5}
