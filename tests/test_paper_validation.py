"""Regression tests for the manuscript validation operating point."""

import numpy as np
import pytest

from paper.validate_models import THRESHOLD, operating_point
from src.inference import _CONTAM_THRESHOLD
from src.validation import DEFAULT_THRESHOLD


def test_production_and_manuscript_defaults_stay_aligned():
    assert _CONTAM_THRESHOLD == pytest.approx(0.25)
    assert DEFAULT_THRESHOLD == pytest.approx(0.25)
    assert THRESHOLD == pytest.approx(0.25)


def test_manuscript_metrics_use_the_binary_production_threshold():
    """A score exactly at 0.25 is positive and contributes to F1."""
    metrics = operating_point(
        np.array([1, 1, 0, 0]),
        np.array([0.25, 0.24, 0.25, 0.20]),
    )

    assert THRESHOLD == pytest.approx(0.25)
    assert {key: metrics[key] for key in ("tp", "fp", "tn", "fn")} == {
        "tp": 1,
        "fp": 1,
        "tn": 1,
        "fn": 1,
    }
    assert metrics["sensitivity"] == pytest.approx(0.50)
    assert metrics["specificity"] == pytest.approx(0.50)
    assert metrics["ppv"] == pytest.approx(0.50)
    assert metrics["npv"] == pytest.approx(0.50)
    assert metrics["f1"] == pytest.approx(0.50)
