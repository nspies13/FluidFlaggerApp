"""Utilities for evaluating reviewed FluidFlagger prediction files.

The Validate tab consumes the CSV produced by the Predict and Review tabs:
prediction probabilities are retained by Review and the reviewer adds a
``human_label`` column.  This module keeps the parsing and metric calculations
independent of the UI so they can be tested and reused.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


# Validate reports the app's binary operating point: probabilities at or above
# 0.50 are positive. Keep the numeric value local so validation remains usable
# independently of the prediction UI.
DEFAULT_THRESHOLD = 0.50
MAX_RENDER_CURVE_POINTS = 2_000


class ValidationDataError(ValueError):
    """Raised when an uploaded validation file cannot support the analysis."""


@dataclass(frozen=True)
class PreparedValidationData:
    """Cleaned binary labels and probabilities ready for performance analysis."""

    labels: np.ndarray
    scores: np.ndarray
    total_rows: int
    excluded_rows: int
    excluded_missing_or_invalid_score: int
    excluded_unusable_label: int


_POSITIVE_LABELS = {
    "1",
    "true",
    "t",
    "yes",
    "y",
    "positive",
    "pos",
    "contaminated",
    "contamination",
    "flagged",
}
_NEGATIVE_LABELS = {
    "0",
    "false",
    "f",
    "no",
    "n",
    "negative",
    "neg",
    "real",
    "uncontaminated",
    "normal",
}
_EQUIVOCAL_LABELS = {
    "equivocal",
    "indeterminate",
    "uncertain",
    "unknown",
    "inconclusive",
}

_LABEL_NAME_HINTS = (
    "human_label",
    "ground_truth",
    "groundtruth",
    "label",
    "target",
    "outcome",
    "truth",
    "expert_review_prediction",
)
_SCORE_NAME_HINTS = ("prob", "probability", "score", "risk", "confidence")


def _normalise_text(value: Any) -> str:
    """Return a lower-case, whitespace-normalised representation of a label."""
    return " ".join(str(value).strip().lower().split())


def _column_name_key(column: Any) -> str:
    return _normalise_text(column).replace("-", "_").replace(" ", "_")


def _ordered_with_preferred(columns: list[str], preferred: tuple[str, ...]) -> list[str]:
    """Order recognised columns predictably, while retaining original names."""
    priority = {name: idx for idx, name in enumerate(preferred)}
    return sorted(
        columns,
        key=lambda col: (priority.get(_column_name_key(col), len(priority)), col.lower()),
    )


def find_score_columns(df: pd.DataFrame) -> list[str]:
    """Find likely probability-score columns in a reviewed predictions file.

    We deliberately only expose score-like columns rather than every numeric
    laboratory analyte.  The expected Predict outputs use ``prob_*`` and
    ``max_*_prob`` names, both of which are covered here.
    """
    candidates: list[str] = []
    for col in df.columns:
        key = _column_name_key(col)
        if not any(hint in key for hint in _SCORE_NAME_HINTS):
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        finite = values[np.isfinite(values)]
        if not finite.empty and finite.between(0, 1).all():
            candidates.append(str(col))

    preferred = (
        "max_retrospective_prob",
        "max_realtime_prob",
        "max_retrospective_prob_with_lr",
        "max_realtime_prob_with_lr",
    )
    return _ordered_with_preferred(candidates, preferred)


def find_label_columns(df: pd.DataFrame) -> list[str]:
    """Find likely binary ground-truth columns in a validation file."""
    candidates: list[str] = []
    for col in df.columns:
        key = _column_name_key(col)
        values = df[col].dropna()
        if values.empty:
            continue
        mapped = [_map_binary_label(value, "exclude") for value in values]
        has_binary_values = any(value is not None for value in mapped)
        has_label_name = any(hint in key for hint in _LABEL_NAME_HINTS)

        # A named column is only useful if it contains recognised binary
        # labels; this avoids offering Review's label_timestamp field.
        if has_label_name and has_binary_values:
            candidates.append(str(col))
            continue

        # A low-cardinality categorical or 0/1 column is also a reasonable
        # fallback for datasets that use an unfamiliar target-column name.
        if values.nunique() <= 6 and has_binary_values:
            candidates.append(str(col))

    preferred = (
        "human_label",
        "ground_truth",
        "ground_truth_label",
        "label",
        "expert_review_prediction",
    )
    return _ordered_with_preferred(list(dict.fromkeys(candidates)), preferred)


def default_score_column(columns: list[str]) -> str | None:
    """Choose the most useful Predict output probability by default."""
    return columns[0] if columns else None


def default_label_column(columns: list[str]) -> str | None:
    """Choose the Review-exported human label when it is available."""
    return columns[0] if columns else None


def _normalise_equivocal_policy(policy: str) -> str:
    aliases = {
        "exclude": "exclude",
        "exclude equivocal labels": "exclude",
        "positive": "positive",
        "treat equivocal as contaminated": "positive",
        "negative": "negative",
        "treat equivocal as real": "negative",
    }
    try:
        return aliases[_normalise_text(policy)]
    except KeyError as exc:
        raise ValidationDataError(f"Unknown equivocal-label policy: {policy!r}") from exc


def _map_binary_label(value: Any, equivocal_policy: str) -> int | None:
    """Map common review / ground-truth encodings to a binary outcome."""
    if value is None or pd.isna(value):
        return None

    if isinstance(value, (bool, np.bool_)):
        return int(value)

    if isinstance(value, (int, float, np.integer, np.floating)):
        if not np.isfinite(value):
            return None
        if float(value) == 1:
            return 1
        if float(value) == 0:
            return 0
        return None

    text = _normalise_text(value)
    if text in _POSITIVE_LABELS:
        return 1
    if text in _NEGATIVE_LABELS:
        return 0
    if text in _EQUIVOCAL_LABELS:
        policy = _normalise_equivocal_policy(equivocal_policy)
        return {"exclude": None, "positive": 1, "negative": 0}[policy]

    # CSV files commonly deserialize 0/1 labels as strings such as "1.0".
    try:
        numeric = float(text)
    except ValueError:
        return None
    if np.isfinite(numeric) and numeric in (0, 1):
        return int(numeric)
    return None


def prepare_validation_data(
    df: pd.DataFrame,
    score_column: str,
    label_column: str,
    equivocal_policy: str = "exclude",
) -> PreparedValidationData:
    """Clean probability scores and mapped binary labels from an uploaded file."""
    if score_column not in df.columns:
        raise ValidationDataError(f"Prediction score column not found: {score_column!r}.")
    if label_column not in df.columns:
        raise ValidationDataError(f"Ground-truth label column not found: {label_column!r}.")

    policy = _normalise_equivocal_policy(equivocal_policy)
    scores = pd.to_numeric(df[score_column], errors="coerce").to_numpy(dtype=float)
    labels = np.asarray([_map_binary_label(value, policy) for value in df[label_column]], dtype=object)

    valid_score = np.isfinite(scores) & (scores >= 0) & (scores <= 1)
    valid_label = labels != None  # noqa: E711 - elementwise comparison is intentional
    usable = valid_score & valid_label

    clean_scores = scores[usable]
    clean_labels = labels[usable].astype(int)

    score_excluded = int((~valid_score).sum())
    label_excluded = int((~valid_label).sum())
    return PreparedValidationData(
        labels=clean_labels,
        scores=clean_scores,
        total_rows=len(df),
        excluded_rows=int((~usable).sum()),
        excluded_missing_or_invalid_score=score_excluded,
        excluded_unusable_label=label_excluded,
    )


def _safe_divide(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def operating_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, int | float | None]:
    """Calculate a binary confusion matrix and threshold-dependent metrics.

    A score greater than or equal to the threshold is counted as contaminated.
    Validate uses a binary 0.50 default operating point, so a probability of
    exactly 0.50 is a positive prediction.
    """
    if not 0 <= threshold <= 1:
        raise ValidationDataError("Threshold must be between 0 and 1.")

    predictions = scores >= threshold
    positives = labels == 1
    negatives = ~positives

    tp = int(np.sum(predictions & positives))
    fp = int(np.sum(predictions & negatives))
    tn = int(np.sum(~predictions & negatives))
    fn = int(np.sum(~predictions & positives))

    sensitivity = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    ppv = _safe_divide(tp, tp + fp)
    npv = _safe_divide(tn, tn + fn)
    f1 = _safe_divide(2 * tp, 2 * tp + fp + fn)
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "f1": f1,
    }


def _calibration_points(labels: np.ndarray, scores: np.ndarray, n_bins: int = 10) -> list[dict[str, float | int]]:
    """Return non-empty uniform calibration bins for client-side rendering."""
    edges = np.linspace(0, 1, n_bins + 1)
    points: list[dict[str, float | int]] = []
    for idx, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        if idx == n_bins - 1:
            in_bin = (scores >= left) & (scores <= right)
        else:
            in_bin = (scores >= left) & (scores < right)
        if not np.any(in_bin):
            continue
        points.append(
            {
                "mean_predicted": float(np.mean(scores[in_bin])),
                "fraction_positive": float(np.mean(labels[in_bin])),
                "count": int(np.sum(in_bin)),
            }
        )
    return points


def _threshold_curve(
    labels: np.ndarray,
    scores: np.ndarray,
) -> tuple[list[dict[str, float | None]], list[dict[str, float]]]:
    """Build ROC and PR points using the Validate score >= threshold rule.

    The selectable ROC points exactly match :func:`operating_metrics`.  A
    score group is added *before* emitting its threshold, which makes the
    point at (for example) 0.50 include probabilities equal to 0.50.  The
    visual zero-positive endpoint is selectable at 1.0 whenever no score is
    exactly 1.0; otherwise it remains a non-selectable sentinel because the
    slider cannot move above 1.0.
    """
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    tp = fp = 0
    roc_points: list[dict[str, float | None]] = []
    pr_points: list[dict[str, float]] = []

    def append_point(threshold: float | None):
        nonlocal tp, fp
        tpr = tp / positives if positives else 0.0
        fpr = fp / negatives if negatives else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        roc_points.append(
            {
                "fpr": float(fpr),
                "tpr": float(tpr),
                "threshold": float(threshold) if threshold is not None else None,
            }
        )
        pr_points.append({"recall": float(tpr), "precision": float(precision)})

    # Sort once and walk equal-score groups. Re-scanning scores for every
    # unique probability turns a typical random-score upload into O(n²).
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    # At a threshold of 1.0, scores below 1.0 are negative.  If a score is
    # exactly 1.0, that threshold instead includes the score group, so retain
    # a non-selectable visual origin until the group is processed below.
    if len(sorted_scores) and sorted_scores[0] < 1:
        append_point(1.0)
    else:
        append_point(None)

    start = 0
    while start < len(sorted_scores):
        score = float(sorted_scores[start])
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == score:
            end += 1

        group_labels = sorted_labels[start:end]
        tp += int(np.sum(group_labels == 1))
        fp += int(np.sum(group_labels == 0))
        # The point at ``score`` represents scores >= score.
        append_point(score)
        start = end

    # If the minimum score is above zero, a threshold of zero also includes
    # every row. Add it so slider values below the first observed probability
    # have an exact cached operating point. When zero is observed, its group
    # above already represents the inclusive threshold correctly.
    if len(sorted_scores) and sorted_scores[-1] > 0:
        append_point(0.0)

    return roc_points, pr_points


def _curve_payload(
    roc_points: list[dict[str, float | None]],
    pr_points: list[dict[str, float]],
    positive_count: int,
    negative_count: int,
) -> tuple[
    list[dict[str, float | None]],
    list[dict[str, float]],
    dict[str, list[float | int]],
]:
    """Compact full operating points and downsample only the drawn curves.

    Slider changes need every distinct score threshold, whereas an SVG cannot
    visually benefit from tens of thousands of adjacent line vertices. Keeping
    the full thresholds as numeric arrays makes client updates logarithmic and
    keeps large reviewed batches responsive.
    """
    selectable = [point for point in roc_points if point["threshold"] is not None]
    selectable.sort(key=lambda point: float(point["threshold"]))
    operating = {
        "thresholds": [float(point["threshold"]) for point in selectable],
        "tp": [int(round(float(point["tpr"]) * positive_count)) for point in selectable],
        "fp": [int(round(float(point["fpr"]) * negative_count)) for point in selectable],
    }

    if len(roc_points) <= MAX_RENDER_CURVE_POINTS:
        return roc_points, pr_points, operating

    indices = np.unique(
        np.linspace(0, len(roc_points) - 1, MAX_RENDER_CURVE_POINTS, dtype=int)
    )
    return (
        [roc_points[index] for index in indices],
        [pr_points[index] for index in indices],
        operating,
    )


def build_validation_payload(
    df: pd.DataFrame,
    score_column: str,
    label_column: str,
    equivocal_policy: str = "exclude",
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """Build a JSON-safe analysis payload for the interactive Validate chart."""
    prepared = prepare_validation_data(df, score_column, label_column, equivocal_policy)
    if len(prepared.labels) == 0:
        raise ValidationDataError(
            "No usable rows remain. Use probabilities from 0 to 1 and recognised binary labels."
        )
    if len(np.unique(prepared.labels)) != 2:
        raise ValidationDataError(
            "ROC and PR curves require at least one Real and one Contaminated ground-truth label."
        )

    # Keep every operating point because the front-end lets the user drag the
    # threshold marker directly along the ROC curve.
    roc_points, pr_points = _threshold_curve(prepared.labels, prepared.scores)
    display_roc, display_pr, operating = _curve_payload(
        roc_points,
        pr_points,
        positive_count=int(np.sum(prepared.labels)),
        negative_count=int(np.sum(prepared.labels == 0)),
    )
    roc_auc = float(roc_auc_score(prepared.labels, prepared.scores))
    pr_auc = float(average_precision_score(prepared.labels, prepared.scores))
    prevalence = float(np.mean(prepared.labels))
    metrics = operating_metrics(prepared.labels, prepared.scores, threshold)

    return {
        "score_column": str(score_column),
        "label_column": str(label_column),
        "threshold": float(threshold),
        "threshold_rule": "score >= threshold",
        "summary": {
            "total_rows": prepared.total_rows,
            "included_rows": int(len(prepared.labels)),
            "excluded_rows": prepared.excluded_rows,
            "excluded_missing_or_invalid_score": prepared.excluded_missing_or_invalid_score,
            "excluded_unusable_label": prepared.excluded_unusable_label,
            "positive_count": int(np.sum(prepared.labels)),
            "negative_count": int(np.sum(prepared.labels == 0)),
            "prevalence": prevalence,
        },
        "auc": {"roc": roc_auc, "pr": pr_auc},
        "metrics": metrics,
        "roc": display_roc,
        "pr": display_pr,
        "operating": operating,
        "calibration": _calibration_points(prepared.labels, prepared.scores),
    }
