"""Tests for the balanced mixture-ratio simulation datasets."""

import numpy as np
import pandas as pd

from src.features import BMP_ANALYTES, CBC_ANALYTES
from src.simulate import (
    get_fluid_concentrations,
    make_regression_training_data_bmp,
    make_regression_training_data_cbc,
)


def _bmp_template(rows: int = 3) -> pd.DataFrame:
    data = {}
    for index, column in enumerate(BMP_ANALYTES):
        value = float(100 + index)
        data[column] = np.full(rows, value)
        data[f"{column}_prior"] = np.full(rows, value + 10)
        data[f"{column}_post"] = np.full(rows, value + 20)
    return pd.DataFrame(data)


def _cbc_template(rows: int = 3) -> pd.DataFrame:
    data = {}
    for index, column in enumerate(CBC_ANALYTES):
        value = float(10 + index)
        data[column] = np.full(rows, value)
        data[f"{column}_prior"] = np.full(rows, value + 10)
        data[f"{column}_post"] = np.full(rows, value + 20)
    return pd.DataFrame(data)


def _assert_uniform_grid(frame: pd.DataFrame, cases_per_ratio: int) -> None:
    expected = np.round(np.arange(0.0, 0.51, 0.01), 2)
    observed = np.sort(frame["mix_ratio"].unique())
    np.testing.assert_allclose(observed, expected)
    assert frame["mix_ratio"].value_counts().eq(cases_per_ratio).all()


def test_bmp_regression_simulation_has_uniform_zero_inclusive_grid():
    template = _bmp_template()
    fluid = get_fluid_concentrations().query("fluid == 'NS'").iloc[0]

    result = make_regression_training_data_bmp(template, fluid, cases_per_ratio=2)

    assert len(result) == 51 * 2
    _assert_uniform_grid(result, cases_per_ratio=2)
    assert result["label"].eq("NS").all()
    assert result.loc[result["mix_ratio"] == 0, "sodium"].eq(template["sodium"].iloc[0]).all()
    assert result.loc[result["mix_ratio"] == 0, "sodium_prior"].eq(template["sodium_prior"].iloc[0]).all()


def test_cbc_regression_simulation_has_uniform_zero_inclusive_grid():
    template = _cbc_template()

    result = make_regression_training_data_cbc(template, cases_per_ratio=2)

    assert len(result) == 51 * 2
    _assert_uniform_grid(result, cases_per_ratio=2)
    assert result.loc[result["mix_ratio"] == 0, "Hgb"].eq(template["Hgb"].iloc[0]).all()
    assert result.loc[result["mix_ratio"] == 0, "Hgb_prior"].eq(template["Hgb_prior"].iloc[0]).all()
    assert result.loc[result["mix_ratio"] == 0.5, "Hgb"].eq(template["Hgb"].iloc[0] * 0.5).all()
