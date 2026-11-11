"""End-to-end test of the t-route <-> tx_fast_hydrology (KF) Interface.

Mirrors ``interface.ipynb``: runs the tx_res_calibrated case three ways
(t-route + KF, t-route alone, t-route with nudging), each with a warmup
followed by a forecast, and sanity-checks the forecast results.

The case is assembled in a temporary directory: configs, geopackage and
KF_Data are copied from ``test/tx_res_calibrated`` and the USGS timeslice and
lateral-inflow folders are regenerated from the consolidated
``tx_res_calibrated_forcing.nc`` (see ``tx_res_forcing.py``).

Run from the repo root with::

    python -m pytest Interface/test_interface.py -v
"""

import copy
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tx_fast_hydrology", reason="tx_fast_hydrology is not installed")

from Interface import Interface  # noqa: E402
from tx_res_forcing import unpack  # noqa: E402

CASE_SRC = Path(__file__).resolve().parents[1] / "test" / "tx_res_calibrated"
FORCING_NC = CASE_SRC / "tx_res_calibrated_forcing.nc"

# restart_parameters.start_datetime in troute.yaml / start_date in KF.yaml
START = pd.Timestamp("2023-05-13 00:00:00", tz="UTC")
WARMUP_STEPS = 96
FORECAST_STEPS = 96

RUNS = ("with_KF", "without_KF", "without_KF_nudging")


@pytest.fixture(scope="module")
def case_dir(tmp_path_factory):
    """Self-contained copy of the tx_res_calibrated case with generated forcing."""
    case = tmp_path_factory.mktemp("tx_res_calibrated")
    for name in ("troute.yaml", "troute_nudge.yaml", "tx_res_calibrated.gpkg"):
        shutil.copy2(CASE_SRC / name, case / name)
    shutil.copytree(CASE_SRC / "KF_Data", case / "KF_Data")
    (case / "outputs").mkdir()

    usgs_dir, qlat_dir = unpack(FORCING_NC, case)
    assert len(list(usgs_dir.glob("*.usgsTimeSlice.ncdf"))) == 384
    assert len(list(qlat_dir.glob("nex-*.csv"))) == 174
    return case


@pytest.fixture(scope="module")
def interfaces(case_dir):
    """The three Interface runs from the notebook, after warmup + forecast."""
    kf_yaml = case_dir / "KF_Data" / "KF.yaml"
    with_kf = Interface(yaml_file_path_troute=case_dir / "troute.yaml", yaml_file_path_KF=kf_yaml)
    without_kf = copy.deepcopy(with_kf)
    without_kf_nudging = Interface(
        yaml_file_path_troute=case_dir / "troute_nudge.yaml", yaml_file_path_KF=kf_yaml
    )

    with_kf.warmup_with_KF(WARMUP_STEPS)
    without_kf.warmup_without_KF(WARMUP_STEPS)
    without_kf_nudging.warmup_without_KF(WARMUP_STEPS)

    runs = dict(zip(RUNS, (with_kf, without_kf, without_kf_nudging)))
    for interface in runs.values():
        interface.forecast_only(FORECAST_STEPS)
    return runs


def _step(interface):
    """Model time covered by one Interface update (BMI time step)."""
    return pd.Timedelta(seconds=interface.troute_model.get_time_step())


@pytest.mark.parametrize("run", RUNS)
def test_warmup_advances_clock(interfaces, run):
    model = interfaces[run].troute_model
    assert model.get_start_datetime() == START
    assert model.get_current_datetime() == START + WARMUP_STEPS * _step(interfaces[run])


@pytest.mark.parametrize("run", RUNS)
def test_forecast_results(interfaces, run):
    interface = interfaces[run]
    df = interface.get_results_dataframe()

    # One row for the forecast start plus one per forecast step.
    assert df.shape[0] == FORECAST_STEPS + 1
    dt = _step(interface)
    forecast_start = START + WARMUP_STEPS * dt
    assert df.index[0] == forecast_start
    assert df.index[-1] == forecast_start + FORECAST_STEPS * dt

    # One column per reach in the t-route network.
    reach_list = interface.troute_model.get_value_ptr("reach_list")
    assert df.shape[1] == len(reach_list) > 0
    np.testing.assert_array_equal(df.columns.to_numpy(), reach_list)

    values = df.to_numpy(dtype=float)
    assert np.isfinite(values).all(), "forecast discharge contains NaN/inf"
    assert (values >= 0).all(), "forecast discharge contains negative values"
    assert values.max() > 0, "forecast discharge is all zero"
