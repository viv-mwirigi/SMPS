import numpy as np
import pandas as pd

from smps.calibration.calibrate import calibrate
from smps.calibration.problem import CalibrationConfig, CalibrationDataset
from smps.physics import create_water_balance_model


def test_calibration_objective_smoke_runs_fast():
    # Build a tiny synthetic dataset where observations match the baseline model.
    model = create_water_balance_model(
        use_full_physics=True, soil_texture="loam")

    n_days = 20
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
    forcings = pd.DataFrame(
        {
            "date": dates,
            "precipitation_mm": np.zeros(n_days),
            "et0_mm": np.full(n_days, 4.0),
            "ndvi": np.full(n_days, 0.5),
            "irrigation_mm": np.zeros(n_days),
            "site_id": "s1",
        }
    )

    sim = model.run_period(forcings.set_index(
        "date"), warmup_days=0, return_fluxes=False)

    # Use the model outputs as perfect observations
    forcings["theta_obs_0"] = sim["theta_layer_0"].to_numpy()
    forcings["theta_obs_1"] = sim["theta_layer_1"].to_numpy()

    ds = CalibrationDataset(df=forcings)
    cfg = CalibrationConfig(
        obs_theta_columns=["theta_obs_0", "theta_obs_1"], warmup_days=0)

    # Keep optimizer extremely small for test speed.
    res = calibrate(
        base_model=model,
        dataset=ds,
        config=cfg,
        global_maxiter=1,
        global_popsize=4,
        refine=False,
    )

    assert np.isfinite(res.best_objective)
    assert "Ks_adj" in res.best_parameters
