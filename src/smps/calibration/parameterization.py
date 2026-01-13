from __future__ import annotations

from dataclasses import replace
from typing import Dict

import numpy as np

from smps.physics.evapotranspiration import CropCoefficientCurve
from smps.physics.soil_hydraulics import FeddesParameters


def apply_parameters_to_model_params(base_params, p: Dict[str, float]):
    """Return a new EnhancedModelParameters with calibration parameters applied."""

    # VG parameter multipliers
    Ks_adj = float(p.get("Ks_adj", 1.0))
    alpha_adj = float(p.get("alpha_adj", 1.0))
    theta_s_adj = float(p.get("theta_s_adj", 1.0))
    n_adj = float(p.get("n_adj", 1.0))
    theta_r_adj = float(p.get("theta_r_adj", 1.0))

    vg_params = [
        vg.with_multipliers(
            Ks_adj=Ks_adj,
            alpha_adj=alpha_adj,
            theta_s_adj=theta_s_adj,
            n_adj=n_adj,
            theta_r_adj=theta_r_adj,
        )
        for vg in base_params.vg_params
    ]

    # Crop coefficient scaling
    kcb_scale = float(p.get("kcb_scale", 1.0))
    crop_curve = base_params.crop_coefficients
    if crop_curve is None:
        crop_curve = CropCoefficientCurve.for_crop(base_params.crop_type)
    crop_curve_scaled = replace(
        crop_curve,
        Kcb_ini=float(np.clip(crop_curve.Kcb_ini * kcb_scale, 0.05, 1.5)),
        Kcb_mid=float(np.clip(crop_curve.Kcb_mid * kcb_scale, 0.05, 1.5)),
        Kcb_end=float(np.clip(crop_curve.Kcb_end * kcb_scale, 0.05, 1.5)),
    )

    # Feddes scaling (coarse “dryness” control): scale |psi| thresholds
    feddes_scale = float(p.get("feddes_dryness_scale", 1.0))
    feddes = base_params.feddes_params
    if feddes is None:
        feddes = FeddesParameters.for_crop(base_params.crop_type)

    def _scale_neg(x: float) -> float:
        # psi parameters are negative (except rice psi_1 can be 0). Keep sign.
        if x >= 0:
            return float(x)
        return -abs(float(x)) * float(feddes_scale)

    feddes_scaled = replace(
        feddes,
        psi_1=_scale_neg(feddes.psi_1),
        psi_2=_scale_neg(feddes.psi_2),
        psi_3h=_scale_neg(feddes.psi_3h),
        psi_3l=_scale_neg(feddes.psi_3l),
        psi_4=_scale_neg(feddes.psi_4),
    )

    # Macropore parameters
    infiltration_macropore_max_fraction = float(
        np.clip(p.get("infiltration_macropore_max_fraction",
                base_params.infiltration_macropore_max_fraction), 0.0, 0.95)
    )
    macropore_conductivity_factor = float(
        np.clip(p.get("macropore_conductivity_factor",
                base_params.macropore_conductivity_factor), 0.0, 1e4)
    )

    return replace(
        base_params,
        vg_params=vg_params,
        crop_coefficients=crop_curve_scaled,
        feddes_params=feddes_scaled,
        infiltration_macropore_max_fraction=infiltration_macropore_max_fraction,
        macropore_conductivity_factor=macropore_conductivity_factor,
    )
