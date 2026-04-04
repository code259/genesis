# pyre-ignore-all-errors
import numpy as np  # pyre-ignore[21]
from scipy import stats  # pyre-ignore[21]

def check_uncertainty_propagation(
    value: float,
    uncertainty: float,
    snr_floor: float = 1.0
) -> dict:
    """
    Uncertainty must be positive and finite.
    Signal-to-noise ratio must exceed floor (default S/N > 1 to be reported).
    Fractional uncertainty > 1 (100%) warrants a warning.
    """
    if uncertainty <= 0 or not np.isfinite(uncertainty):
        return {
            "check": "uncertainty propagation",
            "pass": False,
            "interpretation": f"FAIL: non-physical uncertainty value {uncertainty}"
        }
    snr = abs(value) / uncertainty
    high_frac = (uncertainty / abs(value)) > 1.0 if value != 0 else False

    return {
        "check": "uncertainty propagation",
        "snr": float(f"{snr:.2f}"),
        "fractional_uncertainty": float(f"{uncertainty / abs(value):.4f}") if value != 0 else None,
        "pass": snr >= snr_floor,
        "warning": high_frac,
        "interpretation": (
            f"FAIL: S/N = {snr:.2f} below floor {snr_floor}" if snr < snr_floor else
            f"WARN: fractional uncertainty > 100%" if high_frac else
            "PASS"
        )
    }

def check_chi_squared_fit(
    chi2: float,
    n_data: int,
    n_params: int,
    tolerance: float = 3.0
) -> dict:
    """
    Reduced chi-squared chi2_r = chi2 / (n_data - n_params).
    chi2_r >> 1: poor fit (underestimated errors or wrong model).
    chi2_r << 1: overfitting or overestimated errors.
    Flag if chi2_r outside [1/tolerance, tolerance].
    """
    dof = n_data - n_params
    if dof <= 0:
        return {"check": "chi-squared fit quality", "pass": False,
                "interpretation": f"FAIL: non-positive DOF ({dof})"}

    chi2_r = chi2 / dof
    p_value = 1.0 - stats.chi2.cdf(chi2, dof)

    good = (1.0 / tolerance) <= chi2_r <= tolerance

    return {
        "check": "chi-squared fit quality",
        "chi2_reduced": float(f"{chi2_r:.4f}"),
        "dof": dof,
        "p_value": float(f"{p_value:.6f}"),
        "pass": good,
        "interpretation": (
            f"FAIL: chi2_r = {chi2_r:.3f}, poor fit (should be near 1.0)" if not good else
            "PASS"
        )
    }

def check_redshift_distance_consistency(
    redshift: float,
    distance_mpc: float,
    H0: float = 70.0,
    tolerance: float = 0.10
) -> dict:
    """
    Hubble law sanity check for low-z sources (z < 0.3): d ≈ cz/H0.
    For z > 0.3, a full cosmological calculation is needed — flag for manual review.
    """
    if redshift > 0.3:
        return {
            "check": "redshift-distance consistency",
            "pass": None,
            "interpretation": "SKIP: z > 0.3 requires full cosmological computation, not Hubble law"
        }

    d_hubble_mpc = (2.998e5 * redshift) / H0
    fractional = abs(distance_mpc - d_hubble_mpc) / d_hubble_mpc

    return {
        "check": "redshift-distance consistency (Hubble law)",
        "z": redshift,
        "d_reported_mpc": distance_mpc,
        "d_hubble_mpc": float(f"{d_hubble_mpc:.3f}"),
        "fractional_discrepancy": float(f"{fractional:.4f}"),
        "pass": fractional < tolerance,
        "interpretation": (
            f"FAIL: d_reported={distance_mpc} Mpc vs d_Hubble={d_hubble_mpc:.1f} Mpc ({fractional:.1%} discrepancy)" if fractional >= tolerance else
            "PASS"
        )
    }

def check_photon_count_statistics(
    counts: float,
    reported_snr: float,
    tolerance: float = 0.05
) -> dict:
    """
    For Poisson-dominated photon counting (CCD/detector data):
    Expected S/N ≈ sqrt(counts). Check reported S/N is consistent.
    """
    expected_snr = np.sqrt(max(counts, 0))
    if expected_snr == 0:
        return {"check": "photon count statistics", "pass": False,
                "interpretation": "FAIL: zero counts"}

    fractional = abs(reported_snr - expected_snr) / expected_snr

    return {
        "check": "photon count Poisson statistics",
        "counts": counts,
        "expected_snr": float(f"{expected_snr:.2f}"),
        "reported_snr": reported_snr,
        "fractional_discrepancy": float(f"{fractional:.4f}"),
        "pass": fractional < tolerance,
        "interpretation": (
            f"WARN: reported S/N={reported_snr:.1f} vs Poisson-expected {expected_snr:.1f} ({fractional:.1%} discrepancy)" if fractional >= tolerance else
            "PASS"
        )
    }
