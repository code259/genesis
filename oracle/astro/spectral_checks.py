# pyre-ignore-all-errors
import numpy as np  # pyre-ignore[21]

# Common spectral lines: name -> vacuum rest wavelength (Angstroms)
SPECTRAL_LINES = {
    # Hydrogen Balmer series
    "H_alpha":    6564.61,
    "H_beta":     4862.68,
    "H_gamma":    4341.68,
    "H_delta":    4102.89,
    # Lyman series
    "Ly_alpha":   1215.67,
    "Ly_beta":    1025.72,
    # Metal lines
    "CaII_K":     3933.66,
    "CaII_H":     3968.47,
    "NaI_D1":     5895.92,
    "NaI_D2":     5889.95,
    "MgII_2796":  2796.35,
    "MgII_2803":  2803.53,
    "OII_3727":   3727.09,
    "OIII_4959":  4960.30,
    "OIII_5007":  5008.24,
    "NII_6548":   6549.86,
    "NII_6583":   6585.27,
    "SII_6716":   6718.29,
    "SII_6731":   6732.67,
    # CO bandheads (near-IR)
    "CO_2-0":     22935.0,
    "CO_3-1":     23227.0,
}

def check_redshift_from_lines(
    observed_wavelengths: dict,
    tolerance_km_s: float = 50.0
) -> dict:
    """
    Given observed wavelengths for multiple identified lines, check that all
    implied redshifts are mutually consistent.
    observed_wavelengths: dict of line_name -> observed_wavelength_angstrom
    """
    C_KM_S = 2.998e5

    redshifts = {}
    for line, obs_wl in observed_wavelengths.items():
        if line not in SPECTRAL_LINES:
            continue
        rest_wl = SPECTRAL_LINES[line]
        z = (obs_wl - rest_wl) / rest_wl
        redshifts[line] = z

    if len(redshifts) < 2:
        return {
            "check": "multi-line redshift consistency",
            "pass": None,
            "interpretation": "SKIP: fewer than 2 identified lines"
        }

    z_values = list(redshifts.values())
    z_mean = np.mean(z_values)
    z_std = np.std(z_values)
    max_dv = z_std * C_KM_S  # velocity scatter in km/s

    inconsistent = {k: v for k, v in redshifts.items() 
                    if abs(v - z_mean) * C_KM_S > tolerance_km_s}

    return {
        "check": "multi-line redshift consistency",
        "z_mean": float(f"{z_mean:.6f}"),
        "z_std": float(f"{z_std:.8f}"),
        "velocity_scatter_km_s": float(f"{max_dv:.2f}"),
        "inconsistent_lines": inconsistent,
        "pass": len(inconsistent) == 0,
        "interpretation": (
            f"FAIL: lines {list(inconsistent.keys())} inconsistent with z_mean={z_mean:.5f}" if inconsistent else
            f"PASS: all lines consistent at z={z_mean:.5f} ± {max_dv:.1f} km/s"
        )
    }

def check_line_ratio_physical(
    ratio_name: str,
    observed_ratio: float
) -> dict:
    """
    Diagnostic line ratios must fall within physically allowed ranges.
    Known forbidden ranges indicate calibration errors or misidentification.
    """
    PHYSICAL_RANGES = {
        # BPT diagram bounds
        "NII_Ha":      (1e-3, 10.0),    # [NII]6583 / H_alpha
        "OIII_Hb":     (0.01, 100.0),   # [OIII]5007 / H_beta
        "SII_Ha":      (0.01, 5.0),     # [SII]6716+6731 / H_alpha
        # Balmer decrement (intrinsic H_alpha/H_beta = 2.86 for Case B)
        "Balmer_dec":  (2.0, 20.0),     # reddened values up to ~20
        # [OIII] doublet ratio (density sensitive)
        "OIII_doublet":(0.33, 3.0),     # 4959/5007 ≈ 1/3 (theoretical), allow range
        # [SII] doublet ratio (density sensitive: 0.44 < r < 1.42)
        "SII_doublet": (0.40, 1.50),
    }

    if ratio_name not in PHYSICAL_RANGES:
        return {"check": f"line ratio physical range: {ratio_name}", "pass": None,
                "interpretation": f"SKIP: {ratio_name} not in known ratio catalog"}

    lo, hi = PHYSICAL_RANGES[ratio_name]
    in_range = lo <= observed_ratio <= hi

    return {
        "check": f"line ratio physical range: {ratio_name}",
        "observed": observed_ratio,
        "allowed_range": (lo, hi),
        "pass": in_range,
        "interpretation": (
            f"FAIL: {ratio_name} = {observed_ratio:.3f} outside physical range [{lo}, {hi}]" if not in_range else
            "PASS"
        )
    }
