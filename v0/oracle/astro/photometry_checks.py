# pyre-ignore-all-errors
import numpy as np  # pyre-ignore[21]

# AB magnitude zero points (Jy) for common filters
# m_AB = -2.5*log10(f_nu / 3631 Jy)
AB_ZEROPOINT_JY = 3631.0

# Approximate effective wavelengths (Angstroms) for common filter systems
FILTER_WAVELENGTHS = {
    # SDSS
    "u": 3543, "g": 4770, "r": 6231, "i": 7625, "z": 9134,
    # 2MASS
    "J": 12350, "H": 16620, "K": 21590,
    # HST WFC3
    "F275W": 2750, "F336W": 3360, "F435W": 4350,
    "F606W": 6060, "F814W": 8140,
    # Johnson-Cousins
    "U": 3650, "B": 4450, "V": 5510, "R": 6580, "I": 8060,
}

def check_color_physical(
    filter1: str,
    filter2: str,
    color: float
) -> dict:
    """
    Colors (mag1 - mag2) must be within physically plausible stellar ranges.
    Extreme colors indicate: calibration failure, high extinction, or unusual objects
    (which should be noted explicitly if intentional).
    """
    # Bluest to reddest normal stellar colors (approximate)
    COLOR_RANGES = {
        ("B", "V"):   (-0.4, 2.5),
        ("V", "I"):   (-0.5, 4.0),
        ("V", "K"):   (-0.5, 8.0),
        ("g", "r"):   (-0.5, 2.5),
        ("r", "i"):   (-0.4, 1.5),
        ("J", "K"):   (-0.2, 2.5),
    }

    key = (filter1, filter2)
    key_rev = (filter2, filter1)

    if key in COLOR_RANGES:
        lo, hi = COLOR_RANGES[key]
        in_range = lo <= color <= hi
    elif key_rev in COLOR_RANGES:
        lo, hi = COLOR_RANGES[key_rev]
        in_range = -hi <= color <= -lo
    else:
        return {"check": f"color ({filter1}-{filter2}) physical range", "pass": None,
                "interpretation": f"SKIP: no bounds defined for ({filter1}-{filter2})"}

    return {
        "check": f"color ({filter1}-{filter2}) physical range",
        "color": color,
        "allowed_range": (lo, hi),
        "pass": in_range,
        "interpretation": (
            f"FAIL: ({filter1}-{filter2}) = {color:.3f} outside stellar range [{lo}, {hi}]" if not in_range else
            "PASS"
        )
    }

def check_flux_conservation(
    broadband_flux_jy: float,
    integrated_spectrum_flux_jy: float,
    tolerance: float = 0.05
) -> dict:
    """
    Integrated flux from a spectrum convolved with a filter bandpass
    should match the broadband photometry in that filter.
    Discrepancy > tolerance indicates flux calibration error.
    """
    fractional = abs(broadband_flux_jy - integrated_spectrum_flux_jy) / broadband_flux_jy

    return {
        "check": "flux conservation: photometry vs spectrum",
        "broadband_jy": broadband_flux_jy,
        "spectrum_integrated_jy": integrated_spectrum_flux_jy,
        "fractional_discrepancy": float(f"{fractional:.5f}"),
        "pass": fractional < tolerance,
        "interpretation": (
            f"FAIL: {fractional:.1%} flux discrepancy between photometry and spectrum" if fractional >= tolerance else
            "PASS"
        )
    }

def check_magnitude_system_consistent(
    mag_ab: float,
    mag_vega: float,
    filter_name: str,
    tolerance: float = 0.05
) -> dict:
    """
    AB and Vega magnitudes differ by known, filter-dependent offsets.
    If both are reported, verify their difference matches the expected offset.
    Approximate AB-Vega offsets:
    V: +0.02, B: -0.10, R: +0.16, I: +0.40, J: +0.91, H: +1.39, K: +1.85
    g: -0.08, r: +0.16, i: +0.37, z: +0.54
    """
    AB_VEGA_OFFSETS = {
        "V": 0.02, "B": -0.10, "R": 0.16, "I": 0.40,
        "J": 0.91, "H": 1.39, "K": 1.85,
        "g": -0.08, "r": 0.16, "i": 0.37, "z": 0.54,
    }

    if filter_name not in AB_VEGA_OFFSETS:
        return {"check": "magnitude system consistency", "pass": None,
                "interpretation": f"SKIP: no AB-Vega offset defined for filter {filter_name}"}

    expected_offset = AB_VEGA_OFFSETS[filter_name]
    observed_offset = mag_ab - mag_vega
    residual = abs(observed_offset - expected_offset)

    return {
        "check": f"AB/Vega magnitude system consistency: {filter_name}",
        "mag_ab": mag_ab,
        "mag_vega": mag_vega,
        "observed_offset": float(f"{observed_offset:.4f}"),
        "expected_offset": expected_offset,
        "residual": float(f"{residual:.4f}"),
        "pass": residual < tolerance,
        "interpretation": (
            f"FAIL: AB-Vega offset = {observed_offset:.3f}, expected {expected_offset:.3f} for {filter_name}" if residual >= tolerance else
            "PASS"
        )
    }
