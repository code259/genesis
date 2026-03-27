# pyre-ignore-all-errors
import numpy as np  # pyre-ignore[21]

# Fundamental constants (SI)
C_LIGHT = 2.998e8        # m/s
H_PLANCK = 6.626e-34     # J·s
K_BOLTZMANN = 1.381e-23  # J/K
G_NEWTON = 6.674e-11     # m^3 kg^-1 s^-2
M_SUN = 1.989e30         # kg
L_SUN = 3.828e26         # W
PC_TO_M = 3.086e16       # meters per parsec

def check_velocity_physical(velocity_km_s: float, context: str = "") -> dict:
    """
    Velocities must be sub-relativistic for most astrophysical contexts,
    and must not exceed c under any circumstances.
    For stellar/galactic dynamics: flag if |v| > 0.1c (30,000 km/s).
    For cosmological redshifts: apply relativistic formula check separately.
    """
    v_c = abs(velocity_km_s) / (C_LIGHT / 1e3)
    superluminal = v_c >= 1.0
    suspicious = v_c > 0.1 and context not in ("cosmological", "relativistic_jet")

    return {
        "check": "velocity physical bound",
        "velocity_km_s": velocity_km_s,
        "v_over_c": float(f"{v_c:.6f}"),
        "pass": not superluminal,
        "warning": suspicious and not superluminal,
        "interpretation": (
            "FAIL: superluminal velocity" if superluminal else
            f"WARN: v/c = {v_c:.3f}, verify context is relativistic" if suspicious else
            "PASS"
        )
    }

def check_luminosity_physical(luminosity_lsun: float) -> dict:
    """
    Stellar luminosities must be within plausible astrophysical range.
    Below ~1e-5 L_sun: sub-stellar / brown dwarf territory (flag if claimed stellar).
    Above ~1e7 L_sun: hyperluminous, near Eddington for massive stars — flag for justification.
    """
    too_faint = luminosity_lsun < 1e-6
    hyperluminous = luminosity_lsun > 5e6

    return {
        "check": "stellar luminosity plausibility",
        "luminosity_lsun": luminosity_lsun,
        "pass": not too_faint,
        "warning": hyperluminous,
        "interpretation": (
            f"FAIL: luminosity {luminosity_lsun:.2e} L_sun below sub-stellar floor" if too_faint else
            f"WARN: hyperluminous {luminosity_lsun:.2e} L_sun — verify Eddington justification" if hyperluminous else
            "PASS"
        )
    }

def check_eddington_limit(luminosity_lsun: float, mass_msun: float) -> dict:
    """
    For accreting compact objects, luminosity should not exceed Eddington limit
    (unless super-Eddington accretion is explicitly the subject of study).
    L_Edd ≈ 3.2e4 * (M / M_sun) L_sun
    """
    l_edd_lsun = 3.2e4 * mass_msun
    ratio = luminosity_lsun / l_edd_lsun

    return {
        "check": "Eddington luminosity limit",
        "L_over_L_Edd": float(f"{ratio:.4f}"),
        "L_Edd_lsun": l_edd_lsun,
        "pass": ratio <= 10.0,  # allow moderate super-Eddington
        "warning": ratio > 1.0,
        "interpretation": (
            f"FAIL: L/L_Edd = {ratio:.2f}, far exceeds Eddington — requires explicit justification" if ratio > 10.0 else
            f"WARN: super-Eddington L/L_Edd = {ratio:.2f}" if ratio > 1.0 else
            "PASS"
        )
    }

def check_distance_modulus_consistent(
    apparent_mag: float, 
    absolute_mag: float, 
    distance_pc: float,
    extinction_mag: float = 0.0
) -> dict:
    """
    Distance modulus: m - M = 5*log10(d/10 pc) + A
    Check that apparent mag, absolute mag, distance, and extinction are mutually consistent.
    """
    mu_derived = 5 * np.log10(distance_pc / 10.0) + extinction_mag
    m_predicted = absolute_mag + mu_derived
    residual = abs(apparent_mag - m_predicted)

    return {
        "check": "distance modulus self-consistency",
        "mu_derived": float(f"{mu_derived:.3f}"),
        "m_predicted": float(f"{m_predicted:.3f}"),
        "m_reported": apparent_mag,
        "residual_mag": float(f"{residual:.4f}"),
        "pass": residual < 0.05,
        "interpretation": (
            f"FAIL: distance modulus inconsistency of {residual:.3f} mag" if residual >= 0.05 else
            "PASS"
        )
    }

def check_stefan_boltzmann_consistent(
    luminosity_lsun: float, 
    radius_rsun: float, 
    temperature_k: float,
    tolerance: float = 0.05
) -> dict:
    """
    Stefan-Boltzmann: L = 4π R² σ T⁴
    Given any two of {L, R, T}, the third should be consistent.
    """
    SIGMA = 5.670e-8  # W m^-2 K^-4
    R_SUN = 6.957e8   # m

    R_m = radius_rsun * R_SUN
    L_predicted_W = 4 * np.pi * R_m**2 * SIGMA * temperature_k**4
    L_predicted_lsun = L_predicted_W / L_SUN
    
    ratio = luminosity_lsun / L_predicted_lsun
    fractional_error = abs(ratio - 1.0)

    return {
        "check": "Stefan-Boltzmann self-consistency",
        "L_reported_lsun": luminosity_lsun,
        "L_predicted_lsun": float(f"{L_predicted_lsun:.4f}"),
        "ratio": float(f"{ratio:.4f}"),
        "pass": fractional_error < tolerance,
        "interpretation": (
            f"FAIL: L/R/T inconsistency — ratio {ratio:.3f}, expected 1.0 ± {tolerance}" if fractional_error >= tolerance else
            "PASS"
        )
    }
