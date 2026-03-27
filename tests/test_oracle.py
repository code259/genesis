# pyre-ignore-all-errors[21]
import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from oracle.astro import physical_checks  # type: ignore
from oracle.astro import catalog_checks  # type: ignore
from oracle.astro import statistical_checks  # type: ignore
from oracle.astro import spectral_checks  # type: ignore
from oracle.astro import photometry_checks  # type: ignore
from oracle.astro import run_oracle  # type: ignore

def test_physical_velocity():
    # Known good: 300 km/s (subluminal)
    res = physical_checks.check_velocity_physical(300.0)
    assert res["pass"] is True
    assert res["warning"] is False

    # Known bad: 4e5 km/s (superluminal)
    res = physical_checks.check_velocity_physical(400000.0)
    assert res["pass"] is False
    assert res["warning"] is False

def test_physical_eddington():
    # Good: 1 Solar mass, 1 Solar luminosity
    res = physical_checks.check_eddington_limit(1.0, 1.0)
    assert res["pass"] is True
    assert res["warning"] is False

    # Bad: 1 Solar mass, 1e6 Solar luminosity
    res = physical_checks.check_eddington_limit(1e6, 1.0)
    assert res["pass"] is False

def test_catalog_recovery():
    # Good: exact match for Vega
    res = catalog_checks.check_benchmark_star_recovery("Vega", {"T_eff": 9600, "log_g": 3.95})
    assert res["pass"] is True

    # Bad: totally wrong T_eff for Vega
    res = catalog_checks.check_benchmark_star_recovery("Vega", {"T_eff": 5000})
    assert res["pass"] is False

def test_statistical_snr():
    # Good: Value 10, error 1 -> S/N = 10 (floor 1.0)
    res = statistical_checks.check_uncertainty_propagation(10.0, 1.0)
    assert res["pass"] is True

    # Bad: Value 10, error 20 -> S/N = 0.5 < 1.0
    res = statistical_checks.check_uncertainty_propagation(10.0, 20.0)
    assert res["pass"] is False

def test_spectral_redshift():
    # Good: H_alpha and H_beta redshifted by identically z=0.1
    obs = {
        "H_alpha": spectral_checks.SPECTRAL_LINES["H_alpha"] * 1.1,
        "H_beta": spectral_checks.SPECTRAL_LINES["H_beta"] * 1.1
    }
    res = spectral_checks.check_redshift_from_lines(obs)
    assert res["pass"] is True

    # Bad: H_alpha z=0.1 and H_beta z=0.5
    obs = {
        "H_alpha": spectral_checks.SPECTRAL_LINES["H_alpha"] * 1.1,
        "H_beta": spectral_checks.SPECTRAL_LINES["H_beta"] * 1.5
    }
    res = spectral_checks.check_redshift_from_lines(obs)
    assert res["pass"] is False

def test_photometry_color():
    # Good: B-V = 0.5
    res = photometry_checks.check_color_physical("B", "V", 0.5)
    assert res["pass"] is True

    # Bad: B-V = 50.0 (unphysical)
    res = photometry_checks.check_color_physical("B", "V", 50.0)
    assert res["pass"] is False

def test_run_oracle_aggregator():
    checks = [
        physical_checks.check_velocity_physical(300.0),
        photometry_checks.check_color_physical("B", "V", 0.5)
    ]
    report = run_oracle.run_all_checks(checks)
    assert report["oracle_pass"] is True
    assert report["summary"]["total"] == 2
    assert report["summary"]["passed"] == 2
