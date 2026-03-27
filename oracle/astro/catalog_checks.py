# pyre-ignore-all-errors
import typing

# Benchmark objects: name -> known properties (literature values)
# Extend this as the target problem requires.
BENCHMARK_STARS: typing.Dict[str, typing.Dict[str, typing.Any]] = {
    "Vega":        {"T_eff": 9600,  "log_g": 3.95, "M_V": 0.582,  "SpType": "A0V"},
    "Sun":         {"T_eff": 5778,  "log_g": 4.44, "M_V": 4.83,   "SpType": "G2V"},
    "Sirius_A":    {"T_eff": 9940,  "log_g": 4.33, "M_V": 1.43,   "SpType": "A1V"},
    "Betelgeuse":  {"T_eff": 3500,  "log_g": 0.5,  "M_V": -5.85,  "SpType": "M1Ia"},
    "Proxima_Cen": {"T_eff": 3042,  "log_g": 5.20, "M_V": 15.53,  "SpType": "M5.5Ve"},
}

BENCHMARK_GALAXIES: typing.Dict[str, typing.Dict[str, typing.Any]] = {
    "M87":   {"z": 0.00436, "M_BH_msun": 6.5e9,  "type": "E0"},
    "M31":   {"z": -0.001,  "dist_kpc": 785,      "type": "SA(s)b"},
    "M82":   {"z": 0.000677,"SFR_msun_yr": 13.0,  "type": "starburst"},
    "NGC1052":{"z": 0.00504, "M_BH_msun": 1.5e8,  "type": "E4"},
}

def check_benchmark_star_recovery(
    name: str,
    derived_properties: typing.Dict[str, typing.Any],
    tolerances: typing.Optional[typing.Dict[str, float]] = None
) -> dict:
    """
    Verify that derived stellar parameters match literature values for benchmark stars.
    tolerances: dict of property -> fractional tolerance (default 5% for T_eff, 0.1 dex for log_g)
    """
    if tolerances is None:
        tolerances = {"T_eff": 0.05, "log_g": 0.1, "M_V": 0.1}

    if name not in BENCHMARK_STARS:
        return {"check": "benchmark star recovery", "pass": None,
                "interpretation": f"SKIP: {name} not in benchmark catalog"}

    known = BENCHMARK_STARS[name]
    failures = []

    for prop, tol in tolerances.items():
        if prop not in derived_properties or prop not in known:
            continue
        derived = float(derived_properties[prop])
        reference = float(known[prop])
        fractional = abs(derived - reference) / abs(reference)
        if fractional > tol:
            failures.append(f"{prop}: derived={derived}, known={reference}, err={fractional:.1%}")

    return {
        "check": f"benchmark star recovery: {name}",
        "failures": failures,
        "pass": len(failures) == 0,
        "interpretation": (
            f"FAIL: {'; '.join(failures)}" if failures else "PASS"
        )
    }

def check_benchmark_galaxy_recovery(
    name: str,
    derived_properties: typing.Dict[str, typing.Any],
    tolerances: typing.Optional[typing.Dict[str, float]] = None
) -> dict:
    """
    Verify derived galaxy properties against literature values.
    """
    if tolerances is None:
        tolerances = {"z": 0.001, "dist_kpc": 0.05}

    if name not in BENCHMARK_GALAXIES:
        return {"check": "benchmark galaxy recovery", "pass": None,
                "interpretation": f"SKIP: {name} not in benchmark catalog"}

    known = BENCHMARK_GALAXIES[name]
    failures = []

    for prop, tol in tolerances.items():
        if prop not in derived_properties or prop not in known:
            continue
        derived = float(derived_properties[prop])
        reference = float(known[prop])
        fractional = abs(derived - reference) / (abs(reference) + 1e-12)
        if fractional > tol:
            failures.append(f"{prop}: derived={derived}, known={reference}, err={fractional:.1%}")

    return {
        "check": f"benchmark galaxy recovery: {name}",
        "failures": failures,
        "pass": len(failures) == 0,
        "interpretation": (
            f"FAIL: {'; '.join(failures)}" if failures else "PASS"
        )
    }
