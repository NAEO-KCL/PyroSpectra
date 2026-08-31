"""Unit tests for the v2.0 optically-thick retrieval (detection logic, config, reference layout).

These do not require RADIS or any data files: they exercise the detection framework on synthetic
retrieval records and check the static configuration.
"""
import numpy as np
import pytest

from pyrospectra import thick_smoke as TS


def _rec(**bykey):
    """Build a minimal retrieval-style record. Keys look like 'EF__CO'=52.0."""
    d = {"EF": {}, "EF_thin": {}, "SNR": {}, "SNR_thin": {}, "R2": {}, "EF_DL": {},
         "EF_thin_DL": {}, "n_plume": 200}
    for k, v in bykey.items():
        base, sp = k.split("__")
        d.setdefault(base, {})[sp] = v
    d["EF"].setdefault("CO", 52.0)   # CO reference for the carbon-plausibility check
    return d


# -------------------------------------------------------------------------------- configuration ---
def test_config_has_expected_species_and_windows():
    assert TS.CFG["HCN"][2] == [[3306, 3325]]          # nu1 Q-branch, corrected in v2.0
    assert TS.CFG["NO2"][3] == "targ"                  # NO2 is line-by-line, not xsec (UV-Vis .xsc)
    assert TS.CFG["CO2"][1] == "1,2,3"                 # CO2 isotopologues
    assert "N2O" in TS.NO_BASELINE                     # N2O window excluded from the continuum term
    assert all(len(w) == 2 for w in TS.baseline_windows())  # returns [lo, hi] window pairs

def test_continuum_excludes_n2o_window():
    wins = TS.baseline_windows()
    assert [2200.0, 2235.0] not in wins                # N2O window has no continuum term
    assert [1116.0, 1130.0] in wins                    # HCOOH window does

def test_configure_sets_hyperparameters():
    old = TS.CONFIG.snr_detect
    try:
        TS.configure(snr_detect=5.0)
        assert TS.CONFIG.snr_detect == 5.0
    finally:
        TS.CONFIG.snr_detect = old
    with pytest.raises(AttributeError):
        TS.configure(not_a_param=1)


# ------------------------------------------------------------------------------------ detection ---
def test_co2_reference_species_always_detected():
    assert TS.detected(_rec(EF__CO2=1500.0), "CO2") is True

def test_real_species_detected_low_r2_high_snr():
    # CO/CH4/CH2O on the thickest plumes: low R2 but unambiguous SNR
    assert TS.detected(_rec(EF__CO=52, SNR__CO=22, R2__CO=0.17), "CO") is True

def test_negative_slope_is_non_detection():
    assert TS.detected(_rec(EF__C4H4O=-1.7, SNR__C4H4O=13, R2__C4H4O=-0.3), "C4H4O") is False

def test_interference_residual_rejected_by_low_r2():
    # ethanol fitting methanol residual: significant SNR but R2<=0 and not >=10
    assert TS.detected(_rec(EF__C2H6O=3.0, SNR__C2H6O=8.3, R2__C2H6O=-0.18), "C2H6O") is False

def test_carbon_implausible_rejected():
    # propane 66 g/kg: carbon exceeds CO's -> physically impossible, rejected regardless of stats
    assert TS.detected(_rec(EF__C3H8=66, SNR__C3H8=7.6, R2__C3H8=0.13), "C3H8") is False

def test_marginal_real_species_detected():
    assert TS.detected(_rec(EF__CH3COOH=25, SNR__CH3COOH=11, R2__CH3COOH=0.44), "CH3COOH") is True

def test_below_min_plume_not_detected():
    r = _rec(EF__CO=52, SNR__CO=22, R2__CO=0.9); r["n_plume"] = 5
    assert TS.detected(r, "CO") is False

def test_detection_limit_ok_range():
    assert TS.detection_limit_ok(_rec(EF_DL__HCN=0.13), "HCN") is True          # usable bound
    assert TS.detection_limit_ok(_rec(EF_DL__C3H8=26.0), "C3H8") is False       # too loose (>ceiling)
    assert TS.detection_limit_ok(_rec(EF_DL__CH3CN=1e-12), "CH3CN") is False    # band ~absent

def test_carbon_helper_monotonic():
    assert TS.carbon(2.0, "CH4") == pytest.approx(2.0 * 1 / TS.MOLAR_MASS["CH4"])
    assert TS.carbon(1.0, "NO2") == 0.0                                          # no carbon
