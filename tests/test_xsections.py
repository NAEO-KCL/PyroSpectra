"""
Tests for the measured cross-section pathway.

The fixtures written here are SYNTHETIC - a Gaussian band with an analytically known
integral, written out in the HITRAN .xsc and PNNL two-column formats. They exercise the
parser, the unit conversion and the state handling; they are not, and must not be
mistaken for, real spectroscopic data. Real cross-sections must be obtained from the
sources listed in DATA_SOURCES.md.
"""

import numpy as np
import pytest

from pyrospectra.conventions import (
    ATM_TO_PA, BAR_TO_PA, LN10, cross_section_to_absorbance, number_density,
    pnnl_to_absorbance,
)
from pyrospectra.xsections import (
    CrossSectionError, apply_instrument_lineshape, cross_section_reference,
    instrument_kernel, read_hitran_xsc, read_pnnl, select_state,
)

PEAK = 3.0e-19          # cm^2 / molecule
CENTRE, WIDTH = 1225.0, 4.0


def gaussian_band(w, peak=PEAK, centre=CENTRE, width=WIDTH):
    return peak * np.exp(-0.5 * ((w - centre) / width) ** 2)


def write_xsc(path, temperatures=(278.0, 298.0, 323.0), pressure_torr=760.0,
              wmin=1200.0, wmax=1250.0, npts=501, resolution=0.112,
              strength_slope=0.0):
    """Write a synthetic multi-temperature HITRAN-format .xsc file."""
    lines = []
    w = np.linspace(wmin, wmax, npts)
    for T in temperatures:
        vals = gaussian_band(w) * (1.0 + strength_slope * (T - temperatures[0]))
        header = (f"{'SyntheticVOC':<20}{wmin:>10.4f}{wmax:>10.4f}{npts:>7d}"
                  f"{T:>7.2f}{pressure_torr:>6.1f}{vals.max():>10.3E}"
                  f"{resolution:>5.3f}{'SyntheticVOC':<15}{'':<4}{'air':<3}{1:>3d}")
        lines.append(header)
        for i in range(0, npts, 10):
            lines.append("".join(f"{v:10.3E}" for v in vals[i:i + 10]))
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def xsc_file(tmp_path):
    return write_xsc(tmp_path / "synthetic.xsc")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_xsc_roundtrip(xsc_file):
    bands = read_hitran_xsc(xsc_file)
    assert len(bands) == 3
    b = bands[1]
    assert b.molecule == "SyntheticVOC"
    assert b.temperature_k == 298.0
    assert b.values.size == 501
    np.testing.assert_allclose(b.pressure_bar, 760.0 * 133.32236842105263 / BAR_TO_PA,
                               rtol=1e-9)
    np.testing.assert_allclose(b.values.max(), PEAK, rtol=2e-3)


def test_xsc_truncated_file_raises(tmp_path):
    p = write_xsc(tmp_path / "trunc.xsc", temperatures=(298.0,))
    lines = p.read_text().splitlines()
    p.write_text("\n".join(lines[:5]))
    with pytest.raises(CrossSectionError, match="ends after"):
        read_hitran_xsc(p)


def test_xsc_malformed_header_raises(tmp_path):
    p = tmp_path / "bad.xsc"
    p.write_text("this is not a fixed width hitran header\n1.0 2.0 3.0\n")
    with pytest.raises(CrossSectionError, match="malformed"):
        read_hitran_xsc(p)


def test_pnnl_reader_skips_headers(tmp_path):
    p = tmp_path / "pnnl.txt"
    w = np.linspace(1200, 1250, 200)
    v = gaussian_band(w) * 1e18
    p.write_text("# PNNL style header\nwavenumber  absorbance\n" +
                 "\n".join(f"{a:.4f} {b:.6e}" for a, b in zip(w, v)))
    band = read_pnnl(p, molecule="SyntheticVOC")
    assert band.values.size == 200
    assert band.units == "ppm-1 m-1"


def test_two_column_requirement(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text("# nothing but comments\n")
    with pytest.raises(CrossSectionError, match="two numeric rows"):
        read_pnnl(p)


# ---------------------------------------------------------------------------
# Unit conversion - checked against an independent hand calculation
# ---------------------------------------------------------------------------

def test_cross_section_to_absorbance_matches_beer_lambert():
    sigma, P, T, L, x = 1e-19, 1.01325, 448.15, 500.0, 1e-6
    n = number_density(P, T, x)
    tau_expected = sigma * n * L
    np.testing.assert_allclose(
        cross_section_to_absorbance(sigma, P, T, L, x, "napierian"), tau_expected,
        rtol=1e-12)
    np.testing.assert_allclose(
        cross_section_to_absorbance(sigma, P, T, L, x, "decadic"), tau_expected / LN10,
        rtol=1e-12)


def test_pnnl_conversion_at_reference_state_is_identity():
    """1 ppm.m of a 1 (ppm m)^-1 coefficient is unit decadic absorbance."""
    a = pnnl_to_absorbance(1.0, ATM_TO_PA / BAR_TO_PA, 296.15,
                           path_length_cm=100.0, mole_fraction=1e-6,
                           convention="decadic")
    np.testing.assert_allclose(a, 1.0, rtol=1e-12)


def test_pnnl_density_scaling_with_temperature():
    """A fixed mole fraction is fewer molecules per cm^3 at higher T, as n ~ P/T."""
    kw = dict(path_length_cm=500.0, mole_fraction=1e-6, convention="decadic")
    hot = pnnl_to_absorbance(1.0, 1.01325, 448.15, **kw)
    cold = pnnl_to_absorbance(1.0, 1.01325, 296.15, **kw)
    np.testing.assert_allclose(hot / cold, 296.15 / 448.15, rtol=1e-9)


def test_cross_section_and_line_by_line_share_a_scale():
    """
    Both pathways must express 'absorbance of 1 ppm over the cell path' in the same
    convention, or the retrieved ppm of an xsec species is wrong by ln(10) relative to
    a line-by-line one. This is the property that makes a mixed reference matrix valid.
    """
    sigma, P, T, L = 1e-19, 1.01325, 448.15, 500.0
    decadic = cross_section_to_absorbance(sigma, P, T, L, 1e-6, "decadic")
    napierian = cross_section_to_absorbance(sigma, P, T, L, 1e-6, "napierian")
    np.testing.assert_allclose(napierian / decadic, LN10, rtol=1e-12)


# ---------------------------------------------------------------------------
# State selection
# ---------------------------------------------------------------------------

def test_temperature_interpolation_is_linear(xsc_file):
    bands = read_hitran_xsc(xsc_file)
    for b, scale in zip(bands, (1.0, 2.0, 3.0)):
        b.values = b.values * scale
    band, prov = select_state(bands, 288.0, 1.0)       # midway between 278 and 298
    assert prov["interpolated"] is True
    np.testing.assert_allclose(prov["interpolation_weight"], 0.5, rtol=1e-9)
    np.testing.assert_allclose(band.values.max(), 1.5 * PEAK, rtol=5e-3)


def test_extrapolation_warns_and_records_distance(xsc_file):
    bands = read_hitran_xsc(xsc_file)
    with pytest.warns(RuntimeWarning, match="hot-band"):
        band, prov = select_state(bands, 448.15, 1.0)
    assert prov["interpolated"] is False
    assert prov["temperature_extrapolation_K"] > 100
    assert prov["temperature_used_K"] == 323.0


def test_band_strength_extrapolation(tmp_path):
    """Contour from the nearest measurement, integral from the trend in T."""
    p = write_xsc(tmp_path / "trend.xsc", strength_slope=0.004)   # +0.4%/K
    bands = read_hitran_xsc(p)
    with pytest.warns(RuntimeWarning):
        band, prov = select_state(bands, 400.0, 1.0, window=(1200, 1250),
                                  strength_extrapolation=True)
    assert prov["strength_rescaled"] is True
    assert prov["strength_scale"] > 1.0

    with pytest.warns(RuntimeWarning):
        plain, prov2 = select_state(bands, 400.0, 1.0, window=(1200, 1250),
                                    strength_extrapolation=False)
    assert prov2["strength_rescaled"] is False
    assert band.values.max() > plain.values.max()


def test_window_outside_coverage_raises(xsc_file):
    bands = read_hitran_xsc(xsc_file)
    with pytest.raises(CrossSectionError, match="no cross-section coverage"):
        select_state(bands, 298.0, 1.0, window=(2000, 2100))


# ---------------------------------------------------------------------------
# Instrument lineshape
# ---------------------------------------------------------------------------

def test_kernel_is_area_normalised():
    for shape in ("gaussian", "triangular"):
        x, k = instrument_kernel(0.01, 0.5, lineshape=shape)
        np.testing.assert_allclose(np.trapezoid(k, x), 1.0, rtol=1e-6)


def test_convolution_conserves_integrated_band(xsc_file):
    """Broadening redistributes intensity; it must not create or destroy it."""
    band = read_hitran_xsc(xsc_file)[1]
    before = band.integrated(1205, 1245)
    out, prov = apply_instrument_lineshape(band, 0.5)
    assert prov["convolved"] is True
    np.testing.assert_allclose(out.integrated(1205, 1245), before, rtol=1e-3)


def test_broadening_is_applied_in_quadrature(xsc_file):
    """A library at finite resolution is not broadened as if monochromatic."""
    band = read_hitran_xsc(xsc_file)[1]
    _, prov = apply_instrument_lineshape(band, 0.5)
    src = 0.112 / 2.3548200450309493
    np.testing.assert_allclose(prov["effective_sigma_cm1"],
                               np.sqrt(0.5 ** 2 - src ** 2), rtol=1e-9)


def test_coarse_library_is_not_deconvolved(tmp_path):
    p = write_xsc(tmp_path / "coarse.xsc", temperatures=(298.0,), resolution=4.0)
    band = read_hitran_xsc(p)[0]
    with pytest.warns(RuntimeWarning, match="no finer"):
        out, prov = apply_instrument_lineshape(band, 0.5)
    assert prov["convolved"] is False
    np.testing.assert_array_equal(out.values, band.values)


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

def test_cross_section_reference_end_to_end(xsc_file):
    grid = np.arange(800.0, 1400.0, 0.241)
    with pytest.warns(RuntimeWarning):
        ref, prov = cross_section_reference(
            xsc_file, (1205, 1245), grid, pressure_bar=1.01325,
            temperature_k=448.15, sigma_cm1=0.5)

    assert ref.shape == grid.shape
    assert np.all(ref[(grid < 1205) | (grid > 1245)] == 0)      # zero outside window
    assert ref.max() > 0
    assert prov["channels_in_window"] > 100
    assert "sha256" in prov and len(prov["sha256"]) == 64

    # Magnitude check against a hand calculation at the band centre.
    n = number_density(1.01325, 448.15, 1e-6)
    expected_peak = PEAK * n * 500.0 / LN10
    np.testing.assert_allclose(ref.max(), expected_peak, rtol=0.05)


def test_missing_file_raises(tmp_path):
    grid = np.arange(1200.0, 1250.0, 0.241)
    with pytest.raises(FileNotFoundError):
        cross_section_reference(tmp_path / "absent.xsc", (1205, 1245), grid,
                                1.01325, 448.15, 0.5)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


# ---------------------------------------------------------------------------
# Packing (round-trip fidelity of the transfer format)
# ---------------------------------------------------------------------------

def test_pack_roundtrip_preserves_fitted_channels(tmp_path):
    """A packed archive must restore exactly the channels the retrieval would fit."""
    from pyrospectra import fitted_channel_mask, pack_burn, read_packed

    d = tmp_path / "burn"
    (d / "Spectra").mkdir(parents=True)
    w = np.arange(900.0, 1300.0, 0.241)
    rng = np.random.default_rng(0)
    raw = np.array([1000 + 50 * np.sin(w / 40) + rng.normal(0, 2, w.size)
                    for _ in range(12)])
    for i, row in enumerate(raw):
        np.savetxt(d / "Spectra" / f"spectrum_{i}.prn",
                   np.column_stack([w, row]), fmt="%.4f %.4f")
    (d / "PT_Log.txt").write_text(
        "\n".join(f"2026-03-01,10:00:{i:02d},448.1,1013.2" for i in range(12)))

    manifest = pack_burn(d, tmp_path / "packed.npz")
    spectra, wp, P, T, _ = read_packed(tmp_path / "packed.npz")

    mask = fitted_channel_mask(w, pad=2.0)
    assert spectra.shape == (12, mask.sum())
    np.testing.assert_allclose(wp, w[mask], rtol=1e-12)
    np.testing.assert_allclose(spectra, raw[:, mask], rtol=1e-5)
    assert manifest["max_relative_roundtrip_error"] < 1e-5
    np.testing.assert_allclose(T, 448.1, rtol=1e-6)


def test_pack_full_grid_discards_nothing(tmp_path):
    from pyrospectra import pack_burn, read_packed

    d = tmp_path / "burn"
    (d / "Spectra").mkdir(parents=True)
    w = np.arange(900.0, 1000.0, 0.241)
    raw = np.array([np.linspace(1000, 1100, w.size) + i for i in range(5)])
    for i, row in enumerate(raw):
        np.savetxt(d / "Spectra" / f"spectrum_{i}.prn",
                   np.column_stack([w, row]), fmt="%.4f %.4f")
    (d / "PT_Log.txt").write_text("2026-03-01,10:00:00,448.1,1013.2\n")

    pack_burn(d, tmp_path / "full.npz", windows=False)
    spectra, wp, _, _, _ = read_packed(tmp_path / "full.npz")
    np.testing.assert_allclose(wp, w, rtol=1e-12)
    np.testing.assert_allclose(spectra, raw, rtol=1e-5)
