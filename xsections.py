"""
Measured absorption cross-sections for species without line lists.

Seven of the species reported in the manuscript (HNO2, CH3COOH, ethanol, acetone,
furan, isoprene, acetaldehyde) cannot be synthesised line-by-line: at the pressures and
temperatures of interest their spectra are unresolved band contours rather than
assignable transitions, and the spectroscopic databases distribute them as measured
composite cross-sections. This module reads those data and puts them on exactly the
same footing as the RADIS line-by-line references - same absorbance convention, same
1 ppm / 5 m normalisation, same instrument lineshape, same wavenumber grid - so that a
single reference matrix can mix the two.

Supported input formats
-----------------------
``.xsc``   HITRAN cross-section format (Kochanov et al. 2019). Fixed-width 100-character
           header followed by the cross-section values in cm^2 molecule^-1. One file
           may hold several (T, P) bands.
``.txt``   PNNL / Northwest-Infrared style two-column ASCII (Sharpe et al. 2004),
           tabulated as decadic absorbance per ppm per metre at 296 K, 1 atm. This is
           the form underlying the biomass-burning library of Johnson et al. (2010).

Temperature and pressure
------------------------
Cross-sections are state-dependent, and the MG5 cell runs at 448 K - well above the
range at which most of these species have been measured (PNNL: 278/298/323 K). The
band *shape* is taken from the nearest measured temperature; where two or more
temperatures are available the integrated band *strength* is extrapolated linearly in T
and the contour renormalised to it, since integrated intensity is far better behaved
under extrapolation than the rotational envelope. The extrapolation distance is recorded
in the provenance so it can be carried into the error budget - it is a systematic term
that the posterior covariance does not see.
"""

import hashlib
import os
import warnings

import numpy as np

try:                                  # numpy >= 2.0
    from numpy import trapezoid as _trapz
except ImportError:                   # numpy < 2.0
    from numpy import trapz as _trapz

from .conventions import (
    ATM_TO_PA, BAR_TO_PA, TORR_TO_PA,
    cross_section_to_absorbance, pnnl_to_absorbance,
)


class CrossSectionError(Exception):
    """Raised when cross-section data are missing, malformed or unusable."""


class CrossSectionBand:
    """
    One measured cross-section at a single (T, P).

    Attributes
    ----------
    wavenumber : np.ndarray  (cm^-1, ascending)
    values : np.ndarray      (units given by `units`)
    temperature_k, pressure_bar : float
    resolution_cm1 : float or None   spectral resolution of the measurement
    units : {'cm2/molecule', 'ppm-1 m-1'}
    molecule, source : str
    """

    __slots__ = ("wavenumber", "values", "temperature_k", "pressure_bar",
                 "resolution_cm1", "units", "molecule", "source")

    def __init__(self, wavenumber, values, temperature_k, pressure_bar,
                 units, molecule="", resolution_cm1=None, source=""):
        w = np.asarray(wavenumber, dtype=float)
        v = np.asarray(values, dtype=float)
        if w.shape != v.shape:
            raise CrossSectionError(
                f"{molecule}: grid/value length mismatch ({w.shape} vs {v.shape})"
            )
        order = np.argsort(w)
        self.wavenumber = w[order]
        self.values = v[order]
        self.temperature_k = float(temperature_k)
        self.pressure_bar = float(pressure_bar)
        self.resolution_cm1 = resolution_cm1
        self.units = units
        self.molecule = molecule
        self.source = source

    def __repr__(self):
        return (f"<CrossSectionBand {self.molecule} "
                f"{self.wavenumber[0]:.1f}-{self.wavenumber[-1]:.1f} cm-1 "
                f"T={self.temperature_k:.1f} K P={self.pressure_bar:.4f} bar "
                f"[{self.units}]>")

    def integrated(self, wmin=None, wmax=None):
        """Integral of the band over a window; a proxy for band strength."""
        m = np.ones_like(self.wavenumber, dtype=bool)
        if wmin is not None:
            m &= self.wavenumber >= wmin
        if wmax is not None:
            m &= self.wavenumber <= wmax
        if m.sum() < 2:
            return 0.0
        return float(_trapz(self.values[m], self.wavenumber[m]))


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def read_hitran_xsc(path):
    """
    Read a HITRAN ``.xsc`` cross-section file into a list of :class:`CrossSectionBand`.

    The header is fixed-width; fields are, in order, molecule (20), nu_min (10),
    nu_max (10), n_points (7), T/K (7), P/Torr (6), max cross-section (10),
    resolution (5), common name (15), reserved (4), broadener (3), reference (3).
    """
    path = str(path)
    with open(path, "r", errors="replace") as fh:
        lines = [ln.rstrip("\n") for ln in fh if ln.strip()]

    bands, i = [], 0
    while i < len(lines):
        header = lines[i]
        try:
            molecule = header[0:20].strip()
            wmin = float(header[20:30])
            wmax = float(header[30:40])
            npts = int(header[40:47])
            temperature = float(header[47:54])
            pressure_torr = float(header[54:60])
            resolution = header[70:75].strip()
            resolution = float(resolution) if resolution else None
        except (ValueError, IndexError) as exc:
            raise CrossSectionError(
                f"{os.path.basename(path)}: malformed .xsc header at line {i + 1}:\n"
                f"  {header[:100]!r}\n  ({exc})"
            ) from exc

        if npts <= 1:
            raise CrossSectionError(
                f"{os.path.basename(path)}: header declares {npts} points"
            )

        # Consume value lines until npts values have been read.
        values, i = [], i + 1
        while len(values) < npts:
            if i >= len(lines):
                raise CrossSectionError(
                    f"{os.path.basename(path)}: file ends after {len(values)} of "
                    f"{npts} declared points for {molecule} at {temperature} K"
                )
            values.extend(float(tok) for tok in lines[i].split())
            i += 1
        values = np.asarray(values[:npts], dtype=float)

        bands.append(CrossSectionBand(
            wavenumber=np.linspace(wmin, wmax, npts),
            values=values,
            temperature_k=temperature,
            pressure_bar=pressure_torr * TORR_TO_PA / BAR_TO_PA,
            units="cm2/molecule",
            molecule=molecule,
            resolution_cm1=resolution,
            source=path,
        ))

    if not bands:
        raise CrossSectionError(f"{os.path.basename(path)}: no bands found")
    return bands


def read_pnnl(path, temperature_k=296.15, pressure_bar=ATM_TO_PA / BAR_TO_PA,
              resolution_cm1=0.112, molecule="", units="ppm-1 m-1"):
    """
    Read a PNNL-style two-column ASCII cross-section.

    Comment lines (``#``, ``;``, ``!``) and any leading non-numeric header are skipped.
    PNNL data are tabulated as decadic absorbance per ppm per metre at 296 K, 1 atm;
    pass ``units='cm2/molecule'`` if your copy has already been converted.
    """
    path = str(path)
    w, v = [], []
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if not s or s[0] in "#;!":
                continue
            parts = s.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                a, b = float(parts[0]), float(parts[1])
            except ValueError:
                continue          # header text
            w.append(a)
            v.append(b)

    if len(w) < 2:
        raise CrossSectionError(
            f"{os.path.basename(path)}: fewer than two numeric rows found - is this a "
            "two-column (wavenumber, coefficient) file?"
        )
    return CrossSectionBand(w, v, temperature_k, pressure_bar, units,
                            molecule=molecule or os.path.basename(path),
                            resolution_cm1=resolution_cm1, source=path)


def load_cross_section(path, **kwargs):
    """Dispatch on file extension. Returns a list of bands."""
    ext = os.path.splitext(str(path))[1].lower()
    if ext == ".xsc":
        return read_hitran_xsc(path)
    if ext in (".txt", ".dat", ".csv", ".prn", ".asc"):
        return [read_pnnl(path, **kwargs)]
    raise CrossSectionError(
        f"Unrecognised cross-section extension {ext!r} for {path}. "
        "Expected .xsc (HITRAN) or .txt/.dat/.csv (PNNL two-column)."
    )


# ---------------------------------------------------------------------------
# State selection
# ---------------------------------------------------------------------------

def select_state(bands, temperature_k, pressure_bar, window=None,
                 strength_extrapolation=True):
    """
    Reduce a set of measured bands to a single band at the requested (T, P).

    Pressure is matched to the nearest available value (cross-sections in these
    databases are broadened at, or extrapolated to, a small number of pressures and
    interpolating across them is not meaningful). Temperature is interpolated linearly
    between the two bracketing bands. If the request lies outside the measured range,
    the nearest band supplies the contour and - when ``strength_extrapolation`` is set
    and at least two temperatures exist - the integrated band strength is extrapolated
    linearly in T and the contour renormalised to it.

    Returns
    -------
    band : CrossSectionBand
    provenance : dict
        Records which measurements were used and how far the state was extrapolated.
    """
    if not bands:
        raise CrossSectionError("no bands supplied")

    pressures = np.array([b.pressure_bar for b in bands])
    p_target = pressures[np.argmin(np.abs(pressures - pressure_bar))]
    subset = [b for b in bands if np.isclose(b.pressure_bar, p_target)]
    subset.sort(key=lambda b: b.temperature_k)
    temps = np.array([b.temperature_k for b in subset])

    prov = {
        "measured_temperatures_K": temps.tolist(),
        "measured_pressures_bar": sorted(set(np.round(pressures, 6).tolist())),
        "requested_temperature_K": float(temperature_k),
        "requested_pressure_bar": float(pressure_bar),
        "pressure_used_bar": float(p_target),
        "pressure_mismatch_bar": float(pressure_bar - p_target),
        "resolution_cm1": subset[0].resolution_cm1,
        "source": subset[0].source,
        "units": subset[0].units,
    }

    # Common grid for any combination of bands.
    grid = subset[0].wavenumber
    if window is not None:
        lo, hi = window
        pad = 0.05 * (hi - lo)
        grid = grid[(grid >= lo - pad) & (grid <= hi + pad)]
        if grid.size < 2:
            raise CrossSectionError(
                f"{subset[0].molecule}: no cross-section coverage over window "
                f"{window}; file spans {subset[0].wavenumber[0]:.1f}-"
                f"{subset[0].wavenumber[-1]:.1f} cm-1"
            )

    def on_grid(band):
        return np.interp(grid, band.wavenumber, band.values, left=0.0, right=0.0)

    if len(subset) == 1 or temperature_k <= temps[0] or temperature_k >= temps[-1]:
        # Nearest contour, optionally rescaled to an extrapolated band strength.
        k = int(np.argmin(np.abs(temps - temperature_k)))
        values = on_grid(subset[k])
        prov["temperature_used_K"] = float(temps[k])
        prov["temperature_extrapolation_K"] = float(temperature_k - temps[k])
        prov["interpolated"] = False
        prov["strength_rescaled"] = False

        if strength_extrapolation and len(subset) >= 2:
            areas = np.array([b.integrated(*(window or (None, None))) for b in subset])
            if np.all(areas > 0):
                slope, intercept = np.polyfit(temps, areas, 1)
                target_area = slope * temperature_k + intercept
                current = _trapz(values, grid)
                if target_area > 0 and current > 0:
                    scale = target_area / current
                    # Refuse implausible extrapolations rather than apply them silently.
                    if 0.25 <= scale <= 4.0:
                        values = values * scale
                        prov["strength_rescaled"] = True
                        prov["strength_scale"] = float(scale)
                    else:
                        warnings.warn(
                            f"{subset[0].molecule}: band-strength extrapolation to "
                            f"{temperature_k:.1f} K implies a factor {scale:.2f}; "
                            "refusing to apply. Using the nearest measured contour "
                            "unscaled - treat this species' EF as indicative.",
                            RuntimeWarning,
                        )
                        prov["strength_scale_rejected"] = float(scale)
        if abs(prov["temperature_extrapolation_K"]) > 50.0:
            warnings.warn(
                f"{subset[0].molecule}: cross-section measured at "
                f"{temps[k]:.1f} K but the cell is at {temperature_k:.1f} K "
                f"({prov['temperature_extrapolation_K']:+.0f} K). The band contour is "
                "not corrected for hot-band population; carry this as a systematic "
                "term on this species' emission factor.",
                RuntimeWarning,
            )
    else:
        hi = int(np.searchsorted(temps, temperature_k))
        lo = hi - 1
        f = (temperature_k - temps[lo]) / (temps[hi] - temps[lo])
        values = (1.0 - f) * on_grid(subset[lo]) + f * on_grid(subset[hi])
        prov.update({
            "temperature_used_K": [float(temps[lo]), float(temps[hi])],
            "temperature_extrapolation_K": 0.0,
            "interpolated": True,
            "interpolation_weight": float(f),
            "strength_rescaled": False,
        })

    band = CrossSectionBand(grid, values, temperature_k, p_target,
                            subset[0].units, molecule=subset[0].molecule,
                            resolution_cm1=subset[0].resolution_cm1,
                            source=subset[0].source)
    return band, prov


# ---------------------------------------------------------------------------
# Lineshape and gridding
# ---------------------------------------------------------------------------

def instrument_kernel(step_cm1, sigma_cm1, lineshape="gaussian", truncate=4.0):
    """
    Area-normalised instrument lineshape sampled on a grid of spacing ``step_cm1``.

    ``sigma_cm1`` is a Gaussian standard deviation, matching the ``sigma`` argument of
    v1.0 (which passed 0.5 cm^-1). For ``lineshape='triangular'`` it is interpreted as
    the half-width of the triangle, the classic FTIR response under triangular
    apodisation.
    """
    half = max(truncate * sigma_cm1, step_cm1)
    x = np.arange(-half, half + 0.5 * step_cm1, step_cm1)
    if lineshape == "gaussian":
        k = np.exp(-0.5 * (x / sigma_cm1) ** 2)
    elif lineshape == "triangular":
        k = np.clip(1.0 - np.abs(x) / sigma_cm1, 0.0, None)
    else:
        raise ValueError(f"Unknown lineshape {lineshape!r}; use 'gaussian' or 'triangular'")
    area = _trapz(k, x)
    if area <= 0:
        raise CrossSectionError("degenerate instrument kernel")
    return x, k / area


def apply_instrument_lineshape(band, sigma_cm1, lineshape="gaussian"):
    """
    Degrade a measured cross-section to the instrument resolution.

    Measured cross-sections already carry the resolution of the instrument that
    recorded them. Only the *additional* broadening is applied, in quadrature, so a
    library recorded at 0.112 cm^-1 is not broadened as though it were monochromatic.
    If the library is already coarser than the instrument, nothing is done and a
    warning is issued - deconvolution is not attempted.
    """
    step = float(np.median(np.diff(band.wavenumber)))
    if step <= 0:
        raise CrossSectionError(f"{band.molecule}: non-monotonic wavenumber grid")

    src_sigma = 0.0
    if band.resolution_cm1:
        # Interpret the tabulated resolution as a FWHM; convert to a Gaussian sigma.
        src_sigma = float(band.resolution_cm1) / 2.3548200450309493

    extra = sigma_cm1 ** 2 - src_sigma ** 2
    if extra <= 0:
        warnings.warn(
            f"{band.molecule}: library resolution ({band.resolution_cm1} cm-1) is no "
            f"finer than the instrument (sigma={sigma_cm1} cm-1); leaving the contour "
            "unconvolved rather than deconvolving it.",
            RuntimeWarning,
        )
        return band, {"convolved": False, "source_sigma_cm1": src_sigma}

    eff = float(np.sqrt(extra))
    _, kernel = instrument_kernel(step, eff, lineshape=lineshape)
    # np.convolve sums; multiply by the grid step to approximate the integral.
    smoothed = np.convolve(band.values, kernel, mode="same") * step
    out = CrossSectionBand(band.wavenumber, smoothed, band.temperature_k,
                           band.pressure_bar, band.units, molecule=band.molecule,
                           resolution_cm1=band.resolution_cm1, source=band.source)
    return out, {"convolved": True, "effective_sigma_cm1": eff,
                 "source_sigma_cm1": src_sigma, "lineshape": lineshape}


def band_to_absorbance(band, pressure_bar, temperature_k, path_length_cm=500.0,
                       mole_fraction=1e-6, convention=None):
    """Convert a band to absorbance at 1 ppm (by default) over the cell path length."""
    if band.units == "cm2/molecule":
        return cross_section_to_absorbance(
            band.values, pressure_bar, temperature_k, path_length_cm,
            mole_fraction=mole_fraction, convention=convention)
    if band.units == "ppm-1 m-1":
        return pnnl_to_absorbance(
            band.values, pressure_bar, temperature_k, path_length_cm,
            mole_fraction=mole_fraction, convention=convention)
    raise CrossSectionError(
        f"{band.molecule}: unsupported cross-section units {band.units!r}"
    )


def cross_section_reference(path, window, wavenumber_grid, pressure_bar,
                            temperature_k, sigma_cm1, path_length_cm=500.0,
                            mole_fraction=1e-6, lineshape="gaussian",
                            convention=None, strength_extrapolation=True,
                            reader_kwargs=None):
    """
    Full pathway: file -> reference absorbance on the observed wavenumber grid.

    Returns
    -------
    absorbance : np.ndarray
        Same length as ``wavenumber_grid``, zero outside ``window``.
    provenance : dict
    """
    bands = load_cross_section(path, **(reader_kwargs or {}))
    band, prov = select_state(bands, temperature_k, pressure_bar, window=window,
                              strength_extrapolation=strength_extrapolation)
    band, conv_prov = apply_instrument_lineshape(band, sigma_cm1, lineshape=lineshape)
    prov.update(conv_prov)

    absorbance = band_to_absorbance(band, pressure_bar, temperature_k,
                                    path_length_cm=path_length_cm,
                                    mole_fraction=mole_fraction,
                                    convention=convention)

    grid = np.asarray(wavenumber_grid, dtype=float)
    out = np.interp(grid, band.wavenumber, absorbance, left=0.0, right=0.0)
    lo, hi = window
    out[(grid < lo) | (grid > hi)] = 0.0

    covered = ((grid >= lo) & (grid <= hi)).sum()
    if covered == 0:
        raise CrossSectionError(
            f"{band.molecule}: window {window} does not intersect the measurement grid"
        )
    if not np.any(out != 0):
        raise CrossSectionError(
            f"{band.molecule}: reference is identically zero over {window}. "
            "Check that the file covers this window."
        )

    prov.update({
        "file": str(path),
        "sha256": file_checksum(path),
        "window_cm1": [float(lo), float(hi)],
        "channels_in_window": int(covered),
        "path_length_cm": float(path_length_cm),
        "mole_fraction": float(mole_fraction),
        "peak_absorbance_per_ppm": float(np.max(np.abs(out))),
    })
    return out, prov


def file_checksum(path, algorithm="sha256", chunk=1 << 20):
    """Checksum of a data file, recorded in the provenance so runs are traceable."""
    h = hashlib.new(algorithm)
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


__all__ = [
    "CrossSectionBand", "CrossSectionError", "read_hitran_xsc", "read_pnnl",
    "load_cross_section", "select_state", "instrument_kernel",
    "apply_instrument_lineshape", "band_to_absorbance", "cross_section_reference",
    "file_checksum",
]
