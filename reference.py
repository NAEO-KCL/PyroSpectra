"""
Reference spectra generation.

Builds the reference matrix R (Ns x Nl) whose rows are the absorbance of 1 ppm of each
species over the 5 m cell path, on the observed wavenumber grid, zero outside that
species' fitting windows.

Two pathways feed the same matrix:

  line-by-line   RADIS + HITRAN / HITEMP / GEISA, for molecules with line lists
  cross-section  :mod:`pyrospectra.xsections`, for the heavy VOCs that have none

Both are returned in the package absorbance convention (decadic by default; see
:mod:`pyrospectra.conventions`), which is what makes them mixable. RADIS returns
napierian optical depth and is converted on the way in - v1.0 did not do this, which
scaled every retrieved concentration by 1/ln(10).

Failures are raised, not swallowed. In v1.0 a bare ``except`` turned any failure into an
all-zero reference row, which the non-zero-column mask then silently removed; that is
how eight species vanished from the retrieval without an error being reported.
"""

import json
import os
import warnings

import numpy as np

from .conventions import from_radis_absorbance, check_convention
from .registry import build_compounds
from .xsections import cross_section_reference, CrossSectionError, file_checksum


class ReferenceGenerationError(Exception):
    """Raised when one or more reference spectra could not be generated."""


def gaussian(x, mu, sigma):
    """Gaussian, retained from v1.0 for the RADIS slit function."""
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


# ---------------------------------------------------------------------------
# Line-by-line pathway
# ---------------------------------------------------------------------------

def _lbl_window(molecule, window, T, P, w_obs, sigma_cm1, databank,
                isotope="1", path_length_cm=500.0, mole_fraction=1e-6,
                convention=None):
    """
    One species, one window, via RADIS. Returns (absorbance on w_obs, provenance).

    ``databank`` is passed through to :func:`radis.calc_spectrum`, so it accepts
    'hitran', 'hitemp', 'geisa', or a path to a local ``.par`` file - which is how you
    run this offline once the line lists have been downloaded.
    """
    from radis import calc_spectrum, Spectrum
    from radis.tools import convolve_with_slit

    lo, hi = float(window[0]), float(window[1])
    s = calc_spectrum(
        lo, hi,
        molecule=molecule,
        isotope=isotope,
        pressure=P,
        Tgas=T,
        mole_fraction=mole_fraction,
        path_length=path_length_cm,
        wstep="auto",
        databank=databank,
        warnings={"AccuracyError": "ignore"},
    )
    w, tau = s.get("absorbance", wunit="cm-1")     # RADIS: napierian optical depth

    # Instrument lineshape (RADIS normalises the slit internally).
    x_values = np.arange(-1, 1, 0.001)
    kernel = gaussian(x_values, 0, sigma_cm1)
    kernel /= np.max(kernel)
    w, tau = convolve_with_slit(w, tau, x_values, kernel, wunit="cm-1")

    absorbance = from_radis_absorbance(tau, convention=convention)

    grid = np.asarray(w_obs, dtype=float)
    out = np.interp(grid, w, absorbance, left=0.0, right=0.0)
    out[(grid < lo) | (grid > hi)] = 0.0

    if not np.any(out != 0):
        raise ReferenceGenerationError(
            f"{molecule}: line-by-line reference is identically zero over "
            f"[{lo}, {hi}] cm-1 (databank={databank!r}, isotope={isotope!r}). "
            "Either the window holds no lines for this isotopologue, or it lies "
            "outside the observed grid."
        )

    return out, {
        "pathway": "line-by-line",
        "databank": str(databank),
        "molecule": molecule,
        "isotope": isotope,
        "window_cm1": [lo, hi],
        "radis_convention": "napierian optical depth",
        "converted_to": check_convention(convention),
        "channels_in_window": int(((grid >= lo) & (grid <= hi)).sum()),
        "peak_absorbance_per_ppm": float(np.max(np.abs(out))),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_reference(result_dir, emission_species, w, P, T, sigma=0.5,
                       xsec_paths=None, path_length_cm=500.0, mole_fraction=1e-6,
                       convention=None, lineshape="gaussian",
                       strength_extrapolation=True, cache=True,
                       on_error="raise"):
    """
    Generate the reference matrix for all target species.

    Parameters
    ----------
    result_dir : str or Path
        Written to: ``reference_information/`` (per-species plots, provenance JSON)
        and ``reference_cache/`` (npz cache).
    emission_species : dict
        ``{name: {'bounds': [[wmin, wmax], ...], 'databank': ..., 'name': ...}}``, as
        produced by :func:`pyrospectra.registry.build_compounds`. A bare
        ``{'bounds': ...}`` dict (the v1.0 format) is accepted and assumed to be HITRAN.
    w : np.ndarray
        Observed wavenumber grid (cm^-1).
    P, T : float
        Gas cell pressure (bar) and temperature (K), from the PT log.
    sigma : float
        Instrument lineshape Gaussian standard deviation, cm^-1.
    xsec_paths : dict, optional
        ``{species: path}`` for every species whose databank is ``'xsec'``.
        Missing entries raise: see ``DATA_SOURCES.md`` for how to obtain the files.
    on_error : {'raise', 'warn_and_drop'}
        ``'raise'`` (default) fails loudly listing every species that could not be
        built. ``'warn_and_drop'`` reproduces the v1.0 behaviour of continuing without
        them - but warns rather than staying silent.

    Returns
    -------
    reference_spectra : np.ndarray, shape (Ns, Nl_masked)
    full_storage_mtx : np.ndarray, shape (Ns, Nl)
    mask : np.ndarray of bool, shape (Nl,)
    provenance : dict
    """
    os.makedirs(f"{result_dir}/reference_information", exist_ok=True)
    cache_dir = f"{result_dir}/reference_cache"
    if cache:
        os.makedirs(cache_dir, exist_ok=True)

    conv = check_convention(convention)
    w = np.asarray(w, dtype=float)
    xsec_paths = dict(xsec_paths or {})

    species_names = list(emission_species)
    full = np.zeros((len(species_names), w.size), dtype=float)
    provenance = {
        "absorbance_convention": conv,
        "cell_temperature_K": float(T),
        "cell_pressure_bar": float(P),
        "instrument_sigma_cm1": float(sigma),
        "path_length_cm": float(path_length_cm),
        "reference_mole_fraction": float(mole_fraction),
        "species": {},
    }
    failures = {}

    for i, name in enumerate(species_names):
        spec = emission_species[name]
        bounds = spec["bounds"] if isinstance(spec, dict) else spec
        databank = spec.get("databank", "hitran") if isinstance(spec, dict) else "hitran"
        radis_name = spec.get("name", name) if isinstance(spec, dict) else name
        isotope = str(spec.get("isotope", "1")) if isinstance(spec, dict) else "1"

        key = _cache_key(name, bounds, databank, radis_name, isotope, T, P, sigma,
                         conv, path_length_cm, mole_fraction,
                         xsec_paths.get(name))
        cache_file = os.path.join(cache_dir, f"{name}_{key}.npz") if cache else None

        if cache_file and os.path.exists(cache_file):
            with np.load(cache_file, allow_pickle=True) as z:
                full[i] = z["absorbance"]
                provenance["species"][name] = json.loads(str(z["provenance"]))
                provenance["species"][name]["from_cache"] = True
            continue

        try:
            row, prov = _build_species(
                name, bounds, databank, radis_name, isotope, w, P, T, sigma,
                xsec_paths, path_length_cm, mole_fraction, conv, lineshape,
                strength_extrapolation)
        except Exception as exc:                      # noqa: BLE001 - reported below
            failures[name] = f"{type(exc).__name__}: {exc}"
            continue

        full[i] = row
        provenance["species"][name] = prov
        if cache_file:
            np.savez_compressed(cache_file, absorbance=row,
                                provenance=json.dumps(prov))

    if failures:
        report = "\n".join(f"  {k}: {v}" for k, v in failures.items())
        msg = (f"{len(failures)} of {len(species_names)} reference spectra could not be "
               f"generated:\n{report}\n"
               "For 'xsec' species supply the measured cross-section via xsec_paths= "
               "(see DATA_SOURCES.md). For line-by-line species check the databank and "
               "the molecule name (HITRAN calls formaldehyde H2CO, not CH2O).")
        if on_error == "raise":
            raise ReferenceGenerationError(msg)
        warnings.warn(msg + "\nContinuing without them (on_error='warn_and_drop').",
                      RuntimeWarning)
        keep = [i for i, n in enumerate(species_names) if n not in failures]
        full = full[keep]
        species_names = [species_names[i] for i in keep]
        provenance["dropped"] = failures

    _plot_references(w, full, species_names, result_dir)

    full = np.nan_to_num(full)
    mask = ~np.all(full == 0, axis=0)
    reference_spectra = full[:, mask]

    provenance["species_order"] = species_names
    provenance["n_channels_fitted"] = int(mask.sum())
    with open(f"{result_dir}/reference_information/reference_provenance.json", "w") as fh:
        json.dump(provenance, fh, indent=2, default=str)

    print(f"Reference matrix: {reference_spectra.shape[0]} species x "
          f"{reference_spectra.shape[1]} fitted channels "
          f"({conv} absorbance, {mole_fraction * 1e6:g} ppm, {path_length_cm} cm)")

    return reference_spectra, full, mask, provenance


def _build_species(name, bounds, databank, radis_name, isotope, w, P, T, sigma,
                   xsec_paths, path_length_cm, mole_fraction, conv, lineshape,
                   strength_extrapolation):
    """Build one species' reference row across all its windows."""
    row = np.zeros_like(w)
    windows = []

    if databank == "xsec":
        path = xsec_paths.get(name)
        if path is None:
            raise CrossSectionError(
                f"{name} has no line list and requires a measured cross-section, but no "
                f"file was given. Pass xsec_paths={{'{name}': '/path/to/file.xsc'}}."
            )
        if not os.path.exists(path):
            raise CrossSectionError(f"{name}: cross-section file not found: {path}")
        for bound in bounds:
            piece, prov = cross_section_reference(
                path, bound, w, P, T, sigma,
                path_length_cm=path_length_cm, mole_fraction=mole_fraction,
                lineshape=lineshape, convention=conv,
                strength_extrapolation=strength_extrapolation)
            row = row + piece
            windows.append(prov)
        return row, {"pathway": "cross-section", "databank": "xsec",
                     "molecule": name, "file": str(path),
                     "sha256": file_checksum(path), "windows": windows,
                     "peak_absorbance_per_ppm": float(np.max(np.abs(row)))}

    for bound in bounds:
        piece, prov = _lbl_window(radis_name, bound, T, P, w, sigma, databank,
                                  isotope=isotope, path_length_cm=path_length_cm,
                                  mole_fraction=mole_fraction, convention=conv)
        row = row + piece
        windows.append(prov)
    return row, {"pathway": "line-by-line", "databank": str(databank),
                 "molecule": radis_name, "isotope": isotope, "windows": windows,
                 "peak_absorbance_per_ppm": float(np.max(np.abs(row)))}


def _cache_key(*parts):
    import hashlib
    return hashlib.sha256(repr(parts).encode()).hexdigest()[:12]


def _plot_references(w, full, names, result_dir):
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    for row, name in zip(full, names):
        plt.figure(figsize=(12, 3.2))
        plt.plot(w, row, lw=0.7)
        plt.xlabel("Wavenumber (cm$^{-1}$)")
        plt.ylabel("Absorbance / ppm")
        plt.title(f"{name} reference")
        plt.tight_layout()
        plt.savefig(f"{result_dir}/reference_information/{name}.pdf")
        plt.close()


def get_reference_matrix(emission_species, T, P, W_obs, sigma, result_dir):
    """Deprecated v1.0 entry point. Use :func:`generate_reference`."""
    warnings.warn("get_reference_matrix() is deprecated; use generate_reference()",
                  DeprecationWarning, stacklevel=2)
    ref, full, mask, _ = generate_reference(result_dir, emission_species, W_obs, P, T,
                                            sigma=sigma)
    return full


__all__ = ["generate_reference", "ReferenceGenerationError", "gaussian",
           "get_reference_matrix"]
