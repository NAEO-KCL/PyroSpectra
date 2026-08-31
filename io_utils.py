"""
File I/O for MATRIX-MG5 data and retrieval results.
"""

import os
import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:                                  # tqdm is optional
    def tqdm(it, **_kw):
        return it


def read_spectrum(fname):
    """Read one ``.prn`` spectrum (column 1 = intensity)."""
    return np.loadtxt(fname, usecols=[1])


def read_spectra(spectral_data, cutoff=800):
    """
    Read a directory of ``.prn`` files into a (Nt, Nl) array, sorted by filename.

    Filenames are sorted naturally, so ``spectrum_9`` precedes ``spectrum_10``. v1.0
    used a plain lexical sort, which orders them the other way round whenever the index
    is not zero-padded - and a shuffled time axis silently invalidates the temporal
    smoothness constraint.
    """
    spectral_data = Path(spectral_data)
    files = sorted(spectral_data.glob("*.prn"), key=_natural_key)
    if not files:
        raise FileNotFoundError(f"No .prn files found in {spectral_data}")

    wv = np.loadtxt(files[0], usecols=[0])
    spectra = np.array([read_spectrum(f) for f in tqdm(files, desc="Reading spectra")])

    bad = [f.name for f, s in zip(files, spectra) if s.shape != wv.shape]
    if bad:
        raise ValueError(f"{len(bad)} spectra have a different wavenumber grid to "
                         f"{files[0].name}, e.g. {bad[:3]}")

    keep = wv > cutoff
    print(f"Loaded {len(spectra)} spectra, {keep.sum()} channels above {cutoff} cm-1 "
          f"({wv[keep].min():.1f}-{wv[keep].max():.1f}, "
          f"step {np.median(np.diff(wv[keep])):.4f} cm-1)")
    return spectra[:, keep], wv[keep]


def _natural_key(path):
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def get_pt(directory):
    """
    Read the gas cell pressure/temperature log.

    Returns
    -------
    P : float, bar   T : float, K   datetime : np.ndarray or None
    """
    for filename in sorted(os.listdir(directory)):
        if filename.endswith("PT_Log.txt"):
            # MATRIX-MG5 progression logs write: Date, Time, Pressure/mbar, Temperature/degC.
            # (Column order and units confirmed against the raw logs: col 3 ~840 mbar,
            # col 4 ~167 degC == ~440 K, consistent with the heated MG5 cell. v1.1 as
            # shipped named these ["T","P"] in the opposite order and applied no degC->K
            # conversion, which fed RADIS ~840 K / ~0.167 bar. Corrected here.)
            df = pd.read_csv(os.path.join(directory, filename), delimiter=",",
                             names=["Date", "Time", "P_mbar", "T_C"])
            T = float(np.median(df["T_C"])) + 273.15          # deg C -> K
            P = float(np.median(df["P_mbar"])) / 1000.0       # mbar -> bar
            dt = pd.to_datetime(df["Date"] + " " + df["Time"]).to_numpy()
            spread_T = float(np.ptp(df["T_C"]))               # a difference: degC == K
            spread_P = float(np.ptp(df["P_mbar"])) / 1000.0
            print(f"Gas cell: T = {T:.1f} K (range {spread_T:.2f} K), "
                  f"P = {P:.5f} bar (range {spread_P:.5f} bar)")
            if spread_T > 5.0 or spread_P > 0.02:
                import warnings
                warnings.warn(
                    f"cell state varied by {spread_T:.1f} K / {spread_P:.3f} bar over "
                    "the burn, beyond the range Appendix A5 tests as negligible. A "
                    "single fixed reference matrix may not be adequate.", RuntimeWarning)
            return P, T, dt

    print("Warning: no PT log found; assuming 300 K, 1.01325 bar. The reference "
          "spectra will be generated for the wrong cell state if this is not right.")
    return 1.01325, 300.0, None


def read_data(directory):
    """
    Read spectra and cell state from an experiment directory.

    Also accepts a packed ``.npz`` archive written by
    :func:`pyrospectra.pack_burn`, which is ~40x smaller than the raw .prn tree.
    """
    if str(directory).endswith(".npz"):
        from .packing import read_packed
        return read_packed(directory)
    spectra, w = read_spectra(os.path.join(directory, "Spectra"))
    P, T, dt = get_pt(directory)
    if dt is None:
        dt = np.arange(len(spectra))
    return spectra, w, P, T, dt


def get_compounds(file=None, species=None, databank_overrides=None):
    """
    Load compound definitions.

    With no argument, builds them from :mod:`pyrospectra.registry` (Table D1), which is
    preferable to a pickle: the windows, databanks and molecule names stay under version
    control and the file cannot silently drift from the manuscript.
    """
    if file is None:
        from .registry import build_compounds
        compounds = build_compounds(species=species,
                                    databank_overrides=databank_overrides)
        print(f"Built {len(compounds)} compound definitions from the registry")
        return compounds
    with open(file, "rb") as handle:
        compounds = pkl.load(handle)
    print(f"Loaded {len(compounds)} compound definitions from {file}")
    return compounds


def align_datetime(datetime, lasso_score):
    """
    Restrict a datetime array to the time-steps that entered the retrieval.

    ``lasso_inversion`` removes the sampled time-steps from the middle of the series;
    the corresponding datetimes must be removed too, not truncated from the end.
    """
    dt = np.asarray(datetime)
    kept = np.asarray(lasso_score["kept_timesteps"], dtype=int)
    if kept.max(initial=-1) >= dt.size:
        raise ValueError(
            f"datetime has {dt.size} entries but the retrieval used time-step "
            f"{kept.max()}. The PT log and the spectra directory are out of step.")
    return dt[kept]


def save_results(result, datetime=None, result_dir=".", prefix="",
                 emission_result=None):
    """
    Save a :class:`~pyrospectra.inversion.RetrievalResult`.

    Writes ``concentrations.npy`` / ``uncertainties.npy`` shaped (Nt, Ns), a tidy CSV
    with paired ``<species>`` and ``<species>_sigma`` columns, and - if given - the
    emission factor table.

    v1.0 wrote ``concentrations.reshape(Ns, Nt, order='F').T``. The solver's output is
    species-major, so that reshape transposes species against time; the saved CSV bore
    the right column names over the wrong data.
    """
    os.makedirs(result_dir, exist_ok=True)
    conc = np.asarray(result.concentrations)
    unc = np.asarray(result.uncertainty)
    species = list(result.species)

    np.save(f"{result_dir}/{prefix}concentrations.npy", conc.T)
    np.save(f"{result_dir}/{prefix}uncertainties.npy", unc.T)

    df = result.to_frame(datetime=datetime)
    df.to_csv(f"{result_dir}/{prefix}concentrations.csv", index=False)

    meta = {"species": species, "gamma": result.gamma, "penalty": result.penalty,
            "sigma_eps": result.sigma_eps, "effective_dof": result.effective_dof,
            "condition_number": result.condition_number}
    pd.Series(meta).to_json(f"{result_dir}/{prefix}retrieval_metadata.json", indent=2)

    if emission_result is not None:
        ef = pd.DataFrame({
            "EF_g_per_kg": pd.Series(emission_result["EF"]),
            "EF_standard_error": pd.Series(emission_result["EF_se"]),
        })
        ef.index.name = "species"
        ef.to_csv(f"{result_dir}/{prefix}emission_factors.csv")

    print(f"Results saved to {result_dir}")
    return df


__all__ = ["read_data", "read_spectra", "read_spectrum", "get_compounds", "get_pt",
           "save_results", "align_datetime"]
