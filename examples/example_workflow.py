"""
Complete analysis pipeline for one burn: spectra -> concentrations -> emission factors.

Adjust the paths and the FUEL / XSEC_PATHS constants at the top. Cross-section data are
not bundled with the package; see DATA_SOURCES.md for where to obtain them.
"""

from pathlib import Path

import numpy as np

from pyrospectra import (
    align_datetime, emission_factors, generate_reference, get_compounds, l_curve,
    lasso_inversion, process_spectra, read_data, save_results, summarise,
    temporally_regularised_inversion,
)

# ---------------------------------------------------------------------------
DATA_DIR = Path("./data/burns/peat_01")
RESULT_DIR = Path("./results/peat_01")
FUEL = "boreal_peat"                 # key into registry.CARBON_FRACTIONS (Table 1)
N_PREIGNITION = 20                   # stabilised scans before ignition
SIGMA_INSTRUMENT = 0.5               # cm^-1, Gaussian standard deviation
PENALTY = "paper"                    # 'paper' = Eq. 3; 'legacy' = v1.0

# Every species whose databank is 'xsec' needs a file here, or generation raises.
XSEC_PATHS = {
    # 'C3H6O':   'data/xsec/acetone.xsc',
    # 'C5H8':    'data/xsec/isoprene.xsc',
    # 'C4H4O':   'data/xsec/furan.xsc',
    # 'C2H6O':   'data/xsec/ethanol.txt',
    # 'CH3COOH': 'data/xsec/acetic_acid.txt',
    # 'HNO2':    'data/xsec/nitrous_acid.txt',
    # 'CH3CHO':  'data/xsec/acetaldehyde.txt',
}
# ---------------------------------------------------------------------------


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72, "\n1. LOADING DATA\n", "=" * 72, sep="")
    spectra, wavenumbers, pressure, temperature, datetime = read_data(DATA_DIR)

    print("=" * 72, "\n2. REFERENCE SPECTRA\n", "=" * 72, sep="")
    # Restrict to the species you actually have data for. Dropping the xsec species
    # entirely is fine - but then say so in the paper, rather than letting them vanish.
    compounds = get_compounds()
    missing = [s for s, v in compounds.items()
               if v["databank"] == "xsec" and s not in XSEC_PATHS]
    if missing:
        print(f"  no cross-section file for {missing} - excluding them from this run")
        compounds = {k: v for k, v in compounds.items() if k not in missing}

    reference, full_reference, mask, provenance = generate_reference(
        result_dir=RESULT_DIR, emission_species=compounds, w=wavenumbers,
        P=pressure, T=temperature, sigma=SIGMA_INSTRUMENT, xsec_paths=XSEC_PATHS)

    print("=" * 72, "\n3. OBSERVED SPECTRA\n", "=" * 72, sep="")
    observed, _ = process_spectra(spectra, mask, RESULT_DIR,
                                  n_preignition=N_PREIGNITION)

    print("=" * 72, "\n4. SPECIES IDENTIFICATION\n", "=" * 72, sep="")
    reference, full_reference, observed, species, lasso_score = lasso_inversion(
        reference, full_reference, observed, compounds, seed=42)

    print("=" * 72, "\n5. REGULARISATION PARAMETER\n", "=" * 72, sep="")
    curve = l_curve(reference, observed, np.logspace(-8, -1, 40), penalty=PENALTY)
    gamma = curve["gamma_optimal"]
    print(f"  L-curve corner at gamma = {gamma:.3e} (penalty={PENALTY!r})")
    print("  NOTE: re-run this per fuel type. The corner for smouldering peat sits at "
          "a\n  higher gamma than for flaming agricultural residues, and the corner "
          "for\n  penalty='paper' is not the corner for penalty='legacy'.")

    print("=" * 72, "\n6. RETRIEVAL\n", "=" * 72, sep="")
    result = temporally_regularised_inversion(
        reference, observed, gamma, RESULT_DIR, list(species), penalty=PENALTY)

    spread = result.effective_smoothing
    worst = max(spread, key=spread.get)
    print(f"  most heavily constrained species: {worst} "
          f"(gamma*<mu>/G_ii = {spread[worst]:.2g})")

    print("=" * 72, "\n7. EMISSION FACTORS\n", "=" * 72, sep="")
    ef = emission_factors(result.concentrations, result.species, fuel=FUEL,
                          n_background=N_PREIGNITION, uncertainty=result.uncertainty)
    print(summarise(ef))

    print("\n" + "=" * 72, "\n8. SAVING\n", "=" * 72, sep="")
    save_results(result, datetime=align_datetime(datetime, lasso_score),
                 result_dir=RESULT_DIR, emission_result=ef)

    _plot(result, ef, RESULT_DIR)
    print("Done.")


def _plot(result, ef, result_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    species = result.species
    fig, axes = plt.subplots(len(species), 1, figsize=(11, 1.8 * len(species)),
                             sharex=True)
    axes = np.atleast_1d(axes)
    t = np.arange(result.concentrations.shape[1])
    for ax, name, conc, unc in zip(axes, species, result.concentrations,
                                   result.uncertainty):
        ax.plot(t, conc, lw=0.9)
        ax.fill_between(t, conc - unc, conc + unc, alpha=0.3)
        ax.set_ylabel(f"{name}\nppm", fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Time step")
    fig.tight_layout()
    fig.savefig(Path(result_dir) / "concentration_timeseries.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(ef["MCE"], lw=0.9)
    ax.axhline(0.9, ls="--", c="k", lw=0.8)
    ax.set_ylabel("MCE")
    ax.set_xlabel("Time step")
    ax.set_title(f"MCE = {ef['MCE_mean']:.3f} +/- {ef['MCE_sd']:.3f}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(result_dir) / "mce.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
