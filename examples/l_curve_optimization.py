"""
Regularisation parameter selection, and why the two penalty forms need separate curves.

Eq. 3 of the manuscript is ||Dc||^2, contributing gamma * D^T D to the normal equations.
PyroSpectra v1.0 added gamma * D instead - a first-order smoothness penalty rather than
a curvature one. Both are reasonable priors, but they are different estimators and the
L-curve corner sits at a different gamma for each, so a gamma chosen under one form
must not be reused under the other.

This script plots both curves on the same axes so the difference is visible.
"""

from pathlib import Path

import numpy as np

from pyrospectra import (
    generate_reference, get_compounds, l_curve, lasso_inversion, process_spectra,
    read_data, temporally_regularised_inversion,
)

DATA_DIR = Path("./data/burns/peat_01")
RESULT_DIR = Path("./results/l_curve")
GAMMAS = np.logspace(-8, -1, 40)


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    spectra, w, P, T, _ = read_data(DATA_DIR)
    compounds = get_compounds()
    compounds = {k: v for k, v in compounds.items() if v["databank"] != "xsec"}

    reference, full_reference, mask, _ = generate_reference(
        RESULT_DIR, compounds, w, P, T, sigma=0.5)
    observed, _ = process_spectra(spectra, mask, RESULT_DIR, n_preignition=20)
    reference, _, observed, species, _ = lasso_inversion(
        reference, full_reference, observed, compounds, seed=42)

    curves = {form: l_curve(reference, observed, GAMMAS, penalty=form)
              for form in ("paper", "legacy")}

    print(f"\n{'penalty':<10}{'gamma at corner':>18}")
    for form, c in curves.items():
        print(f"{form:<10}{c['gamma_optimal']:>18.4e}")
    ratio = curves["legacy"]["gamma_optimal"] / curves["paper"]["gamma_optimal"]
    print(f"\nThe corners differ by a factor of {ratio:.3g}. A gamma tuned under one "
          "form\nis not the same constraint under the other.")

    _plot(curves, RESULT_DIR)

    best = curves["paper"]["gamma_optimal"]
    result = temporally_regularised_inversion(
        reference, observed, best, RESULT_DIR, list(species), penalty="paper")
    print(f"\nRetrieved at gamma = {best:.3e}: median 1-sigma "
          f"{np.median(result.uncertainty):.3g} ppm")


def _plot(curves, result_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for form, c in curves.items():
        k = c["corner_index"]
        ax1.loglog(c["residual_norm"], c["solution_norm"], lw=1.6,
                   label=f"{form} (corner {c['gamma_optimal']:.1e})")
        ax1.loglog(c["residual_norm"][k], c["solution_norm"][k], "o", ms=9)
        ax2.semilogx(c["gamma"], c["penalty_norm"] / c["penalty_norm"].max(), lw=1.6,
                     label=form)
        ax2.axvline(c["gamma_optimal"], ls="--", lw=1)

    ax1.set_xlabel(r"Residual norm $\|Ac - y\|$")
    ax1.set_ylabel(r"Solution norm $\|c\|$")
    ax1.set_title("L-curve")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3, which="both")

    ax2.set_xlabel(r"$\gamma$")
    ax2.set_ylabel(r"$\|Dc\|$ (normalised)")
    ax2.set_title("Roughness of the retrieved series")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(Path(result_dir) / "l_curve_comparison.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
