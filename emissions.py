"""
Emission ratios, emission factors and modified combustion efficiency.

Implements Eqs. 4-7 of the manuscript. There was no equivalent module in v1.0 despite
the changelog listing emission factors as a feature, so this is new code and has not
been cross-checked against the authors' own EF pipeline - please verify the first
burn's numbers against a known result before relying on it.

    ER(X/CO2) = dX / dCO2                                                    (Eq. 4)
    EF_X      = F_C * 1000 * (MM_X / 12) * ER(X/CO2) / C_T                   (Eq. 5)
    C_T       = sum_j  N_j * c_j / c_CO2                                     (Eq. 6)
    MCE       = dCO2 / (dCO2 + dCO)                                          (Eq. 7)

Excess mixing ratios dX are formed against a time-weighted pre-ignition background
measured in the stabilised cell, and emission ratios are computed per time-step rather
than as the slope of a regression over the burn, so the full distribution of ER across
combustion phases is retained (Section 2.4).
"""

import warnings

import numpy as np

from .registry import CARBON_FRACTIONS, carbon_number, molar_mass


def background_mixing_ratio(concentrations, species, n_background,
                            weights=None, reducer="mean"):
    """
    Pre-ignition background for each species.

    Parameters
    ----------
    concentrations : np.ndarray, shape (Ns, Nt)
    species : sequence of str
    n_background : int
        Number of leading timesteps that sampled ambient air before ignition.
    weights : np.ndarray, optional
        Per-timestep weights over the background block, for the time-weighted mean of
        Section 2.4. Default: uniform.

    Returns
    -------
    dict {species: float}
    """
    conc = np.asarray(concentrations, dtype=float)
    n = int(n_background)
    if n < 1 or n > conc.shape[1]:
        raise ValueError(f"n_background={n} outside 1..{conc.shape[1]}")
    block = conc[:, :n]
    if reducer == "median":
        bg = np.median(block, axis=1)
    elif weights is not None:
        w = np.asarray(weights, dtype=float)[:n]
        bg = (block * w).sum(axis=1) / w.sum()
    else:
        bg = block.mean(axis=1)
    return dict(zip(species, bg))


def excess_mixing_ratios(concentrations, species, n_background, **kw):
    """Background-subtracted concentrations, same shape as the input."""
    bg = background_mixing_ratio(concentrations, species, n_background, **kw)
    offsets = np.array([bg[s] for s in species])[:, None]
    return np.asarray(concentrations, dtype=float) - offsets, bg


def modified_combustion_efficiency(delta_co2, delta_co, min_delta_co2=None):
    """
    MCE, Eq. 7. Returns NaN wherever the denominator is not usable.

    ``min_delta_co2`` masks timesteps where the plume is too weak for the ratio to mean
    anything; without it the pre- and post-burn tails dominate the distribution with
    noise-on-noise ratios.
    """
    d2 = np.asarray(delta_co2, dtype=float)
    d1 = np.asarray(delta_co, dtype=float)
    denom = d2 + d1
    mce = np.full(d2.shape, np.nan)
    ok = np.isfinite(denom) & (denom > 0) & (d2 > 0)
    if min_delta_co2 is not None:
        ok &= d2 >= min_delta_co2
    mce[ok] = d2[ok] / denom[ok]
    return mce


def total_carbon(excess, species, reference="CO2", min_reference=None,
                 carbon_numbers=None):
    """
    Total measured carbon C_T, Eq. 6: sum over carbon-bearing species of
    ``N_j * c_j / c_CO2``.

    Species with no registered carbon number are skipped with a warning rather than
    assumed carbon-free, so that an unregistered VOC cannot quietly deflate C_T and
    inflate every emission factor.
    """
    excess = np.asarray(excess, dtype=float)
    species = list(species)
    if reference not in species:
        raise KeyError(f"{reference} is not among the retrieved species {species}")
    ref = excess[species.index(reference)]

    ok = np.isfinite(ref) & (ref > 0)
    if min_reference is not None:
        ok &= ref >= min_reference

    ct = np.full(ref.shape, np.nan)
    total = np.zeros_like(ref)
    used, skipped = [], []
    for i, s in enumerate(species):
        try:
            n = (carbon_numbers or {}).get(s) if carbon_numbers else None
            n = carbon_number(s) if n is None else n
        except KeyError:
            skipped.append(s)
            continue
        if n == 0:
            continue
        total = total + n * np.where(ok, excess[i], 0.0)
        used.append(s)
    if skipped:
        warnings.warn(
            f"no carbon number registered for {skipped}; excluded from C_T. If any is "
            "carbon-bearing, every emission factor here is biased high. Add it to "
            "registry.CARBON_NUMBER.", RuntimeWarning)

    ct[ok] = total[ok] / ref[ok]
    return ct, used


def emission_ratio(excess, species, target, reference="CO2", min_reference=None):
    """ER(X/reference) per timestep, Eq. 4. NaN where the reference is unusable."""
    excess = np.asarray(excess, dtype=float)
    species = list(species)
    x = excess[species.index(target)]
    ref = excess[species.index(reference)]
    ok = np.isfinite(ref) & (ref > 0)
    if min_reference is not None:
        ok &= ref >= min_reference
    er = np.full(x.shape, np.nan)
    er[ok] = x[ok] / ref[ok]
    return er


def emission_factors(concentrations, species, fuel=None, carbon_fraction=None,
                     n_background=10, reference="CO2", min_reference=None,
                     uncertainty=None, weights=None):
    """
    Emission factors for every retrieved species, by carbon mass balance.

    Parameters
    ----------
    concentrations : np.ndarray, shape (Ns, Nt)
        Retrieved mixing ratios in ppm, as returned by the retrieval (already shaped -
        do not flatten).
    species : sequence of str
    fuel : str, optional
        Key into :data:`pyrospectra.registry.CARBON_FRACTIONS` (Table 1), e.g.
        'boreal_peat', 'wheat'.
    carbon_fraction : float, optional
        Overrides ``fuel``. One of the two must be given.
    n_background : int
        Leading timesteps treated as pre-ignition background.
    min_reference : float, optional
        Minimum excess CO2 (ppm) for a timestep to contribute. Strongly recommended:
        without it the smouldering tail contributes ratios of noise to noise. A few
        times the retrieved CO2 1-sigma is a reasonable choice.
    uncertainty : np.ndarray, optional
        Retrieval 1-sigma, shape (Ns, Nt). If given, ``min_reference`` defaults to
        3 x the median CO2 uncertainty.

    Returns
    -------
    dict
        ``{'EF': {species: mean_g_per_kg}, 'EF_se': {...}, 'EF_series': {...},
        'ER': {...}, 'MCE': array, 'MCE_mean': float, 'C_T': array,
        'n_valid': int, 'background': {...}, 'carbon_fraction': float,
        'species_in_carbon_balance': [...]}``

    Notes
    -----
    ``EF_se`` is the standard error of the mean EF across the contributing timesteps -
    the spread of the burn, as reported in Table 3. It is not the retrieval uncertainty
    propagated through Eq. 5, and it does not contain the sampling systematics
    (adsorptive losses, cross-section state extrapolation) discussed in Section 4.3.
    """
    conc = np.asarray(concentrations, dtype=float)
    species = list(species)
    if conc.ndim != 2 or conc.shape[0] != len(species):
        raise ValueError(
            f"concentrations must be (Ns, Nt) with Ns={len(species)}; got {conc.shape}. "
            "If you have a flat vector from the solver, reshape with (Ns, Nt) in C "
            "order - NOT order='F'.")

    if carbon_fraction is None:
        if fuel is None:
            raise ValueError("give either fuel= (Table 1 key) or carbon_fraction=")
        if fuel not in CARBON_FRACTIONS:
            raise KeyError(f"unknown fuel {fuel!r}; known: {sorted(CARBON_FRACTIONS)}")
        carbon_fraction = CARBON_FRACTIONS[fuel]

    excess, background = excess_mixing_ratios(conc, species, n_background,
                                              weights=weights)

    if min_reference is None and uncertainty is not None:
        unc = np.asarray(uncertainty, dtype=float)
        min_reference = 3.0 * float(np.median(unc[species.index(reference)]))

    ct, used = total_carbon(excess, species, reference=reference,
                            min_reference=min_reference)

    ef_series, er_series, ef_mean, ef_se = {}, {}, {}, {}
    for s in species:
        er = emission_ratio(excess, species, s, reference=reference,
                            min_reference=min_reference)
        er_series[s] = er
        try:
            mm = molar_mass(s)
        except KeyError:
            warnings.warn(f"no molar mass for {s!r}; emission factor not computed",
                          RuntimeWarning)
            continue
        ef = carbon_fraction * 1000.0 * (mm / 12.0) * er / ct
        ef_series[s] = ef
        finite = ef[np.isfinite(ef)]
        if finite.size:
            ef_mean[s] = float(np.mean(finite))
            ef_se[s] = float(np.std(finite, ddof=1) / np.sqrt(finite.size)) \
                if finite.size > 1 else float("nan")
        else:
            ef_mean[s] = float("nan")
            ef_se[s] = float("nan")

    d_co2 = excess[species.index(reference)]
    mce = (modified_combustion_efficiency(d_co2, excess[species.index("CO")],
                                          min_delta_co2=min_reference)
           if "CO" in species else np.full(d_co2.shape, np.nan))

    n_valid = int(np.isfinite(ct).sum())
    if n_valid == 0:
        warnings.warn("no timestep passed the excess-CO2 threshold; every EF is NaN. "
                      "Lower min_reference or check the background window.",
                      RuntimeWarning)

    return {
        "EF": ef_mean, "EF_se": ef_se, "EF_series": ef_series, "ER": er_series,
        "MCE": mce, "MCE_mean": float(np.nanmean(mce)) if np.any(np.isfinite(mce)) else float("nan"),
        "MCE_sd": float(np.nanstd(mce)) if np.any(np.isfinite(mce)) else float("nan"),
        "C_T": ct, "n_valid": n_valid, "background": background,
        "carbon_fraction": float(carbon_fraction),
        "species_in_carbon_balance": used, "min_reference": min_reference,
    }


def summarise(result, digits=2):
    """One-line-per-species text summary of an :func:`emission_factors` result."""
    lines = [f"carbon fraction F_C = {result['carbon_fraction']:.3f}",
             f"MCE = {result['MCE_mean']:.3f} +/- {result['MCE_sd']:.3f} "
             f"({result['n_valid']} timesteps)",
             f"C_T species: {', '.join(result['species_in_carbon_balance'])}",
             "", f"{'species':<10}{'EF / g kg-1':>14}{'s.e.':>10}"]
    for s, v in sorted(result["EF"].items(), key=lambda kv: -(kv[1] if np.isfinite(kv[1]) else -1)):
        lines.append(f"{s:<10}{v:>14.{digits}f}{result['EF_se'].get(s, float('nan')):>10.{digits}f}")
    return "\n".join(lines)


__all__ = ["background_mixing_ratio", "excess_mixing_ratios",
           "modified_combustion_efficiency", "total_carbon", "emission_ratio",
           "emission_factors", "summarise"]
