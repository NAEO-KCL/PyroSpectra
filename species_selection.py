"""
Automated species identification by lasso regression (Section 2.3.3, Appendix A3).

A species whose lasso coefficient is zero at every sampled time-step is judged
practically undetectable and removed from the reference matrix. The sampled time-steps
are then excluded from the final retrieval to avoid overfitting.

Changes from v1.0
-----------------
* The sampled time-steps were drawn with an unseeded ``random.sample``, so the species
  set - and therefore which spectra entered the retrieval - differed between runs of
  the same data. A seed is now required (default 42) and is recorded in the result.
* The indices of the removed time-steps are returned, so that the datetime array can be
  aligned to the retained spectra. v1.0's example truncated the *tail* of the datetime
  array instead, mislabelling every spectrum after the first removed one.
* Lasso coefficients are constrained non-negative by default: they stand for
  concentrations, and allowing a negative coefficient lets one species' noise be
  cancelled by another's, which is exactly the misattribution the step exists to stop.
"""

import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score
from joblib import Parallel, delayed

from .registry import CORE_SPECIES


def fit_lasso(reference_spectra, observed_spectra, timesteps, positive=True,
              cv=5, n_jobs=-1):
    """
    Fit a lasso to each sampled time-step.

    Parameters
    ----------
    reference_spectra : np.ndarray, shape (Ns, Nl)
    observed_spectra : np.ndarray, shape (Nt, Nl)
    timesteps : sequence of int

    Returns
    -------
    coefficients : np.ndarray, shape (Ns, len(timesteps))
    diagnostics : dict
    """
    design = np.asarray(reference_spectra, dtype=float).T      # (Nl, Ns)
    obs = np.asarray(observed_spectra, dtype=float)

    def one(t):
        model = LassoCV(cv=cv, fit_intercept=False, positive=positive, n_jobs=1)
        model.fit(design, obs[t])
        pred = model.predict(design)
        return (model.coef_, float(model.alpha_),
                float(r2_score(obs[t], pred)),
                float(np.sqrt(mean_squared_error(obs[t], pred))))

    out = Parallel(n_jobs=n_jobs)(delayed(one)(t) for t in timesteps)
    coef = np.column_stack([o[0] for o in out])
    return coef, {"alpha": [o[1] for o in out], "R2": [o[2] for o in out],
                  "RMSE": [o[3] for o in out], "timesteps": list(timesteps)}


def lasso_inversion(reference_spectra, full_reference_spectra, observed_spectra,
                    emission_species, n_samples=None, seed=42, positive=True,
                    core_species=CORE_SPECIES, remove_sampled=True):
    """
    Identify detectably present species and drop the rest from the reference matrix.

    Parameters
    ----------
    n_samples : int, optional
        Number of time-steps to sample. Appendix A3 specifies 10 to 20; default is
        ``min(20, Nt)``.
    seed : int
        Reproducibility. Recorded in the returned diagnostics.
    remove_sampled : bool
        Exclude the sampled time-steps from the returned observations, per Appendix A3.
        Note this leaves gaps in an otherwise evenly spaced series, which the temporal
        difference operator treats as though they were adjacent; with 10-20 removed from
        several hundred the distortion is small, but set False to avoid it entirely.

    Returns
    -------
    reference_spectra, full_reference_spectra : filtered to detected species
    observed_spectra : with sampled time-steps removed (if requested)
    new_emission_species : dict
    lasso_score : dict, including 'dropped_timesteps' and 'kept_timesteps'
    """
    R = np.asarray(reference_spectra, dtype=float)
    obs = np.asarray(observed_spectra, dtype=float)
    Ns, Nt = R.shape[0], obs.shape[0]
    names = list(emission_species)
    if len(names) != Ns:
        raise ValueError(f"{len(names)} species names for {Ns} reference rows")

    n_samples = min(20, Nt) if n_samples is None else min(int(n_samples), Nt)
    rng = np.random.default_rng(seed)
    sampled = np.sort(rng.choice(Nt, size=n_samples, replace=False))

    print(f"Lasso species identification: {n_samples} of {Nt} time-steps, seed={seed}")
    coef, diag = fit_lasso(R, obs, sampled, positive=positive)

    detected = np.any(np.abs(coef) > 0, axis=1)
    keep_idx = [i for i in range(Ns)
                if detected[i] or names[i] in set(core_species)]
    kept = [names[i] for i in keep_idx]
    dropped = [n for n in names if n not in kept]
    forced = [names[i] for i in keep_idx if not detected[i]]

    print(f"  detected: {[n for n in kept if n not in forced]}")
    if forced:
        print(f"  retained as core species despite zero coefficients: {forced}")
    if dropped:
        print(f"  dropped as undetectable: {dropped}")

    if remove_sampled:
        mask = np.ones(Nt, dtype=bool)
        mask[sampled] = False
        obs_out = obs[mask]
        kept_timesteps = np.flatnonzero(mask)
    else:
        obs_out = obs
        kept_timesteps = np.arange(Nt)

    diag.update({
        "coefficients": coef, "seed": seed, "positive": positive,
        "species": names, "detected": kept, "dropped": dropped,
        "forced_core": forced,
        "dropped_timesteps": sampled.tolist(),
        "kept_timesteps": kept_timesteps,
    })

    new_species = {k: emission_species[k] for k in kept}
    return (R[keep_idx], np.asarray(full_reference_spectra)[keep_idx], obs_out,
            new_species, diag)


def filter_compounds(results, compounds, always_present=CORE_SPECIES):
    """Filter a compound dictionary on lasso coefficients."""
    present = [k for k, v in zip(compounds, results) if np.any(np.abs(v) > 0)]
    for s in always_present:
        if s in compounds and s not in present:
            present.append(s)
    return {k: compounds[k] for k in compounds if k in present}


__all__ = ["lasso_inversion", "fit_lasso", "filter_compounds"]
