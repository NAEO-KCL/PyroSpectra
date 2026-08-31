"""
Temporally regularised concentration retrieval.

Solves, for the whole time series at once,

    c_hat = argmin_c  || A c - y ||^2  +  gamma || D c ||^2                      (Eq. 3)

and reports the posterior 1-sigma uncertainty

    sigma_i = sigma_eps * sqrt( [ (A^T A + gamma D^T D)^-1 ]_ii )                (Eq. 2)

with sigma_eps taken from the RMS of the retrieval fit residuals, as Section 2.3.1
specifies. Setting ``gamma = 0`` reduces the estimator exactly to Classical Least
Squares, which is the benchmark of Section 3.2.

"""

import os
import warnings

import numpy as np

from .preprocessing import penalty_eigenvalues


class RetrievalResult:
    """
    Container for a retrieval.

    Attributes
    ----------
    concentrations : np.ndarray, shape (Ns, Nt)   ppm
    uncertainty : np.ndarray, shape (Ns, Nt)      ppm, 1-sigma, noise-scaled
    species : list of str
    sigma_eps : float                             absorbance noise level
    residuals : np.ndarray, shape (Nt, Nl)        observed - modelled absorbance
    correlation : np.ndarray, shape (Ns, Ns)      per-timestep posterior correlation
    effective_dof : float
    effective_smoothing : dict {species: float}
        ``gamma * <mu> / G_ii`` per species - how hard the temporal penalty pulls
        relative to that species' own information content. A single scalar gamma does
        not smooth all species equally: G_ii scales with the square of the reference
        amplitude, which spans orders of magnitude between a strong absorber like CO2
        and a weak trace VOC, so the same gamma is a far heavier constraint on the
        former. Values >> 1 mean that species is prior-dominated.
    gamma, penalty : retrieval settings

    Unpacks as ``concentrations, uncertainty`` for convenience.
    """

    __slots__ = ("concentrations", "uncertainty", "species", "sigma_eps",
                 "residuals", "correlation", "effective_dof", "gamma", "penalty",
                 "condition_number", "sigma_eps_source", "effective_smoothing")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __iter__(self):
        yield self.concentrations
        yield self.uncertainty

    def __repr__(self):
        ns, nt = self.concentrations.shape
        return (f"<RetrievalResult {ns} species x {nt} timesteps, "
                f"penalty={self.penalty!r}, gamma={self.gamma:g}, "
                f"sigma_eps={self.sigma_eps:.3e}>")

    def as_dict(self):
        """``{species: (concentration, uncertainty)}`` in ppm."""
        return {s: (self.concentrations[i], self.uncertainty[i])
                for i, s in enumerate(self.species)}

    def to_frame(self, datetime=None):
        """Tidy pandas DataFrame of concentrations and their 1-sigma uncertainties."""
        import pandas as pd
        df = pd.DataFrame(self.concentrations.T, columns=list(self.species))
        for i, s in enumerate(self.species):
            df[f"{s}_sigma"] = self.uncertainty[i]
        if datetime is not None:
            df.insert(0, "datetime", np.asarray(datetime)[:df.shape[0]])
        return df


def _decoupled_solve(R, obs, gamma, penalty):
    """
    Exact solve and exact posterior diagonal via the eigenbasis of D.

    Returns concentrations (Ns, Nt), unscaled variance (Ns, Nt), effective dof,
    and the per-timestep Gram matrix G.
    """
    Ns, Nl = R.shape
    Nt = obs.shape[0]

    G = R @ R.T                                  # (Ns, Ns) = A^T A block
    mu, U = penalty_eigenvalues(Nt, form=penalty)
    b = R @ obs.T                                # (Ns, Nt) = A^T y, species-major
    B = b @ U                                    # into the eigenbasis

    Z = np.empty((Ns, Nt))
    inv_diag = np.empty((Ns, Nt))
    dof = 0.0
    eye = np.eye(Ns)

    for k in range(Nt):
        Mk = G + gamma * mu[k] * eye
        try:
            Mk_inv = np.linalg.inv(Mk)
        except np.linalg.LinAlgError as exc:
            raise np.linalg.LinAlgError(
                f"singular normal equations at eigenmode {k} (gamma={gamma:g}). "
                "Two reference rows are probably collinear over the fitted channels; "
                "check the correlation matrix and widen or separate those windows."
            ) from exc
        Z[:, k] = Mk_inv @ B[:, k]
        inv_diag[:, k] = np.diag(Mk_inv)
        dof += float(np.trace(Mk_inv @ G))

    conc = Z @ U.T
    variance = inv_diag @ (U.T ** 2)             # sum_k Minv_ii(k) * U[t,k]^2
    return conc, variance, dof, G


def temporally_regularised_inversion(reference_spectra, residual_spectra, lambda_,
                                     result_dir=None, compound_list=None,
                                     penalty="paper", noise_estimate="cls",
                                     plot_correlation=True, **legacy):
    """
    Retrieve concentration time series.

    Parameters
    ----------
    reference_spectra : np.ndarray, shape (Ns, Nl)
        Absorbance of 1 ppm of each species over the cell path, in the same absorbance
        convention as the observations (decadic by default).
    residual_spectra : np.ndarray, shape (Nt, Nl)
        Baseline-corrected observed absorbance.
    lambda_ : float
        Regularisation strength gamma. Select with :func:`l_curve`; note that the
        optimum differs between ``penalty='paper'`` and ``penalty='legacy'``.
    penalty : {'paper', 'legacy'}
        'paper' applies gamma * D^T D (Eq. 3). 'legacy' applies gamma * D (v1.0).
    noise_estimate : {'cls', 'rms', 'reduced_chi2', float}
        How sigma_eps is obtained.

        'cls' (default) takes the RMS of the *unregularised* fit residuals. This still
        follows Section 2.3.1 in absorbing reference-spectra and forward-model mismatch
        rather than detector noise alone, and it reduces exactly to Eq. 2 when
        gamma = 0, but it is not contaminated by the prior.

        'rms' is the literal reading of Section 2.3.1: the RMS of the residuals of the
        regularised fit itself. Be careful with it. Those residuals contain the
        smoothing bias as well as the noise, and the bias grows with gamma - on a test
        case with 3e-5 absorbance noise, this estimator returns 3.0e-5 at gamma = 0 but
        2.4e-2 at gamma = 1e-3, a factor of 800. Because the reported uncertainty is
        proportional to sigma_eps, using it makes the uncertainty *increase* with
        regularisation strength, which inverts the effect reported in Section 3.2. The
        measurement noise level is a property of the data, not of the prior.

        'reduced_chi2' divides the regularised residuals by the residual degrees of
        freedom, using the effective parameter count of the regularised estimator; this
        partly but not wholly compensates the same effect. A float sets sigma_eps
        directly, e.g. from a measured detector noise floor.

    Returns
    -------
    RetrievalResult
        Unpacks as ``(concentrations, uncertainty)``, both shaped (Ns, Nt), in ppm.
    """
    if "do_spilu" in legacy:
        warnings.warn(
            "do_spilu is obsolete: the solver is now exact and direct, and no longer "
            "uses an incomplete LU factorisation as though it were a direct solve.",
            DeprecationWarning, stacklevel=2)
    if "post_cov" in legacy:
        warnings.warn("post_cov is obsolete; the posterior diagonal is always exact.",
                      DeprecationWarning, stacklevel=2)

    R = np.asarray(reference_spectra, dtype=float)
    obs = np.asarray(residual_spectra, dtype=float)
    if R.ndim != 2 or obs.ndim != 2:
        raise ValueError("reference_spectra must be (Ns, Nl) and residual_spectra (Nt, Nl)")
    Ns, Nl = R.shape
    Nt = obs.shape[0]
    if obs.shape[1] != Nl:
        raise ValueError(
            f"spectral dimension mismatch: reference has {Nl} channels, observations "
            f"have {obs.shape[1]}. The reference mask and the observation mask must be "
            "the same one returned by generate_reference().")

    species = list(compound_list) if compound_list is not None else [f"S{i}" for i in range(Ns)]
    if len(species) != Ns:
        raise ValueError(f"compound_list has {len(species)} names for {Ns} reference rows")

    gamma = float(lambda_)
    print(f"Tikhonov retrieval: {Ns} species, {Nt} timesteps, {Nl} channels, "
          f"penalty={penalty!r}, gamma={gamma:g}")

    conc, variance, dof, G = _decoupled_solve(R, obs, gamma, penalty)

    # --- residuals and the noise level -------------------------------------
    model = conc.T @ R                              # (Nt, Nl)
    residuals = obs - model
    n_data = residuals.size

    if isinstance(noise_estimate, (int, float)) and not isinstance(noise_estimate, bool):
        sigma_eps = float(noise_estimate)
        source = "supplied"
    elif noise_estimate == "cls":
        if gamma == 0.0:
            sigma_eps = float(np.sqrt(np.mean(residuals ** 2)))
        else:
            conc0, _, _, _ = _decoupled_solve(R, obs, 0.0, penalty)
            sigma_eps = float(np.sqrt(np.mean((obs - conc0.T @ R) ** 2)))
        source = "RMS of unregularised (CLS) fit residuals"
    elif noise_estimate == "rms":
        sigma_eps = float(np.sqrt(np.mean(residuals ** 2)))
        source = "RMS of regularised fit residuals (includes smoothing bias)"
    elif noise_estimate == "reduced_chi2":
        denom = max(n_data - dof, 1.0)
        sigma_eps = float(np.sqrt(np.sum(residuals ** 2) / denom))
        source = "reduced chi-squared of regularised fit"
    else:
        raise ValueError(
            f"Unknown noise_estimate {noise_estimate!r}; use 'cls', 'rms', "
            "'reduced_chi2' or a float")

    if sigma_eps <= 0:
        warnings.warn("residual noise estimated as zero; uncertainties will be zero",
                      RuntimeWarning)

    uncertainty = sigma_eps * np.sqrt(np.clip(variance, 0.0, None))

    # --- diagnostics --------------------------------------------------------
    cond = float(np.linalg.cond(G))
    if cond > 1e10:
        warnings.warn(
            f"per-timestep Gram matrix is ill-conditioned (cond = {cond:.2e}). Some "
            "reference rows are close to collinear over the fitted channels; the "
            "posterior correlations below will show which, and the affected species' "
            "uncertainties should not be read as independent.",
            RuntimeWarning)

    mu_mean = float(np.mean(penalty_eigenvalues(Nt, form=penalty)[0]))
    g_diag = np.clip(np.diag(G), 1e-300, None)
    effective_smoothing = {sp: float(gamma * mu_mean / g)
                           for sp, g in zip(species, g_diag)}
    if gamma > 0:
        vals = np.array(list(effective_smoothing.values()))
        if vals.max() / max(vals.min(), 1e-300) > 100:
            hi = max(effective_smoothing, key=effective_smoothing.get)
            lo = min(effective_smoothing, key=effective_smoothing.get)
            warnings.warn(
                f"the single gamma={gamma:g} constrains species very unevenly: "
                f"{hi} is smoothed {vals.max() / vals.min():.0f}x harder than {lo} "
                "relative to its own information content. Check that no strongly "
                "absorbing species has been flattened - inspect result."
                "effective_smoothing, and re-run the L-curve for this fuel type.",
                RuntimeWarning)

    G_inv = np.linalg.pinv(G)
    sd = np.sqrt(np.clip(np.diag(G_inv), 1e-300, None))
    correlation = G_inv / np.outer(sd, sd)

    if plot_correlation and result_dir is not None:
        _plot_correlation_matrix(correlation, species, result_dir)

    print(f"  sigma_eps = {sigma_eps:.4e} absorbance ({source})")
    print(f"  effective dof = {dof:.1f} of {Ns * Nt}  |  median 1-sigma = "
          f"{np.median(uncertainty):.3g} ppm")

    return RetrievalResult(
        concentrations=conc, uncertainty=uncertainty, species=species,
        sigma_eps=sigma_eps, residuals=residuals, correlation=correlation,
        effective_dof=dof, gamma=gamma, penalty=penalty, condition_number=cond,
        sigma_eps_source=source, effective_smoothing=effective_smoothing)


def classical_least_squares(reference_spectra, residual_spectra, compound_list=None,
                            noise_estimate="rms"):
    """
    Per-spectrum Classical Least Squares - the estimator at the core of MALT and
    OPUS GA, and the benchmark of Section 3.2. Identical to the regularised retrieval
    with gamma = 0, which is how it is implemented, so that a comparison isolates the
    temporal constraint and nothing else.
    """
    return temporally_regularised_inversion(
        reference_spectra, residual_spectra, 0.0, result_dir=None,
        compound_list=compound_list, noise_estimate=noise_estimate,
        plot_correlation=False)


def l_curve(reference_spectra, residual_spectra, gammas=None, penalty="paper"):
    """
    L-curve for the regularisation parameter (Calvetti et al. 2000).

    Returns
    -------
    dict with 'gamma', 'residual_norm', 'solution_norm', 'penalty_norm', 'corner_index',
    'gamma_optimal'.

    The corner is located by maximum distance from the chord joining the endpoints of
    the log-log curve. Re-run this whenever ``penalty`` changes: the corner for
    ``'paper'`` sits at a different gamma from the corner for ``'legacy'``, so the
    values quoted in the manuscript are not transferable between the two forms.
    """
    R = np.asarray(reference_spectra, dtype=float)
    obs = np.asarray(residual_spectra, dtype=float)
    gammas = np.logspace(-8, 0, 40) if gammas is None else np.asarray(gammas, float)
    Nt = obs.shape[0]

    from .preprocessing import create_smoother
    D = create_smoother(Nt)

    res_norm, sol_norm, pen_norm = [], [], []
    for g in gammas:
        conc, _, _, _ = _decoupled_solve(R, obs, float(g), penalty)
        res_norm.append(float(np.linalg.norm(obs - conc.T @ R)))
        sol_norm.append(float(np.linalg.norm(conc)))
        pen_norm.append(float(np.linalg.norm(conc @ D.T)))

    x = np.log10(np.asarray(res_norm))
    y = np.log10(np.asarray(sol_norm))
    xs = (x - x.min()) / max(np.ptp(x), 1e-300)
    ys = (y - y.min()) / max(np.ptp(y), 1e-300)
    p1, p2 = np.array([xs[0], ys[0]]), np.array([xs[-1], ys[-1]])
    seg = p2 - p1
    def _cross2(a, b):
        return a[0] * b[1] - a[1] * b[0]
    dist = [abs(_cross2(seg, p1 - np.array([xi, yi]))) / np.linalg.norm(seg)
            for xi, yi in zip(xs, ys)]
    corner = int(np.argmax(dist))

    return {"gamma": gammas, "residual_norm": np.asarray(res_norm),
            "solution_norm": np.asarray(sol_norm), "penalty_norm": np.asarray(pen_norm),
            "corner_index": corner, "gamma_optimal": float(gammas[corner]),
            "penalty": penalty}


def inversion_residual(ref_spec, obs_spec, result):
    """
    Modelled absorbance and residuals from a :class:`RetrievalResult`.

    v1.0's version took ``np.sqrt(sigma)`` of a quantity that was already 1-sigma, and
    reshaped the concentrations with ``order='F'``. Both are corrected here.
    """
    R = np.asarray(ref_spec, dtype=float)
    obs = np.asarray(obs_spec, dtype=float)
    conc = result.concentrations if isinstance(result, RetrievalResult) else np.asarray(result)
    err = result.uncertainty if isinstance(result, RetrievalResult) else None

    y_model = conc.T @ R
    y_err = (np.sqrt((err.T ** 2) @ (R ** 2)) if err is not None else None)
    return y_model, obs, y_err, obs - y_model


def _plot_correlation_matrix(corr, species, result_dir):
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    os.makedirs(f"{result_dir}/reference_information", exist_ok=True)
    plt.figure(figsize=(1 + 0.55 * len(species), 0.9 + 0.5 * len(species)))
    plt.imshow(corr, cmap="seismic", vmin=-1, vmax=1)
    ticks = np.arange(len(species))
    plt.xticks(ticks, species, rotation=45, ha="right")
    plt.yticks(ticks, species)
    for i in range(len(species)):
        for j in range(len(species)):
            plt.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=6)
    cb = plt.colorbar(pad=0.02)
    cb.set_label("Correlation coefficient", rotation=-90, labelpad=14)
    plt.title("Posterior correlation between retrieved species")
    plt.tight_layout()
    plt.savefig(f"{result_dir}/reference_information/Correlation_Matrix.png", dpi=250)
    plt.close()


__all__ = ["temporally_regularised_inversion", "classical_least_squares", "l_curve",
           "inversion_residual", "RetrievalResult"]
