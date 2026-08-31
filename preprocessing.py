"""
Spectral preprocessing: baseline estimation, absorbance conversion, and the
temporal difference operators used by the regularised retrieval.
"""

import os
import warnings

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

def get_baseline(y, lam=5e7, p=0.02, niter=10):
    """
    Optimised asymmetric least squares (O-ALS) baseline.

    Minimises ``||W(y - z)||^2 + lam ||D2 z||^2`` with asymmetric weights, so the
    baseline tracks the lower envelope of the spectrum.

    Kept sparse throughout. v1.0 called ``.toarray()`` on the second-difference matrix
    and used ``np.diag(w)``, so it allocated three dense (L, L) arrays: on the MG5 grid
    (~29,900 channels between 800 and 8000 cm^-1 at 0.241 cm^-1) that is ~7 GB each and
    the call cannot complete. The arithmetic below is identical; only the storage
    differs.

    References
    ----------
    Dong & Xu (2024), Measurement 233, 114731.
    Eilers (2003), Analytical Chemistry 75(14), 3631-3636.
    """
    y = np.asarray(y, dtype=float)
    L = y.size
    if L < 3:
        raise ValueError("spectrum too short for a second-difference baseline")

    # Second-difference operator, (L-2, L), sparse.
    D2 = sp.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(L - 2, L), format="csc")
    DTD = lam * (D2.T @ D2)

    w = np.ones(L)
    baseline = y.copy()
    for _ in range(niter):
        Z = (sp.diags(w, format="csc") + DTD).tocsc()
        baseline = spsolve(Z, w * y)
        w = p * (y < baseline) + (1 - p) * (y > baseline)
    return baseline


def estimate_background(spectra, n_preignition=None, lam=5e7, p=0.02, niter=10,
                        reducer="median"):
    """
    Single-beam background I0 from the pre-ignition block.

    Section A1 of the manuscript takes I0 from "the initial pre-ignition spectra -
    recorded after gas cell conditions have stabilised". v1.0 used ``spectra[0]`` alone,
    so I0 carried the full detector noise of one 4-second scan straight into every
    absorbance in the burn. Averaging the stabilised pre-ignition block reduces that by
    ~sqrt(n).

    Parameters
    ----------
    spectra : np.ndarray, shape (Nt, Nl)
    n_preignition : int, optional
        Number of leading spectra to combine. Default 1, i.e. v1.0 behaviour, so that
        nothing changes unless you say how many pre-ignition scans you have.
    reducer : {'median', 'mean'}

    Returns
    -------
    np.ndarray, shape (Nl,)
    """
    spectra = np.asarray(spectra, dtype=float)
    n = 1 if n_preignition is None else int(n_preignition)
    if n < 1 or n > spectra.shape[0]:
        raise ValueError(f"n_preignition={n} outside 1..{spectra.shape[0]}")
    if n == 1:
        block = spectra[0]
    else:
        block = (np.median(spectra[:n], axis=0) if reducer == "median"
                 else np.mean(spectra[:n], axis=0))
    return get_baseline(block, lam=lam, p=p, niter=niter)


def process_spectra(spectra, mask, result_dir, n_preignition=None,
                    plot_every=None):
    """
    Convert raw single-beam spectra to decadic absorbance and apply the window mask.

    ``A = -log10(I / I0)``, matching the decadic convention used for the reference
    matrix (see :mod:`pyrospectra.conventions`).

    Parameters
    ----------
    plot_every : int or None
        Write a QC plot every N spectra. v1.0 wrote one PDF per timestep, which for a
        666-spectrum burn is 666 files and dominates the runtime. Default None (no
        plots); set e.g. 50 for a sample.

    Returns
    -------
    observed_spectra : np.ndarray, shape (Nt, mask.sum())
    full_observed_spectra : np.ndarray, shape (Nt, Nl)
    """
    print("Processing spectra")
    spectra = np.asarray(spectra, dtype=float)
    baseline = estimate_background(spectra, n_preignition=n_preignition)

    safe_baseline = np.where(baseline <= 0, np.nan, baseline)
    safe_spectra = np.where(spectra <= 0, np.nan, spectra)
    with np.errstate(divide="ignore", invalid="ignore"):
        full_observed = np.nan_to_num(-np.log10(safe_spectra / safe_baseline))

    os.makedirs(f"{result_dir}/results", exist_ok=True)
    np.save(f"{result_dir}/results/full_obs.npy", full_observed)

    observed = full_observed[:, mask]

    if plot_every:
        _plot_spectra(observed, result_dir, every=int(plot_every))
    return observed, full_observed


# ---------------------------------------------------------------------------
# Temporal operators
# ---------------------------------------------------------------------------

def create_smoother(N):
    """
    The operator D of Appendix A4: the (N, N) second-difference matrix with reflecting
    boundaries, so that ``(Dc)_t = 2 c_t - c_{t-1} - c_{t+1}``.

    Note this is the *operator*, not the penalty. The penalty of Eq. 3 is ||Dc||^2,
    which contributes ``gamma * D.T @ D`` to the normal equations. v1.0 added
    ``gamma * D``. See :func:`penalty_matrix`.
    """
    D = 2 * np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1)
    D[0, 0] = 1
    D[N - 1, N - 1] = 1
    return D


def penalty_matrix(N, form="paper", sparse=True):
    """
    Temporal smoothness penalty contributed to the normal equations.

    form='paper'  ->  D.T @ D  with D from :func:`create_smoother`.
                      This is ||Dc||^2 exactly as written in Eq. 3 and Appendix A4:
                      a penalty on curvature squared.

    form='legacy' ->  D itself, reproducing PyroSpectra v1.0, which added
                      ``lambda_ * D_mat`` rather than ``lambda_ * D^T D``. Because D is
                      the path-graph Laplacian, c^T D c = sum_t (c_{t+1} - c_t)^2, so
                      v1.0 in fact applied a *first-order* smoothness penalty - which is
                      what the v1.0 README describes, and is a perfectly reasonable
                      prior, but is one power of D away from Eq. 3.

    The two are not on the same scale: gamma values are NOT transferable between them,
    and the L-curve must be re-run when the form is changed. The corner sits at a
    markedly different gamma for each.
    """
    D = create_smoother(N)
    M = D.T @ D if form == "paper" else D if form == "legacy" else None
    if M is None:
        raise ValueError(f"Unknown penalty form {form!r}; use 'paper' or 'legacy'")
    return sp.csc_matrix(M) if sparse else M


def penalty_eigenvalues(N, form="paper"):
    """
    Eigen-decomposition of the temporal penalty, used by the fast exact solver.

    D is symmetric, so ``D = U diag(nu) U^T`` and ``D^T D = U diag(nu^2) U^T`` share the
    eigenvectors U. Returning (mu, U) for either form lets one solver cover both.
    """
    nu, U = np.linalg.eigh(create_smoother(N))
    mu = nu ** 2 if form == "paper" else nu
    return np.clip(mu, 0.0, None), U


def build_A_matrix(spectra, Ns, Nl, Nt):
    """
    Block design matrix A, shape (Nl*Nt, Ns*Nt), column-ordered species-major:
    column ``i*Nt + j`` holds species i's reference at timestep j.

    Retained for the L-curve helper and for tests. The production solver in
    :mod:`pyrospectra.inversion` never forms A - it uses the identity
    ``A^T A = kron(R R^T, I_Nt)`` and ``A^T y = R @ obs.T``.
    """
    S = []
    for i in range(Ns):
        a = sp.lil_matrix((Nl * Nt, Nt), dtype=np.float64)
        for j in range(Nt):
            a[(j * Nl):(j + 1) * Nl, j] = spectra[i, :].reshape(-1, 1)
        S.append(a)
    return sp.hstack(S).tocsc()


def _plot_spectra(spectra, result_dir, every=50):
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    os.makedirs(f"{result_dir}/processed_data", exist_ok=True)
    for i in range(0, len(spectra), every):
        plt.figure(figsize=(10, 3.2))
        plt.plot(spectra[i], lw=0.6)
        plt.xlabel("Fitted channel index")
        plt.ylabel("Absorbance")
        plt.title(f"Processed spectrum {i}")
        plt.tight_layout()
        plt.savefig(f"{result_dir}/processed_data/{i}.png", dpi=110)
        plt.close()


__all__ = ["get_baseline", "estimate_background", "process_spectra",
           "create_smoother", "penalty_matrix", "penalty_eigenvalues",
           "build_A_matrix"]
