"""
Ground-truth tests for the retrieval.

Each test constructs a problem whose answer is known analytically, so a regression in
the conventions (ordering, penalty form, absorbance base, uncertainty scale) fails
loudly rather than producing plausible-looking numbers. The first three encode the
defects found in v1.0.

Run with:  python -m pytest tests/ -v      (or:  python tests/test_inversion.py)
"""

import numpy as np
import pytest
import scipy.sparse as sp

from pyrospectra.conventions import LN10, from_radis_absorbance, number_density
from pyrospectra.inversion import (
    classical_least_squares, l_curve, temporally_regularised_inversion,
)
from pyrospectra.preprocessing import (
    build_A_matrix, create_smoother, get_baseline, penalty_eigenvalues, penalty_matrix,
)


def synthetic_case(Ns=3, Nl=80, Nt=12, noise=0.0, seed=0):
    """Three separable bands, known concentration series, optional noise."""
    rng = np.random.default_rng(seed)
    k = np.arange(Nl)
    R = np.vstack([
        np.exp(-0.5 * ((k - 15) / 3.0) ** 2),
        np.exp(-0.5 * ((k - 40) / 2.5) ** 2) + 0.4 * np.exp(-0.5 * ((k - 60) / 4.0) ** 2),
        np.exp(-0.5 * ((k - 58) / 6.0) ** 2),
    ])[:Ns] * 1e-3
    C = np.vstack([
        np.linspace(100, 900, Nt),
        np.full(Nt, 250.0),
        np.r_[np.zeros(Nt // 3), 400 * np.ones(Nt // 3), np.zeros(Nt - 2 * (Nt // 3))],
    ])[:Ns]
    obs = C.T @ R
    if noise:
        obs = obs + rng.normal(0, noise, obs.shape)
    return R, C, obs


# ---------------------------------------------------------------------------
# 1. Ordering
# ---------------------------------------------------------------------------

def test_concentrations_are_returned_species_by_time():
    """
    The solution vector is species-major. v1.0's callers reshaped it with order='F',
    which transposes species against time; on this case that maps species 0's ramp
    across all three rows.
    """
    R, C, obs = synthetic_case()
    result = temporally_regularised_inversion(R, obs, 1e-12, compound_list=list("ABC"))

    assert result.concentrations.shape == C.shape
    np.testing.assert_allclose(result.concentrations, C, rtol=1e-4, atol=1e-2)

    # The v1.0 reshape must NOT reproduce the truth, or the test is vacuous.
    wrong = result.concentrations.ravel().reshape(C.shape, order="F")
    assert not np.allclose(wrong, C, rtol=1e-2, atol=1.0)


def test_species_order_matches_compound_list():
    R, C, obs = synthetic_case()
    names = ["first", "second", "third"]
    result = temporally_regularised_inversion(R, obs, 1e-12, compound_list=names)
    assert result.species == names
    for i, n in enumerate(names):
        np.testing.assert_allclose(result.as_dict()[n][0], C[i], rtol=1e-4, atol=1e-2)


# ---------------------------------------------------------------------------
# 2. Penalty form
# ---------------------------------------------------------------------------

def test_paper_penalty_is_D_transpose_D():
    """Eq. 3 is ||Dc||^2, contributing gamma * D^T D - not gamma * D as in v1.0."""
    N = 9
    D = create_smoother(N)
    np.testing.assert_allclose(penalty_matrix(N, "paper").toarray(), D.T @ D)
    np.testing.assert_allclose(penalty_matrix(N, "legacy").toarray(), D)
    assert not np.allclose(D.T @ D, D)


def test_penalty_eigenvalues_match_explicit_matrix():
    """The fast solver's eigen-decoupling must reproduce the explicit penalty."""
    N = 11
    for form in ("paper", "legacy"):
        mu, U = penalty_eigenvalues(N, form)
        np.testing.assert_allclose(U @ np.diag(mu) @ U.T,
                                   penalty_matrix(N, form).toarray(), atol=1e-10)


def test_penalty_forms_give_different_answers():
    """If they agreed, gamma would be transferable between them. It is not."""
    R, _, obs = synthetic_case(noise=2e-5, seed=3)
    a = temporally_regularised_inversion(R, obs, 1e-3, penalty="paper").concentrations
    b = temporally_regularised_inversion(R, obs, 1e-3, penalty="legacy").concentrations
    assert np.max(np.abs(a - b)) > 1e-6


# ---------------------------------------------------------------------------
# 3. Uncertainty scaling
# ---------------------------------------------------------------------------

def test_uncertainty_scales_with_noise():
    """
    v1.0 returned sqrt(diag(C^-1)) with no sigma_eps factor, so the reported 1-sigma
    was identical for clean and noisy data. Doubling the noise must double it.
    """
    R, _, obs_clean = synthetic_case()
    rng = np.random.default_rng(7)
    n = rng.normal(0, 1.0, obs_clean.shape)
    r1 = classical_least_squares(R, obs_clean + 1e-5 * n)
    r2 = classical_least_squares(R, obs_clean + 2e-5 * n)
    ratio = np.median(r2.uncertainty / r1.uncertainty)
    assert 1.8 < ratio < 2.2, f"uncertainty scaled by {ratio:.2f}, expected ~2"


def test_uncertainty_reduces_to_equation_2_when_unregularised():
    """
    With gamma = 0 the posterior must be exactly sigma_eps * sqrt(diag((R R^T)^-1)),
    which is Eq. 2 of the manuscript.
    """
    R, _, obs = synthetic_case(noise=1e-5, seed=11)
    r = classical_least_squares(R, obs, compound_list=list("ABC"))
    expected = r.sigma_eps * np.sqrt(np.diag(np.linalg.inv(R @ R.T)))
    for i in range(R.shape[0]):
        np.testing.assert_allclose(r.uncertainty[i], expected[i], rtol=1e-8)


def smooth_case(Nl=80, Nt=80, noise=3e-5, seed=5):
    """
    Smoothly varying concentrations - the regime the smoothness prior assumes. With a
    ~12 s cell residence time and 4 s spectra, real series are correlated between
    consecutive samples, which is the physical justification given in Section 2.3.4.
    """
    rng = np.random.default_rng(seed)
    R, _, _ = synthetic_case(Nl=Nl, Nt=Nt)
    t = np.linspace(0, 1, Nt)
    C = np.vstack([
        800 * np.exp(-0.5 * ((t - 0.40) / 0.15) ** 2) + 60,
        300 * np.exp(-0.5 * ((t - 0.55) / 0.25) ** 2) + 20,
        150 * (1 - np.exp(-5 * t)) * np.exp(-1.5 * t) + 10,
    ])
    return R, C, C.T @ R + rng.normal(0, noise, (Nt, Nl))


def test_regularisation_reduces_uncertainty_and_roughness():
    """
    Section 3.2: on identical inputs the temporal constraint should reduce both the
    reported uncertainty and the empirical roughness of the retrieved series, without
    changing the recovered dynamics. gamma is taken from the L-curve corner rather than
    chosen by hand.
    """
    R, C, obs = smooth_case()
    gamma = l_curve(R, obs, np.logspace(-9, -2, 30))["gamma_optimal"]

    cls = classical_least_squares(R, obs)
    reg = temporally_regularised_inversion(R, obs, gamma, penalty="paper")

    # Both scaled by the same sigma_eps (default noise_estimate='cls'), so the
    # reduction reflects the temporal constraint alone.
    np.testing.assert_allclose(reg.sigma_eps, cls.sigma_eps, rtol=1e-12)
    assert np.median(reg.uncertainty) < np.median(cls.uncertainty)

    roughness = lambda c: np.mean(np.abs(np.diff(c, axis=1)))
    assert roughness(reg.concentrations) < roughness(cls.concentrations)

    # The dynamics must survive: still faithful to the truth.
    assert np.corrcoef(reg.concentrations.ravel(), C.ravel())[0, 1] > 0.999


def test_smoothness_prior_smears_a_step():
    """
    The counterpart limitation, and a real one for flaming fuels. Where concentrations
    genuinely change on the sampling interval, the curvature penalty cannot represent
    the transition and rings around it. Section 4.2 notes the assumption holds better
    for smouldering than flaming emissions; this is why gamma must be re-optimised per
    fuel type rather than carried across.
    """
    R, C, obs = synthetic_case(Nt=60, noise=3e-5, seed=5)   # C[2] is a step
    reg = temporally_regularised_inversion(R, obs, 1e-3, penalty="paper")
    cls = classical_least_squares(R, obs)

    step = 2
    assert (np.abs(reg.concentrations[step] - C[step]).max()
            > np.abs(cls.concentrations[step] - C[step]).max())
    assert np.min(reg.concentrations[step]) < -1.0        # negative ringing lobe


def test_regularised_residual_rms_is_not_the_noise_level():
    """
    The RMS of the *regularised* residual is dominated by smoothing bias, not noise, so
    it must not be the default scale for the posterior. With 3e-5 absorbance noise it
    returns a value orders of magnitude too large at a realistic gamma - which, being a
    multiplier on the reported uncertainty, would invert the reduction of Section 3.2.
    """
    R, _, obs = synthetic_case(Nt=60, noise=3e-5, seed=5)
    default = temporally_regularised_inversion(R, obs, 1e-3, noise_estimate="cls")
    literal = temporally_regularised_inversion(R, obs, 1e-3, noise_estimate="rms")
    assert default.sigma_eps == pytest.approx(3e-5, rel=0.3)
    assert literal.sigma_eps > 100 * default.sigma_eps
    assert np.median(literal.uncertainty) > np.median(default.uncertainty)


# ---------------------------------------------------------------------------
# 4. Fast solver == explicit normal equations
# ---------------------------------------------------------------------------

def test_fast_solver_matches_explicit_normal_equations():
    """
    The decoupled solver must agree with forming A, A^T A + gamma D^T D and inverting
    it densely - the route v1.0 took, and the one that runs out of memory at scale.
    """
    R, _, obs = synthetic_case(Ns=3, Nl=50, Nt=14, noise=1e-5, seed=2)
    Ns, Nl = R.shape
    Nt = obs.shape[0]
    gamma = 5e-4

    A = build_A_matrix(R, Ns, Nl, Nt)
    P = sp.kron(sp.eye(Ns), penalty_matrix(Nt, "paper"))
    C = (A.T @ A + gamma * P).toarray()
    x_ref = np.linalg.solve(C, A.T @ obs.flatten())
    var_ref = np.diag(np.linalg.inv(C))

    r = temporally_regularised_inversion(R, obs, gamma, penalty="paper")
    np.testing.assert_allclose(r.concentrations.ravel(), x_ref, rtol=1e-6, atol=1e-8)

    unscaled = (r.uncertainty / r.sigma_eps) ** 2
    np.testing.assert_allclose(unscaled.ravel(), var_ref, rtol=1e-6, atol=1e-12)


def test_l_curve_finds_a_corner():
    R, _, obs = synthetic_case(Nt=40, noise=5e-5, seed=9)
    out = l_curve(R, obs, np.logspace(-8, 0, 25))
    assert 0 < out["corner_index"] < 24
    assert np.all(np.diff(out["solution_norm"]) <= 1e-9)  # smoother with larger gamma


# ---------------------------------------------------------------------------
# 5. Conventions
# ---------------------------------------------------------------------------

def test_radis_absorbance_is_converted_to_decadic():
    """
    RADIS's 'absorbance' is napierian optical depth (transmittance = exp(-absorbance)).
    Fitting it against -log10(I/I0) without dividing by ln(10) scales every retrieved
    concentration by 1/2.302585.
    """
    tau = np.array([0.0, 0.5, 1.0, 2.5])
    np.testing.assert_allclose(from_radis_absorbance(tau, "decadic"), tau / LN10)
    np.testing.assert_allclose(from_radis_absorbance(tau, "napierian"), tau)
    transmittance = np.exp(-tau)
    np.testing.assert_allclose(from_radis_absorbance(tau, "decadic"),
                               -np.log10(transmittance))


def test_ln10_error_cancels_from_ratios():
    """
    The v1.0 convention error is a common factor on all species, so it divides out of
    emission ratios, total carbon, emission factors and MCE. Published EF/MCE values
    are unaffected; absolute mixing ratios are not.
    """
    R, C, obs = synthetic_case()
    correct = temporally_regularised_inversion(R, obs, 1e-12).concentrations
    v1 = temporally_regularised_inversion(R * LN10, obs, 1e-12).concentrations
    np.testing.assert_allclose(v1, correct / LN10, rtol=1e-6, atol=1e-3)
    np.testing.assert_allclose(v1[1] / v1[0], correct[1] / correct[0], rtol=1e-6)


def test_number_density_matches_loschmidt():
    """2.6868e19 cm^-3 at 273.15 K, 1 atm."""
    n = number_density(1.01325, 273.15)
    np.testing.assert_allclose(n, 2.6867811e19, rtol=1e-5)


# ---------------------------------------------------------------------------
# 6. Baseline
# ---------------------------------------------------------------------------

def test_sparse_baseline_tracks_lower_envelope():
    w = np.linspace(0, 1, 400)
    drift = 1.0 + 0.3 * w
    y = drift - 0.4 * np.exp(-0.5 * ((w - 0.5) / 0.02) ** 2)
    base = get_baseline(y, lam=1e5, p=0.02, niter=12)
    assert np.mean(np.abs(base - drift)) < 0.02
    assert base[200] > y[200]          # baseline sits above the absorption dip


def test_baseline_memory_is_linear_in_length():
    """v1.0 allocated dense (L, L) arrays; 30k channels needed ~21 GB."""
    for L in (500, 4000):
        y = np.ones(L) + 0.01 * np.sin(np.linspace(0, 20, L))
        assert get_baseline(y, lam=1e6, niter=3).shape == (L,)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
