"""
Preprocessing Module

Functions for spectral preprocessing including baseline correction,
absorbance calculation, and matrix construction.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt


def get_baseline(y, lam=5e7, p=0.02, niter=10):
    """
    Asymmetric Least Squares Smoothing (O-ALS) for baseline correction.
    
    This is a key preprocessing step that removes instrumental baseline drift
    from FTIR spectra. The asymmetric weighting ensures baselines follow the
    lower envelope of the spectrum.
    
    Parameters
    ----------
    y : np.ndarray
        The spectrum data (intensity vs wavenumber)
    lam : float, optional
        Smoothness parameter. Higher values produce smoother baselines.
        Default: 5e7
    p : float, optional
        Asymmetry parameter (0 < p < 1). Lower values make the baseline
        follow the lower envelope more closely. Default: 0.02
    niter : int, optional
        Number of iterations for convergence. Default: 10
    
    Returns
    -------
    np.ndarray
        The estimated baseline with same shape as input
        
    References
    ----------
    Dong & Xu (2024). Baseline estimation using optimized asymmetric 
    least squares (O-ALS). Measurement, 233, 114731.
    """
    L = len(y)
    
    # Construct second-order difference matrix for smoothness constraint
    D = diags([1, -2, 1], [0, -1, -2], shape=(L, L)).toarray()
    D = lam * np.dot(D.T, D)
    
    # Initialize weights uniformly
    w = np.ones(L)
    
    # Iteratively reweight to fit baseline
    for i in range(niter):
        W = np.diag(w)
        Z = sp.csc_matrix(W + D)
        baseline = spsolve(Z, w * y)
        
        # Asymmetric weighting: penalize points above baseline more heavily
        w = p * (y < baseline) + (1 - p) * (y > baseline)
    
    return baseline


def process_spectra(spectra, mask, result_dir):
    """
    Process raw FTIR spectra to absorbance units with baseline correction.
    
    This function:
    1. Applies O-ALS baseline correction to the first spectrum
    2. Converts all spectra from transmittance to absorbance
    3. Applies spectral windowing mask
    4. Saves processed data for analysis
    
    Parameters
    ----------
    spectra : np.ndarray
        Raw spectra array, shape (Nt, Nl) where Nt is number of time steps
        and Nl is number of wavenumbers
    mask : np.ndarray
        Boolean mask indicating which wavenumbers to keep (selected spectral windows)
    result_dir : str or Path
        Directory to save results and plots
    
    Returns
    -------
    observed_spectra : np.ndarray
        Processed absorbance spectra with mask applied, shape (Nt, Nl_masked)
    full_observed_spectra : np.ndarray
        Full absorbance spectra without masking, shape (Nt, Nl)
        
    Notes
    -----
    Absorbance A is calculated as: A = -log10(I / I0)
    where I is the measured intensity and I0 is the baseline.
    """
    print('Processing Spectra')
    
    # Get baseline from first (pre-ignition) spectrum
    baseline = get_baseline(spectra[0])
    
    # Convert to absorbance: A = -log10(I/I0)
    full_observed_spectra = np.nan_to_num(-np.log10(spectra / baseline))
    
    # Save full spectra
    np.save(f'{result_dir}/results/full_obs.npy', full_observed_spectra)
    
    # Apply spectral window mask
    spectra_masked = np.array([s[mask] for s in spectra])
    baseline_masked = np.array(baseline[mask])
    
    # Prevent division by zero
    spectra_masked = np.where(spectra_masked == 0, 1e-10, spectra_masked)
    
    # Convert masked spectra to absorbance
    observed_spectra = np.nan_to_num(-np.log10(spectra_masked / baseline_masked))
    
    # Plot processed spectra for quality control
    _plot_spectra(observed_spectra, result_dir)
    
    return observed_spectra, full_observed_spectra


def build_A_matrix(spectra, Ns, Nl, Nt):
    """
    Build the design matrix A for the linear inversion problem.
    
    For the linear model: y = A*x + ε, where y is the observed absorbance,
    x is the concentrations we want to retrieve, and A encodes how each
    species' reference spectrum contributes to the observation.
    
    Parameters
    ----------
    spectra : np.ndarray
        Reference spectra for each species, shape (Ns, Nl)
    Ns : int
        Number of species
    Nl : int
        Number of wavenumbers (spectral channels)
    Nt : int
        Number of time steps
    
    Returns
    -------
    scipy.sparse matrix
        Design matrix A of shape (Nl*Nt, Ns*Nt)
        
    Notes
    -----
    The matrix is constructed as a block-diagonal structure where each block
    contains the reference spectrum for one species at one time step. This
    allows time-varying concentrations to be retrieved.
    """
    S = []
    
    for i in range(Ns):
        # Create sparse matrix for species i
        a = sp.lil_matrix((Nl * Nt, Nt), dtype=np.float32)
        
        # Fill in reference spectrum at each time step
        for j in range(Nt):
            a[(j * Nl):(j + 1) * Nl, j] = spectra[i, :].reshape(-1, 1)
        
        S.append(a)
    
    # Horizontally stack all species matrices
    return sp.hstack(S)


def create_smoother(N):
    """
    Create a first-order difference matrix for temporal regularization.
    
    This matrix enforces smoothness in the time series by penalizing
    rapid changes between consecutive time steps.
    
    Parameters
    ----------
    N : int
        Length of the time series
    
    Returns
    -------
    np.ndarray
        First-order difference matrix D of shape (N, N)
        
    Notes
    -----
    The matrix implements: (Dx)_t = x_{t+1} - x_t
    Boundary conditions use reflection (first and last points treated specially).
    """
    # Standard first-order difference: D*x computes differences
    D = 2 * np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1)
    
    # Boundary conditions: reflect at edges
    D[0, 0] = 1      # First point
    D[N - 1, N - 1] = 1  # Last point
    
    return D


def _plot_spectra(spectra, result_dir):
    """
    Plot individual processed spectra for quality control.
    
    Parameters
    ----------
    spectra : np.ndarray
        Processed absorbance spectra, shape (Nt, Nl)
    result_dir : str
        Directory to save plots
    """
    import os
    os.makedirs(f'{result_dir}/processed_data', exist_ok=True)
    
    for i, spectrum in enumerate(spectra):
        plt.figure(figsize=(10, 4))
        plt.plot(spectrum)
        plt.xlabel('Wavenumber Index')
        plt.ylabel('Absorbance')
        plt.title(f'Processed Spectrum {i}')
        plt.tight_layout()
        plt.savefig(f'{result_dir}/processed_data/{i}.pdf')
        plt.close()
