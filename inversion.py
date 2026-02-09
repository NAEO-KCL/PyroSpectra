"""
Inversion Module

Temporally regularized concentration retrieval using Tikhonov regularization
and uncertainty quantification.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import matplotlib.pyplot as plt

from .preprocessing import build_A_matrix, create_smoother


def temporally_regularised_inversion(reference_spectra, residual_spectra, 
                                    lambda_, result_dir, compound_list,
                                    post_cov=True, do_spilu=True):
    """
    Perform temporally regularized inversion using Tikhonov regularization.
    
    This function solves the regularized least squares problem:
        minimize ||A*x - y||² + λ||D*x||²
    
    where:
    - A encodes the reference spectra at each timestep
    - x are the concentrations we want to retrieve
    - y are the observed absorbance spectra
    - D is the temporal smoothness operator
    - λ controls the regularization strength
    
    Temporal regularization reduces noise-driven fluctuations while maintaining
    good temporal resolution, as demonstrated in Figure 8 of the paper.
    
    Parameters
    ----------
    reference_spectra : np.ndarray
        Reference absorbance spectra, shape (Ns, Nl)
    residual_spectra : np.ndarray
        Observed absorbance spectra, shape (Nt, Nl)
    lambda_ : float
        Regularization parameter. Controls smoothness vs. fit trade-off.
        Typical values: 1e-4 to 1e-2. Use L-curve method to optimize.
    result_dir : str or Path
        Directory to save correlation matrix plot
    compound_list : list
        Names of compounds (for correlation matrix labels)
    post_cov : bool, optional
        Whether to compute and return posterior covariance. Default: True
    do_spilu : bool, optional
        Use incomplete LU factorization for faster solving. Default: True
    
    Returns
    -------
    x_sol : np.ndarray
        Retrieved concentrations, shape (Ns*Nt,)
        Reshape to (Nt, Ns) for time series of each species
    sigma : np.ndarray
        1-σ uncertainties for each concentration, same shape as x_sol
        
    Notes
    -----
    The posterior covariance matrix C is computed as:
        C = (A^T A + λD^T D)^(-1)
    
    Diagonal elements give variances, off-diagonals show correlations
    between species (indicating spectral overlap).
    
    References
    ----------
    Eilers (2003). A perfect smoother. Analytical Chemistry, 75(14), 3631-3636.
    Calvetti et al. (2000). Tikhonov regularization and the L-curve.
    """
    print('Performing Tikhonov Regularisation')
    
    Ns, Nl = reference_spectra.shape
    Nt = residual_spectra.shape[0]
    
    # Check dimensions
    assert Nl == residual_spectra.shape[1], (
        f"Spectral dimension mismatch: reference has {Nl} wavenumbers "
        f"but observations have {residual_spectra.shape[1]}"
    )
    
    # Flatten observed spectra into single vector
    y = residual_spectra.flatten()
    
    # Build design matrix A: (Nl*Nt, Ns*Nt)
    A_mat = build_A_matrix(reference_spectra, Ns, Nl, Nt)
    
    # Build temporal smoothness operator: (Ns*Nt, Ns*Nt)
    # Kronecker product applies smoothing to each species independently
    D_mat = sp.kron(sp.eye(Ns), create_smoother(Nt))
    
    # Compute regularized normal equations matrix
    C = sp.csc_matrix(A_mat.T @ A_mat + lambda_ * D_mat)
    
    # Compute posterior covariance for uncertainty estimates
    c_inv = np.linalg.inv(C.toarray())
    sigma = np.sqrt(np.diag(c_inv))
    
    # Compute correlation matrix for single-timestep fit (diagnostic)
    S = [sp.lil_matrix(reference_spectra[i, :].reshape(-1, 1)) for i in range(Ns)]
    A_mat_single = sp.hstack(S)
    C_single = sp.csc_matrix(A_mat_single.T @ A_mat_single)
    c_inv_single = np.linalg.inv(C_single.toarray())
    
    # Convert covariance to correlation matrix
    std_devs = np.sqrt(np.diag(c_inv_single))
    inv_corr = c_inv_single / np.outer(std_devs, std_devs)
    
    # Plot correlation matrix
    _plot_correlation_matrix(inv_corr, compound_list, result_dir)
    
    # Solve regularized system
    if do_spilu:
        # Incomplete LU factorization (faster, slight approximation)
        x_sol = spl.spilu(C).solve(A_mat.T @ y)
    else:
        # Direct sparse solver (slower, exact)
        x_sol = spl.spsolve(C, A_mat.T @ y)
    
    print("Complete")
    
    return (x_sol, sigma)


def inversion_residual(ref_spec, obs_spec, x_sol, sigma):
    """
    Calculate model predictions and residuals from inversion results.
    
    This function reconstructs the modeled absorbance spectra from the
    retrieved concentrations and compares them to observations.
    
    Parameters
    ----------
    ref_spec : np.ndarray
        Reference spectra matrix, shape (Ns, Nl)
    obs_spec : np.ndarray
        Observed spectra matrix, shape (Nt, Nl)
    x_sol : np.ndarray
        Retrieved concentrations, shape (Ns*Nt,)
    sigma : np.ndarray
        Concentration uncertainties, shape (Ns*Nt,)
    
    Returns
    -------
    y_model : np.ndarray
        Modeled absorbance (flattened), shape (Nt*Nl,)
    y : np.ndarray
        Observed absorbance (flattened), shape (Nt*Nl,)
    y_model_err : np.ndarray
        Model uncertainties (flattened), shape (Nt*Nl,)
    y_model_wv_reshaped : np.ndarray
        Modeled absorbance by wavenumber, shape (Nt, Nl)
    y_model_time_reshaped : np.ndarray
        First wavenumber of each timestep, shape (Nt,)
    """
    print('Calculating Residuals')
    
    x_err = np.sqrt(sigma)
    
    Nl = ref_spec.shape[1]   # Number of wavenumbers
    Nt = obs_spec.shape[0]   # Number of time steps
    Ns = ref_spec.shape[0]   # Number of species
    
    # Flatten observations
    y = obs_spec.flatten()
    
    # Reshape for broadcasting
    # Reference: (Ns, 1, Nl)
    ref_spec_reshaped = ref_spec.reshape(Ns, 1, Nl)
    
    # Concentrations: (Ns, Nt, 1)
    # x_sol is stored as [species0_time0, species0_time1, ..., species1_time0, ...]
    x_sol_reshaped = x_sol.reshape(Ns, Nt, order='F').reshape(Ns, Nt, 1)
    x_err_reshaped = x_err.reshape(Ns, Nt, order='F').reshape(Ns, Nt, 1)
    
    # Calculate modeled absorbance: sum over species
    # y_model[t,l] = sum_i ( c_i[t] * R_i[l] )
    y_model = np.sum(ref_spec_reshaped * x_sol_reshaped, axis=0).flatten()
    
    # Propagate uncertainties
    y_model_err = np.sqrt(
        np.sum((ref_spec_reshaped * x_err_reshaped)**2, axis=0)
    ).flatten()
    
    # Reshape for analysis
    y_model_wv_reshaped, y_model_time_reshaped = reshape_residuals(
        y_model, y, Nl
    )
    
    return y_model, y, y_model_err, y_model_wv_reshaped, y_model_time_reshaped


def reshape_residuals(y_model, y, Nl):
    """
    Reshape residual data for visualization.
    
    Parameters
    ----------
    y_model : np.ndarray
        Modeled data (flattened)
    y : np.ndarray
        Observed data (flattened)
    Nl : int
        Number of wavenumbers
    
    Returns
    -------
    y_model_wv_squeezed : np.ndarray
        Modeled data reshaped by wavenumber, shape (Nt, Nl)
    y_model_time_squeezed : np.ndarray
        First element of each timestep, shape (Nt,)
    """
    # Reshape to (Nt, Nl)
    y_model_wv_squeezed = y_model.reshape(-1, Nl)
    
    # Extract first wavenumber of each timestep
    y_model_time_squeezed = y_model_wv_squeezed[:, 0]
    
    return y_model_wv_squeezed, y_model_time_squeezed


def _plot_correlation_matrix(corr_matrix, compound_list, result_dir):
    """
    Plot correlation matrix between species.
    
    Parameters
    ----------
    corr_matrix : np.ndarray
        Correlation matrix, shape (Ns, Ns)
    compound_list : list
        Species names for labels
    result_dir : str
        Directory to save plot
    """
    import os
    os.makedirs(f'{result_dir}/reference_information', exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(corr_matrix, cmap='seismic', vmin=-1, vmax=1, aspect='auto')
    
    # Set tick positions and labels
    ticks = np.arange(len(compound_list))
    plt.xticks(ticks, compound_list, rotation=45, ha='right')
    plt.yticks(ticks, compound_list)
    
    # Add colorbar
    cbar = plt.colorbar(pad=0.01)
    cbar.set_label('Correlation Coefficient', rotation=-90, labelpad=15)
    
    plt.title('Species Correlation Matrix')
    plt.tight_layout()
    
    plt.savefig(f'{result_dir}/reference_information/Correlation_Matrix.png', dpi=300)
    plt.savefig(f'{result_dir}/reference_information/Correlation_Matrix.pdf')
    plt.close()
    
    print(f"Correlation matrix saved to {result_dir}/reference_information/")
