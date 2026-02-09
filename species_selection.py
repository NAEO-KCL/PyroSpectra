"""
Species Selection Module

Automated gas species identification using Lasso (L1-regularized) regression
with cross-validation.
"""

import numpy as np
import scipy.sparse as sp
import random
from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from joblib import Parallel, delayed

from .preprocessing import build_A_matrix


def lasso_inversion(reference_spectra, full_reference_spectra, 
                   observed_spectra, emission_species):
    """
    Identify present gas species using Lasso regression.
    
    This function uses L1 regularization to automatically select which species
    are actually present in the measured spectra. Species with zero coefficients
    are excluded from further analysis, reducing overfitting and computational cost.
    
    Parameters
    ----------
    reference_spectra : np.ndarray
        Windowed reference spectra, shape (Ns, Nl)
    full_reference_spectra : np.ndarray
        Full reference spectra, shape (Ns, Nl_full)
    observed_spectra : np.ndarray
        Observed absorbance spectra, shape (Nt, Nl)
    emission_species : dict
        Dictionary of all candidate species
    
    Returns
    -------
    reference_spectra : np.ndarray
        Filtered reference spectra for detected species only
    full_reference_spectra : np.ndarray
        Filtered full reference spectra for detected species
    observed_spectra : np.ndarray
        Observed spectra with test timesteps removed
    new_emission_species : dict
        Filtered species dictionary containing only detected species
    lasso_score : dict
        Dictionary containing Lasso fitting diagnostics
        
    Notes
    -----
    The function randomly samples 10 timesteps for species identification,
    then removes these from the dataset to prevent overfitting in the
    final inversion.
    
    Core species (CO2, CO, CH4, N2O) are always retained even if Lasso
    assigns them zero coefficients, as they are known to be present in
    biomass burning smoke.
    """
    print('Performing Lasso Inversion')
    
    Ns, Nl = reference_spectra.shape
    Nt = observed_spectra.shape[0]
    
    # Build design matrix for single timestep
    A = build_A_matrix(reference_spectra, Ns, Nl, 1)
    A_dense = sp.csr_matrix(A).todense()
    
    # Randomly sample 10 timesteps for species identification
    rand_timesteps = random.sample(range(Nt), min(10, Nt))
    
    # Fit Lasso to sampled timesteps
    results, cross_val_scores, r2_scores, rmse_scores = fit_lasso(
        A_dense, observed_spectra, rand_timesteps
    )
    
    # Store fitting diagnostics
    lasso_score = {
        'Results': results,
        'Timesteps': rand_timesteps,
        'Cross Validation Score': cross_val_scores,
        'R2': r2_scores,
        'RMSE': rmse_scores
    }
    
    # Identify species with non-zero coefficients
    species_keys = list(emission_species.keys())
    
    # Core species that should always be included
    core_species = ['CO2', 'CO', 'CH4', 'N2O']
    
    # Keep species with non-zero coefficients OR core species
    present_indices = [
        i for i in range(Ns) 
        if np.any(results[i]) or species_keys[i] in core_species
    ]
    
    # Filter species and spectra
    filtered_keys = [species_keys[i] for i in present_indices]
    new_emission_species = {key: emission_species[key] for key in filtered_keys}
    
    reference_spectra = reference_spectra[present_indices]
    full_reference_spectra = full_reference_spectra[present_indices]
    
    # Remove sampled timesteps from observed spectra to prevent overfitting
    observed_spectra = np.delete(observed_spectra, rand_timesteps, axis=0)
    
    print(f"Species identified: {filtered_keys}")
    print("Complete")
    
    return (reference_spectra, full_reference_spectra, observed_spectra, 
            new_emission_species, lasso_score)


def fit_lasso(A_dense, residual_spectra, timesteps):
    """
    Fit Lasso regression with cross-validation to multiple timesteps.
    
    Parameters
    ----------
    A_dense : np.ndarray
        Dense design matrix, shape (Nl, Ns)
    residual_spectra : np.ndarray
        Observed absorbance spectra, shape (Nt, Nl)
    timesteps : list
        Indices of timesteps to fit
    
    Returns
    -------
    results : np.ndarray
        Lasso coefficients for each species at each timestep, shape (Ns, len(timesteps))
    cross_val_scores : list
        Cross-validation scores (mean, std, all folds) for each timestep
    r2_scores : list
        R² scores for each timestep
    rmse_scores : list
        RMSE scores for each timestep
        
    Notes
    -----
    Uses LassoCV with 5-fold cross-validation to automatically tune the
    regularization parameter. Fitting is parallelized across timesteps.
    """
    def process_timestep(t):
        """Fit Lasso to a single timestep."""
        lasso = LassoCV(cv=5, fit_intercept=False, n_jobs=-1)
        lasso.fit(np.asarray(A_dense), residual_spectra[t])
        
        coef = lasso.coef_
        
        # Cross-validation scores
        scores = cross_val_score(
            lasso, np.asarray(A_dense), residual_spectra[t], 
            cv=5, scoring='neg_mean_absolute_error'
        )
        cross_val_score_mean = (scores.mean(), scores.std(), scores)
        
        # Prediction metrics
        y_pred = lasso.predict(np.asarray(A_dense))
        r2 = abs(r2_score(residual_spectra[t], y_pred))
        rmse = np.sqrt(mean_squared_error(residual_spectra[t], y_pred))
        
        return coef, cross_val_score_mean, r2, rmse
    
    # Parallel fitting across timesteps
    results_list = Parallel(n_jobs=-1)(
        delayed(process_timestep)(t) for t in timesteps
    )
    
    # Unpack results
    results = np.zeros((A_dense.shape[1], len(timesteps)))
    cross_val_scores = []
    r2_scores = []
    rmse_scores = []
    
    for idx, (coef, cv_score, r2, rmse) in enumerate(results_list):
        results[:, idx] = coef
        cross_val_scores.append(cv_score)
        r2_scores.append(r2)
        rmse_scores.append(rmse)
    
    return results, cross_val_scores, r2_scores, rmse_scores


def filter_compounds(results, compounds, always_present=None):
    """
    Filter compounds based on Lasso results.
    
    Parameters
    ----------
    results : np.ndarray
        Lasso coefficients, shape (Ns, Nt)
    compounds : dict
        Dictionary of all candidate compounds
    always_present : list, optional
        Species to always include regardless of Lasso results.
        Default: ['CO2', 'CO', 'CH4']
    
    Returns
    -------
    dict
        Filtered compounds dictionary containing only present species
    """
    if always_present is None:
        always_present = ['CO2', 'CO', 'CH4']
    
    # Find species with any non-zero coefficients
    present_compounds = [
        key for key, values in zip(compounds.keys(), results) 
        if np.any(values)
    ]
    
    # Add core species if not already present
    for species in always_present:
        if species not in present_compounds and species in compounds:
            present_compounds.append(species)
    
    # Create filtered dictionary
    new_compounds = {key: compounds[key] for key in present_compounds}
    
    return new_compounds
