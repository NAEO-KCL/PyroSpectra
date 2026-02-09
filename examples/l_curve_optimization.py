"""
L-Curve Method for Regularization Parameter Selection

This script demonstrates how to use the L-curve criterion to select
the optimal regularization parameter λ for Tikhonov regularization.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

from ftir_fire_emissions import (
    read_data,
    get_compounds,
    generate_reference,
    process_spectra,
    lasso_inversion,
    temporally_regularised_inversion,
    build_A_matrix,
    create_smoother
)


def compute_l_curve(reference_spectra, observed_spectra, lambda_range):
    """
    Compute L-curve for regularization parameter selection.
    
    Parameters
    ----------
    reference_spectra : np.ndarray
        Reference spectra, shape (Ns, Nl)
    observed_spectra : np.ndarray
        Observed spectra, shape (Nt, Nl)
    lambda_range : np.ndarray
        Array of λ values to test
    
    Returns
    -------
    residual_norms : np.ndarray
        ||Ax - y|| for each λ
    solution_norms : np.ndarray
        ||x|| for each λ
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl
    
    Ns, Nl = reference_spectra.shape
    Nt = observed_spectra.shape[0]
    
    y = observed_spectra.flatten()
    A_mat = build_A_matrix(reference_spectra, Ns, Nl, Nt)
    D_mat = sp.kron(sp.eye(Ns), create_smoother(Nt))
    
    residual_norms = []
    solution_norms = []
    
    for lambda_ in lambda_range:
        # Solve regularized system
        C = sp.csc_matrix(A_mat.T @ A_mat + lambda_ * D_mat)
        x_sol = spl.spsolve(C, A_mat.T @ y)
        
        # Compute norms
        residual = A_mat @ x_sol - y
        residual_norm = np.linalg.norm(residual)
        solution_norm = np.linalg.norm(x_sol)
        
        residual_norms.append(residual_norm)
        solution_norms.append(solution_norm)
    
    return np.array(residual_norms), np.array(solution_norms)


def find_corner(residual_norms, solution_norms):
    """
    Find the corner of the L-curve (optimal regularization).
    
    The corner represents the best trade-off between fitting the data
    and solution smoothness.
    
    Parameters
    ----------
    residual_norms : np.ndarray
        Residual norms for different λ
    solution_norms : np.ndarray
        Solution norms for different λ
    
    Returns
    -------
    int
        Index of optimal λ value
    """
    # Normalize coordinates
    x = np.log10(residual_norms)
    y = np.log10(solution_norms)
    
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())
    
    # Find point with maximum distance to line connecting endpoints
    p1 = np.array([x_norm[0], y_norm[0]])
    p2 = np.array([x_norm[-1], y_norm[-1]])
    
    distances = []
    for i in range(len(x_norm)):
        point = np.array([x_norm[i], y_norm[i]])
        # Distance from point to line
        dist = np.abs(np.cross(p2 - p1, p1 - point)) / np.linalg.norm(p2 - p1)
        distances.append(dist)
    
    corner_idx = np.argmax(distances)
    return corner_idx


def main():
    """
    Demonstrate L-curve method for λ selection.
    """
    from pathlib import Path
    
    # Setup paths (adjust as needed)
    data_dir = Path('./data/experiment_2024_01_15')
    compound_file = Path('./compounds.pkl')
    result_dir = Path('./results/l_curve_analysis')
    result_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("L-CURVE ANALYSIS FOR REGULARIZATION PARAMETER SELECTION")
    print("=" * 70)
    
    # Load and prepare data (abbreviated - see example_workflow.py for full version)
    print("\nLoading data...")
    spectra, wavenumbers, pressure, temperature, datetime = read_data(data_dir)
    emission_species = get_compounds(compound_file)
    
    print("Generating reference spectra...")
    reference_spectra, full_reference, mask = generate_reference(
        result_dir, emission_species, wavenumbers, pressure, temperature, sigma=0.5
    )
    
    print("Processing observed spectra...")
    observed_spectra, _ = process_spectra(spectra, mask, result_dir)
    
    print("Identifying species...")
    ref_filtered, _, obs_filtered, species_filtered, _ = lasso_inversion(
        reference_spectra, full_reference, observed_spectra, emission_species
    )
    
    # Define range of λ values to test (logarithmic spacing)
    lambda_range = np.logspace(-6, -1, 50)
    
    print(f"\nComputing L-curve for {len(lambda_range)} λ values...")
    print(f"Range: {lambda_range.min():.2e} to {lambda_range.max():.2e}")
    
    residual_norms, solution_norms = compute_l_curve(
        ref_filtered, obs_filtered, lambda_range
    )
    
    # Find optimal λ
    corner_idx = find_corner(residual_norms, solution_norms)
    lambda_optimal = lambda_range[corner_idx]
    
    print(f"\n{'=' * 70}")
    print(f"OPTIMAL REGULARIZATION PARAMETER")
    print(f"{'=' * 70}")
    print(f"λ_optimal = {lambda_optimal:.2e}")
    print(f"Located at index {corner_idx} of {len(lambda_range)} tested values")
    
    # Plot L-curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # L-curve in log-log space
    ax1.loglog(residual_norms, solution_norms, 'b-', linewidth=2, label='L-curve')
    ax1.loglog(
        residual_norms[corner_idx], 
        solution_norms[corner_idx], 
        'ro', markersize=10, label=f'Corner (λ={lambda_optimal:.2e})'
    )
    ax1.set_xlabel('Residual Norm ||Ax - y||', fontsize=12)
    ax1.set_ylabel('Solution Norm ||x||', fontsize=12)
    ax1.set_title('L-Curve', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Parameter sweep
    ax2.semilogx(lambda_range, residual_norms, 'b-', linewidth=2, label='Residual Norm')
    ax2.axvline(lambda_optimal, color='r', linestyle='--', linewidth=2, label='Optimal λ')
    ax2.set_xlabel('Regularization Parameter λ', fontsize=12)
    ax2.set_ylabel('Residual Norm', fontsize=12)
    ax2.set_title('Residual vs. Regularization', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(result_dir / 'l_curve_analysis.pdf')
    plt.savefig(result_dir / 'l_curve_analysis.png', dpi=300)
    print(f"\nL-curve plot saved to {result_dir}")
    
    # Now run inversion with optimal λ
    print(f"\nRunning inversion with optimal λ = {lambda_optimal:.2e}...")
    concentrations, uncertainties = temporally_regularised_inversion(
        reference_spectra=ref_filtered,
        residual_spectra=obs_filtered,
        lambda_=lambda_optimal,
        result_dir=result_dir,
        compound_list=list(species_filtered.keys())
    )
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
