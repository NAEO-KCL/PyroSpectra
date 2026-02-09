"""
Example Usage Script

This script demonstrates a complete workflow for analyzing FTIR fire emission data.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ftir_fire_emissions import (
    read_data,
    get_compounds,
    generate_reference,
    process_spectra,
    lasso_inversion,
    temporally_regularised_inversion,
    inversion_residual,
    save_results
)


def main():
    """
    Complete analysis pipeline for FTIR fire emission data.
    """
    
    # =========================================================================
    # 1. SETUP
    # =========================================================================
    
    # Define paths
    data_dir = Path('./data/experiment_2024_01_15')  # Adjust to your data
    compound_file = Path('./compounds.pkl')          # Compound definitions
    result_dir = Path('./results/experiment_2024_01_15')
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # Set regularization parameter (optimize using L-curve method)
    lambda_reg = 1e-3
    
    # Set instrumental broadening (Gaussian width in cm^-1)
    sigma_instrumental = 0.5
    
    
    # =========================================================================
    # 2. LOAD DATA
    # =========================================================================
    
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    # Load spectral data and metadata
    spectra, wavenumbers, pressure, temperature, datetime = read_data(data_dir)
    
    print(f"\nData Summary:")
    print(f"  Number of spectra: {len(spectra)}")
    print(f"  Wavenumber range: {wavenumbers.min():.1f} - {wavenumbers.max():.1f} cm⁻¹")
    print(f"  Gas cell pressure: {pressure:.5f} bar")
    print(f"  Gas cell temperature: {temperature:.1f} K")
    
    # Load compound definitions
    emission_species = get_compounds(compound_file)
    print(f"  Number of candidate species: {len(emission_species)}")
    
    
    # =========================================================================
    # 3. GENERATE REFERENCE SPECTRA
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("GENERATING REFERENCE SPECTRA")
    print("=" * 70)
    
    reference_spectra, full_reference, mask = generate_reference(
        result_dir=result_dir,
        emission_species=emission_species,
        w=wavenumbers,
        P=pressure,
        T=temperature,
        sigma=sigma_instrumental
    )
    
    print(f"\nReference spectra shape: {reference_spectra.shape}")
    print(f"Active spectral windows: {mask.sum()} / {len(mask)} wavenumbers")
    
    
    # =========================================================================
    # 4. PROCESS OBSERVED SPECTRA
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("PROCESSING OBSERVED SPECTRA")
    print("=" * 70)
    
    observed_spectra, full_observed = process_spectra(
        spectra, mask, result_dir
    )
    
    print(f"\nProcessed spectra shape: {observed_spectra.shape}")
    
    
    # =========================================================================
    # 5. SPECIES IDENTIFICATION (LASSO)
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("IDENTIFYING PRESENT SPECIES")
    print("=" * 70)
    
    (ref_filtered, full_ref_filtered, obs_filtered, 
     species_filtered, lasso_scores) = lasso_inversion(
        reference_spectra,
        full_reference,
        observed_spectra,
        emission_species
    )
    
    print(f"\nIdentified species: {list(species_filtered.keys())}")
    print(f"Filtered reference shape: {ref_filtered.shape}")
    print(f"Filtered observations shape: {obs_filtered.shape}")
    
    
    # =========================================================================
    # 6. REGULARIZED INVERSION
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("PERFORMING REGULARIZED INVERSION")
    print("=" * 70)
    print(f"Regularization parameter λ = {lambda_reg}")
    
    concentrations, uncertainties = temporally_regularised_inversion(
        reference_spectra=ref_filtered,
        residual_spectra=obs_filtered,
        lambda_=lambda_reg,
        result_dir=result_dir,
        compound_list=list(species_filtered.keys())
    )
    
    print(f"\nConcentration shape: {concentrations.shape}")
    print(f"Uncertainty shape: {uncertainties.shape}")
    
    
    # =========================================================================
    # 7. COMPUTE RESIDUALS
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("COMPUTING MODEL RESIDUALS")
    print("=" * 70)
    
    y_model, y_obs, y_model_err, y_model_wv, y_model_time = inversion_residual(
        ref_filtered, obs_filtered, concentrations, uncertainties
    )
    
    # Calculate fit statistics
    residuals = y_obs - y_model
    rmse = np.sqrt(np.mean(residuals**2))
    r2 = 1 - np.sum(residuals**2) / np.sum((y_obs - np.mean(y_obs))**2)
    
    print(f"\nFit Statistics:")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  R²: {r2:.4f}")
    
    
    # =========================================================================
    # 8. SAVE RESULTS
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    save_results(
        concentrations,
        uncertainties,
        list(species_filtered.keys()),
        datetime[:len(obs_filtered)],  # Adjust for removed timesteps
        result_dir,
        prefix=''
    )
    
    
    # =========================================================================
    # 9. PLOT RESULTS
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    
    # Reshape concentrations for plotting
    Ns = len(species_filtered)
    Nt = len(obs_filtered)
    conc_reshaped = concentrations.reshape(Ns, Nt, order='F')
    uncert_reshaped = uncertainties.reshape(Ns, Nt, order='F')
    
    # Plot time series for each species
    fig, axes = plt.subplots(Ns, 1, figsize=(12, 2*Ns), sharex=True)
    if Ns == 1:
        axes = [axes]
    
    for i, (species, ax) in enumerate(zip(species_filtered.keys(), axes)):
        # Plot concentration with uncertainty band
        time_indices = np.arange(Nt)
        ax.plot(time_indices, conc_reshaped[i], label=species, color='C0')
        ax.fill_between(
            time_indices,
            conc_reshaped[i] - uncert_reshaped[i],
            conc_reshaped[i] + uncert_reshaped[i],
            alpha=0.3, color='C0'
        )
        
        ax.set_ylabel(f'{species}\nConcentration (ppm)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time Step')
    plt.tight_layout()
    plt.savefig(result_dir / 'concentration_timeseries.pdf')
    plt.savefig(result_dir / 'concentration_timeseries.png', dpi=300)
    plt.close()
    
    print(f"\nPlots saved to {result_dir}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
