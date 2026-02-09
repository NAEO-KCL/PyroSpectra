"""
Reference Spectra Generation Module

Functions for generating reference absorbance spectra using RADIS
and the HITRAN molecular spectroscopic database.
"""

import numpy as np
import matplotlib.pyplot as plt
from radis import calc_spectrum
from radis.tools import convolve_with_slit
from radis import Spectrum


def gaussian(x, mu, sigma):
    """
    Gaussian function for instrumental broadening.
    
    Parameters
    ----------
    x : np.ndarray
        Input values
    mu : float
        Mean (center) of Gaussian
    sigma : float
        Standard deviation (width) of Gaussian
    
    Returns
    -------
    np.ndarray
        Gaussian values
    """
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2 * sigma**2))


def generate_reference(result_dir, emission_species, w, P, T, sigma):
    """
    Generate reference spectra for all target species.
    
    This is the main entry point for reference spectra generation. It creates
    both windowed (for fitting) and full (for visualization) reference spectra.
    
    Parameters
    ----------
    result_dir : str or Path
        Directory to save reference spectra plots
    emission_species : dict
        Dictionary of species to simulate. Keys are molecule names (e.g., 'CO2'),
        values are dicts containing 'bounds' - list of [min, max] wavenumber ranges
    w : np.ndarray
        Observed wavenumber array (cm^-1)
    P : float
        Pressure in bar
    T : float
        Temperature in Kelvin
    sigma : float
        Instrumental broadening parameter (Gaussian width in cm^-1)
    
    Returns
    -------
    reference_spectra : np.ndarray
        Windowed reference spectra with zero columns removed, shape (Ns, Nl_masked)
    full_storage_mtx : np.ndarray
        Full reference spectra across entire range, shape (Ns, Nl)
    mask : np.ndarray
        Boolean mask indicating non-zero columns in reference spectra
        
    Notes
    -----
    Reference spectra are simulated at 1 ppm concentration over 5m path length
    and then scaled linearly for concentration retrieval (Beer's Law).
    """
    import os
    os.makedirs(f'{result_dir}/reference_information', exist_ok=True)
    
    # Generate windowed reference matrix (selected spectral regions)
    storage_mtx = get_reference_matrix(
        emission_species, T, P, w, sigma, result_dir
    )
    
    # Generate full reference matrix (entire spectral range)
    full_storage_mtx = get_reference_matrix_full(
        emission_species, T, P, w, sigma, result_dir
    )
    
    print("Reference Matrix Generated!")
    
    # Replace NaN values with zeros
    storage_mtx[np.isnan(storage_mtx)] = 0
    full_storage_mtx[np.isnan(full_storage_mtx)] = 0
    
    # Create mask for non-zero columns (active spectral windows)
    mask = ~np.all(storage_mtx == 0, axis=0)
    
    # Remove zero columns
    reference_spectra = storage_mtx[:, mask]
    
    return reference_spectra, full_storage_mtx, mask


def get_reference_matrix(emission_species, T, P, W_obs, sigma, result_dir):
    """
    Generate windowed reference matrix using selected spectral regions.
    
    This function simulates high-resolution spectra for each species using RADIS,
    applies instrumental broadening, and resamples to the observed wavenumber grid.
    Only user-defined spectral windows are populated.
    
    Parameters
    ----------
    emission_species : dict
        Dictionary with molecule names as keys. Each value is a dict containing:
        - 'bounds': list of [wmin, wmax] pairs defining spectral windows
    T : float
        Gas temperature in Kelvin
    P : float
        Gas pressure in bar
    W_obs : np.ndarray
        Observed wavenumber grid (cm^-1)
    sigma : float
        Gaussian broadening width (cm^-1) to simulate instrumental lineshape
    result_dir : str
        Directory to save individual species plots
    
    Returns
    -------
    np.ndarray
        Reference matrix of shape (Ns, Nl) where Ns is number of species
        and Nl is number of observed wavenumbers
        
    Notes
    -----
    Spectra are calculated using HITRAN database via RADIS at:
    - 1 ppm mole fraction
    - 500 cm (5 m) path length
    - Local thermodynamic equilibrium
    """
    output = []
    
    for molecule in emission_species:
        # Initialize empty spectrum on observed grid
        tmp = np.zeros_like(W_obs)
        
        # Process each spectral window for this molecule
        for bound in emission_species[molecule]['bounds']:
            try:
                # Calculate high-resolution spectrum using RADIS
                s = calc_spectrum(
                    bound[0], bound[1],  # Wavenumber range (cm^-1)
                    molecule=molecule,
                    isotope='1',
                    pressure=P,          # bar
                    Tgas=T,              # K
                    mole_fraction=1e-6,  # 1 ppm
                    path_length=500,     # cm (5 m)
                    wstep='auto',
                    databank='hitran',
                    warnings={'AccuracyError': 'ignore'},
                )
            except Exception as error:
                print(f"Error calculating spectrum for {molecule} "
                      f"in range {bound}: {error}")
                continue
            
            # Get absorbance spectrum
            w, A = s.get('absorbance', wunit='cm-1')
            
            # Apply instrumental broadening (Gaussian convolution)
            x_values = np.arange(-1, 1, 0.001)
            gaussian_kernel = gaussian(x_values, 0, sigma)
            gaussian_kernel /= np.max(gaussian_kernel)  # Normalize
            
            w, A = convolve_with_slit(w, A, x_values, gaussian_kernel, wunit='cm-1')
            
            # Create spectrum object
            s = Spectrum.from_array(w, A, 'absorbance', wunit='cm-1', unit='')
            
            # Find overlap with observed wavenumber range
            iloc = np.argmin(np.abs(w.min() - W_obs))
            jloc = np.argmin(np.abs(w.max() - W_obs))
            
            # Resample to observed grid
            s.resample(W_obs[iloc:jloc], energy_threshold=2)
            w, A = s.get('absorbance', wunit='cm-1')
            
            # Insert into output array
            tmp[iloc:jloc] = A
        
        # Save individual species plot
        plt.figure(figsize=(12, 4))
        plt.plot(W_obs, tmp)
        plt.xlabel('Wavenumber (cm$^{-1}$)')
        plt.ylabel('Absorbance')
        plt.title(f'{molecule} Reference Spectrum')
        plt.tight_layout()
        plt.savefig(f'{result_dir}/reference_information/{molecule}.pdf')
        plt.close()
        
        output.append(tmp)
    
    return np.array(output)


def get_reference_matrix_full(emission_species, T, P, W_obs, sigma, result_dir):
    """
    Generate full reference matrix across entire spectral range.
    
    Similar to get_reference_matrix, but calculates spectra over the full
    800-8000 cm^-1 range without spectral windowing. Used for visualization
    and diagnostic purposes.
    
    Parameters
    ----------
    emission_species : dict
        Dictionary with molecule names as keys
    T : float
        Gas temperature in Kelvin
    P : float
        Gas pressure in bar
    W_obs : np.ndarray
        Observed wavenumber grid (cm^-1)
    sigma : float
        Gaussian broadening width (cm^-1)
    result_dir : str
        Directory to save plots
    
    Returns
    -------
    np.ndarray
        Full reference matrix, shape (Ns, Nl)
    """
    output = []
    
    for molecule in emission_species:
        # Initialize empty spectrum
        tmp = np.zeros_like(W_obs)
        
        try:
            # Calculate full spectral range
            s = calc_spectrum(
                800, 8000,           # Full MATRIX-MG5 range
                molecule=molecule,
                isotope='1',
                pressure=P,
                Tgas=T,
                mole_fraction=1e-6,
                path_length=500,     # cm
                warnings={'AccuracyError': 'ignore'},
            )
        except Exception as error:
            print(f"Error calculating full spectrum for {molecule}: {error}")
            continue
        
        # Get absorbance
        w, A = s.get('absorbance', wunit='cm-1')
        
        # Apply instrumental broadening
        x_values = np.arange(-1, 1, 0.001)
        gaussian_kernel = gaussian(x_values, 0, sigma)
        gaussian_kernel /= np.max(gaussian_kernel)
        
        w, A = convolve_with_slit(w, A, x_values, gaussian_kernel, wunit='cm-1')
        s = Spectrum.from_array(w, A, 'absorbance', wunit='cm-1', unit='')
        
        # Resample to observed grid
        iloc = np.argmin(np.abs(w.min() - W_obs))
        jloc = np.argmin(np.abs(w.max() - W_obs))
        s.resample(W_obs[iloc:jloc], energy_threshold=2)
        
        w, A = s.get('absorbance', wunit='cm-1')
        tmp[iloc:jloc] = A
        
        # Save full spectrum plot
        plt.figure(figsize=(12, 4))
        plt.plot(W_obs, tmp)
        plt.xlabel('Wavenumber (cm$^{-1}$)')
        plt.ylabel('Absorbance')
        plt.title(f'{molecule} Full Reference Spectrum')
        plt.tight_layout()
        plt.savefig(f'{result_dir}/reference_information/{molecule}_full.pdf')
        plt.close()
        
        output.append(tmp)
    
    return np.array(output)
