"""
I/O Utilities Module

Functions for reading spectral data, metadata, and compound definitions.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm


def read_spectrum(fname):
    """
    Read a single spectrum file.
    
    Parameters
    ----------
    fname : str or Path
        Path to spectrum file (.prn format)
    
    Returns
    -------
    np.ndarray
        Spectrum intensity values
    """
    return np.loadtxt(fname, usecols=[1])


def read_spectra(spectral_data, cutoff=800):
    """
    Read all spectral data files from a directory.
    
    The MATRIX-MG5 FTIR outputs .prn files containing wavenumber and
    intensity columns. This function reads all such files and constructs
    a time series of spectra.
    
    Parameters
    ----------
    spectral_data : str or Path
        Directory containing .prn spectral files
    cutoff : int, optional
        Wavenumber cutoff (cm^-1). Removes low-wavenumber noise.
        Default: 800 cm^-1
    
    Returns
    -------
    spectra : np.ndarray
        Array of spectra, shape (Nt, Nl) where Nt is number of timesteps
        and Nl is number of wavenumbers above cutoff
    wvc : np.ndarray
        Wavenumber array (cm^-1) after cutoff applied, shape (Nl,)
        
    Notes
    -----
    Files are sorted by name to ensure correct temporal ordering.
    All files must have the same wavenumber grid.
    """
    if isinstance(spectral_data, str):
        spectral_data = Path(spectral_data)
    
    # Get all .prn files, sorted by name
    files = sorted([f for f in spectral_data.glob("*.prn")])
    
    if not files:
        raise FileNotFoundError(f"No .prn files found in {spectral_data}")
    
    # Read wavenumbers from first file (all files have same grid)
    wv = np.loadtxt(files[0], usecols=[0])
    
    # Read all spectra
    spectra = np.array([read_spectrum(f) for f in tqdm(files, desc="Reading spectra")])
    
    # Apply cutoff to remove low-wavenumber noise
    wvc = wv[wv > cutoff]
    spectra = spectra[:, wv > cutoff]
    
    print(f"Loaded {len(spectra)} spectra with {len(wvc)} wavenumbers")
    
    return spectra, wvc


def get_pt(directory):
    """
    Read pressure and temperature log from MATRIX-MG5.
    
    The MATRIX-MG5 logs gas cell pressure and temperature to a CSV file.
    This function reads the log and returns median values.
    
    Parameters
    ----------
    directory : str or Path
        Directory containing PT_Log.txt file
    
    Returns
    -------
    P : float
        Median pressure in bar
    T : float
        Median temperature in Kelvin
    datetime : np.ndarray or None
        Datetime array if log exists, None otherwise
        
    Notes
    -----
    If no PT log is found, defaults to standard conditions (300 K, 1.01325 bar).
    """
    for filename in os.listdir(directory):
        if filename.endswith('PT_Log.txt'):
            # Read PT log
            df = pd.read_csv(
                os.path.join(directory, filename),
                delimiter=',',
                names=["Date", "Time", "T", "P"]
            )
            
            # Extract median values
            T = np.median(df["T"])      # Kelvin
            P = np.median(df["P"]) / 1000  # mbar to bar
            
            # Parse datetime
            datetime = pd.to_datetime(df["Date"] + " " + df["Time"]).to_numpy()
            
            print(f"Gas cell conditions: T = {T:.1f} K, P = {P:.5f} bar")
            return P, T, datetime
    
    # Default to standard conditions if no log found
    print("Warning: No PT log found. Using default conditions (300 K, 1.01325 bar)")
    T = 300.0
    P = 1.01325
    datetime = None
    
    return P, T, datetime


def read_data(directory):
    """
    Read all data from an experiment directory.
    
    This is the main data loading function. It reads spectral files,
    pressure/temperature logs, and constructs a complete dataset.
    
    Parameters
    ----------
    directory : str or Path
        Experiment directory containing:
        - Spectra/ subdirectory with .prn files
        - PT_Log.txt (optional)
    
    Returns
    -------
    spectra : np.ndarray
        Spectral intensity array, shape (Nt, Nl)
    w : np.ndarray
        Wavenumber array (cm^-1), shape (Nl,)
    P : float
        Gas cell pressure in bar
    T : float
        Gas cell temperature in Kelvin
    datetime : np.ndarray or None
        Datetime array for each spectrum
        
    Example
    -------
    >>> spectra, wv, P, T, dt = read_data('/path/to/experiment')
    >>> print(f"Loaded {len(spectra)} spectra at {T} K, {P} bar")
    """
    # Read spectra
    spectra_dir = os.path.join(directory, 'Spectra')
    spectra, w = read_spectra(spectra_dir)
    
    # Read pressure and temperature
    P, T, datetime = get_pt(directory)
    
    # If no datetime from PT log, create sequential array
    if datetime is None:
        datetime = np.arange(0, len(spectra), 1)
    
    return spectra, w, P, T, datetime


def get_compounds(file):
    """
    Load compound definitions from a pickle file.
    
    The compound dictionary defines which species to look for and which
    spectral windows to use for each species.
    
    Parameters
    ----------
    file : str or Path
        Path to pickle file containing compound dictionary
    
    Returns
    -------
    dict
        Compound dictionary with structure:
        {
            'CO2': {'bounds': [[2200, 2400], [3500, 3800]]},
            'CO': {'bounds': [[2000, 2250]]},
            ...
        }
        
    Example
    -------
    >>> compounds = get_compounds('compounds.pkl')
    >>> print(compounds['CO2']['bounds'])
    [[2200, 2400], [3500, 3800]]
    """
    with open(file, 'rb') as handle:
        compounds = pkl.load(handle)
    
    print(f"Loaded {len(compounds)} compound definitions")
    return compounds


def save_results(concentrations, uncertainties, species_list, datetime, 
                result_dir, prefix=''):
    """
    Save concentration retrieval results.
    
    Parameters
    ----------
    concentrations : np.ndarray
        Retrieved concentrations, shape (Ns*Nt,)
    uncertainties : np.ndarray
        1-σ uncertainties, shape (Ns*Nt,)
    species_list : list
        Names of species
    datetime : np.ndarray
        Datetime array for each timestep
    result_dir : str or Path
        Directory to save results
    prefix : str, optional
        Prefix for output files
    
    Notes
    -----
    Saves both numpy arrays and CSV file for easy analysis.
    """
    import os
    os.makedirs(result_dir, exist_ok=True)
    
    Ns = len(species_list)
    Nt = len(datetime)
    
    # Reshape to (Nt, Ns)
    conc_reshaped = concentrations.reshape(Ns, Nt, order='F').T
    uncert_reshaped = uncertainties.reshape(Ns, Nt, order='F').T
    
    # Save as numpy arrays
    np.save(f'{result_dir}/{prefix}concentrations.npy', conc_reshaped)
    np.save(f'{result_dir}/{prefix}uncertainties.npy', uncert_reshaped)
    
    # Save as CSV
    df = pd.DataFrame(conc_reshaped, columns=species_list)
    df['datetime'] = datetime
    df.to_csv(f'{result_dir}/{prefix}concentrations.csv', index=False)
    
    print(f"Results saved to {result_dir}")
