"""
FTIR Fire Emissions Analysis Package

A comprehensive toolkit for analyzing biomass burning emissions using 
closed-path Fourier Transform Infrared (FTIR) spectroscopy.

Author: Luke Richardson-Foulger et al.
Institution: King's College London
Date: February 2026
"""

__version__ = "1.0.0"
__author__ = "Luke Richardson-Foulger, Martin Wooster, José Gómez-Dans, Mark Grosvenor"

# Import main functions for easy access
from .preprocessing import (
    get_baseline,
    process_spectra,
    build_A_matrix,
    create_smoother
)

from .reference import (
    generate_reference,
    get_reference_matrix,
    get_reference_matrix_full,
    gaussian
)

from .species_selection import (
    lasso_inversion,
    fit_lasso,
    filter_compounds
)

from .inversion import (
    temporally_regularised_inversion,
    inversion_residual,
    reshape_residuals
)

from .io_utils import (
    read_data,
    read_spectra,
    read_spectrum,
    get_compounds,
    get_pt,
    save_results
)

__all__ = [
    # Preprocessing
    'get_baseline',
    'process_spectra',
    'build_A_matrix',
    'create_smoother',
    
    # Reference generation
    'generate_reference',
    'get_reference_matrix',
    'get_reference_matrix_full',
    'gaussian',
    
    # Species selection
    'lasso_inversion',
    'fit_lasso',
    'filter_compounds',
    
    # Inversion
    'temporally_regularised_inversion',
    'inversion_residual',
    'reshape_residuals',
    
    # I/O utilities
    'read_data',
    'read_spectra',
    'read_spectrum',
    'get_compounds',
    'get_pt',
    'save_results',
]
