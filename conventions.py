"""
Absorbance conventions and physical constants.

WHY THIS MODULE EXISTS
----------------------
The observed spectra are converted to absorbance in ``preprocessing.process_spectra``
as the *decadic* absorbance

    A10 = -log10(I / I0)

RADIS, however, defines its ``'absorbance'`` spectral array as the *napierian*
optical depth: throughout ``radis/spectrum/rescale.py`` the canonical conversion is
``transmittance_noslit = exp(-absorbance)`` and ``abscoeff = -ln(T)/L``, i.e.

    tau = -ln(I / I0) = A10 * ln(10)

References
----------
Mayerhoefer & Popp (2019), ChemPhysChem 20(4), 511-515 - on the two conventions.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants (CODATA 2018)
# ---------------------------------------------------------------------------
LN10 = np.log(10.0)                 # 2.302585092994046
K_BOLTZMANN = 1.380649e-23          # J / K
AVOGADRO = 6.02214076e23            # 1 / mol
BAR_TO_PA = 1.0e5
TORR_TO_PA = 133.32236842105263
ATM_TO_PA = 101325.0

#: Absorbance convention used for BOTH observed and reference spectra.
#: 'decadic'  -> A = -log10(I/I0)        
#: 'napierian'-> tau = -ln(I/I0)           
ABSORBANCE_CONVENTION = "decadic"

_VALID_CONVENTIONS = ("decadic", "napierian")


def check_convention(convention=None):
    """Validate and return an absorbance convention string."""
    conv = ABSORBANCE_CONVENTION if convention is None else convention
    if conv not in _VALID_CONVENTIONS:
        raise ValueError(
            f"Unknown absorbance convention {conv!r}; expected one of {_VALID_CONVENTIONS}"
        )
    return conv


def from_radis_absorbance(tau, convention=None):
    """
    Convert a RADIS ``'absorbance'`` array to the package convention.

    RADIS returns napierian optical depth. Under the decadic convention this must be
    divided by ln(10) before it can be fitted against ``-log10(I/I0)``.

    Parameters
    ----------
    tau : np.ndarray
        RADIS ``'absorbance'`` (napierian optical depth), dimensionless.
    convention : str, optional
        'decadic' or 'napierian'. Defaults to :data:`ABSORBANCE_CONVENTION`.

    Returns
    -------
    np.ndarray
        Absorbance in the requested convention.
    """
    conv = check_convention(convention)
    return np.asarray(tau, dtype=float) / LN10 if conv == "decadic" else np.asarray(tau, dtype=float)


def number_density(pressure_bar, temperature_k, mole_fraction=1.0):
    """
    Number density of an absorber, in molecule cm^-3, from the ideal gas law.

    Parameters
    ----------
    pressure_bar : float
        Total gas pressure (bar).
    temperature_k : float
        Gas temperature (K).
    mole_fraction : float, optional
        Mole fraction of the absorbing species. Default 1.0 (i.e. total density).

    Returns
    -------
    float
        Number density in molecule cm^-3.
    """
    n_m3 = (pressure_bar * BAR_TO_PA) / (K_BOLTZMANN * temperature_k)
    return mole_fraction * n_m3 * 1.0e-6  # m^-3 -> cm^-3


def cross_section_to_absorbance(sigma_cm2, pressure_bar, temperature_k,
                                path_length_cm, mole_fraction=1e-6,
                                convention=None):
    """
    Convert an absorption cross-section to absorbance for a given cell state.

    Beer-Lambert in napierian form is ``tau = sigma * n * L``; the decadic absorbance
    is ``tau / ln(10)``.

    Parameters
    ----------
    sigma_cm2 : np.ndarray
        Absorption cross-section, cm^2 molecule^-1.
    pressure_bar, temperature_k : float
        Gas cell state.
    path_length_cm : float
        Optical path length in cm (500 cm for the MG5 cell).
    mole_fraction : float, optional
        Mole fraction the reference is generated at. Default 1e-6 (1 ppm), matching
        the line-by-line pathway, so retrieved coefficients are directly in ppm.
    convention : str, optional
        Absorbance convention. Defaults to :data:`ABSORBANCE_CONVENTION`.

    Returns
    -------
    np.ndarray
        Absorbance of ``mole_fraction`` of the species over ``path_length_cm``.
    """
    conv = check_convention(convention)
    n = number_density(pressure_bar, temperature_k, mole_fraction)
    tau = np.asarray(sigma_cm2, dtype=float) * n * path_length_cm
    return tau / LN10 if conv == "decadic" else tau


def pnnl_to_absorbance(k_ppm_m, pressure_bar, temperature_k,
                       path_length_cm, mole_fraction=1e-6,
                       reference_temperature_k=296.15,
                       reference_pressure_bar=ATM_TO_PA / BAR_TO_PA,
                       convention=None):
    """
    Convert a PNNL-style absorption coefficient to absorbance for a given cell state.

    PNNL (Sharpe et al. 2004) tabulates *decadic* absorbance per ppm per metre at
    296 K and 1 atm. Because a mole fraction maps to a number density through
    ``n ~ P / T``, moving to the cell state scales the coefficient by
    ``(T_ref / T)(P / P_ref)``.

    Parameters
    ----------
    k_ppm_m : np.ndarray
        PNNL absorption coefficient, (ppm m)^-1, decadic.
    pressure_bar, temperature_k : float
        Gas cell state.
    path_length_cm : float
        Optical path length in cm.
    mole_fraction : float, optional
        Mole fraction to generate the reference at. Default 1e-6 (1 ppm).
    reference_temperature_k, reference_pressure_bar : float, optional
        Conditions the PNNL data were tabulated at.
    convention : str, optional
        Absorbance convention.

    Returns
    -------
    np.ndarray
        Absorbance in the requested convention.
    """
    conv = check_convention(convention)
    ppm = mole_fraction * 1.0e6
    path_m = path_length_cm / 100.0
    density_scale = (reference_temperature_k / temperature_k) * (
        pressure_bar / reference_pressure_bar
    )
    a10 = np.asarray(k_ppm_m, dtype=float) * ppm * path_m * density_scale
    return a10 if conv == "decadic" else a10 * LN10


__all__ = [
    "LN10", "K_BOLTZMANN", "AVOGADRO", "BAR_TO_PA", "TORR_TO_PA", "ATM_TO_PA",
    "ABSORBANCE_CONVENTION", "check_convention", "from_radis_absorbance",
    "number_density", "cross_section_to_absorbance", "pnnl_to_absorbance",
]
