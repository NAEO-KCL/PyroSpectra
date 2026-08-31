"""
PyroSpectra - biomass burning emission factors from FTIR time series spectra.

Closed-path FTIR retrieval of trace gas concentrations from biomass burning smoke,
with temporal regularisation, automated species identification, propagated
uncertainties, and emission factor derivation by carbon mass balance.

Implements the methodology of Richardson-Foulger, Wooster, Gomez-Dans & Grosvenor
(2026), JGR: Biogeosciences.

Quick start
-----------
>>> from pyrospectra import read_data, get_compounds, generate_reference
>>> from pyrospectra import process_spectra, lasso_inversion
>>> from pyrospectra import temporally_regularised_inversion, emission_factors
>>> spectra, w, P, T, dt = read_data('burn_01')
>>> compounds = get_compounds()                       # Table D1, from the registry
>>> ref, full, mask, prov = generate_reference('out', compounds, w, P, T, sigma=0.5,
...                                            xsec_paths={'C3H6O': 'data/acetone.xsc'})
>>> obs, _ = process_spectra(spectra, mask, 'out', n_preignition=20)
>>> ref, full, obs, species, score = lasso_inversion(ref, full, obs, compounds)
>>> result = temporally_regularised_inversion(ref, obs, 1e-3, 'out', list(species))
>>> ef = emission_factors(result.concentrations, result.species, fuel='boreal_peat',
...                       uncertainty=result.uncertainty)

Note ``reference`` imports RADIS lazily, so the rest of the package - including the
retrieval and the emission factor calculation - is usable on a machine with no
spectroscopic databases installed.

Version 2.0 adds :mod:`pyrospectra.thick_smoke`, an optically-thick, high-concentration
retrieval for close-to-source undiluted smoke (full-spectrum pervasive absorbers,
per-window non-negative continuum, non-negative bounded least squares, optical-depth
channel weighting, slope-based emission ratios and a 3-sigma detection framework), plus
:mod:`pyrospectra.diagnostics` for publication-quality PDF diagnostics. See
``docs/METHODS_optically_thick.md`` and ``examples/thick_smoke_workflow.py``.

>>> from pyrospectra import thick_smoke as TS, diagnostics
>>> TS.configure(xsec_dir='data/xsec', refcache='results/_refcache')
>>> rec = TS.retrieve(spectra, w, P, T, fuel_cf=0.45, return_fit=True)
>>> [s for s in rec['species'] if TS.detected(rec, s)]      # detected species
>>> diagnostics.all_diagnostics(rec, 'results/corn_01')     # spectral atlas, L-curve, ...
"""

__version__ = "2.0.0"
__author__ = ("Luke Richardson-Foulger, Martin Wooster, Jose Gomez-Dans, "
              "Mark Grosvenor")

from .conventions import (
    ABSORBANCE_CONVENTION, LN10, cross_section_to_absorbance,
    from_radis_absorbance, number_density, pnnl_to_absorbance,
)
from .registry import (
    CARBON_FRACTIONS, CARBON_NUMBER, CORE_SPECIES, MOLAR_MASS, SPECIES_REGISTRY,
    XSEC_SPECIES, build_compounds, carbon_number, get_registry, molar_mass,
)
from .preprocessing import (
    build_A_matrix, create_smoother, estimate_background, get_baseline,
    penalty_eigenvalues, penalty_matrix, process_spectra,
)
from .xsections import (
    CrossSectionBand, CrossSectionError, apply_instrument_lineshape,
    band_to_absorbance, cross_section_reference, file_checksum, instrument_kernel,
    load_cross_section, read_hitran_xsc, read_pnnl, select_state,
)
from .packing import (
    fitted_channel_mask, pack_burn, pack_directory_tree, read_packed,
)
from .species_selection import filter_compounds, fit_lasso, lasso_inversion
from .inversion import (
    RetrievalResult, classical_least_squares, inversion_residual, l_curve,
    temporally_regularised_inversion,
)
from .emissions import (
    background_mixing_ratio, emission_factors, emission_ratio,
    excess_mixing_ratios, modified_combustion_efficiency, summarise, total_carbon,
)
from .io_utils import (
    align_datetime, get_compounds, get_pt, read_data, read_spectra, read_spectrum,
    save_results,
)
# v2.0: optically-thick, high-concentration retrieval (does not import RADIS or matplotlib at import).
from . import thick_smoke
from .thick_smoke import (
    retrieve_thick, thick_reference, thick_emission_ratios, is_detected,
)


def generate_reference(*args, **kwargs):
    """Generate the reference matrix. Imports RADIS on first call."""
    from .reference import generate_reference as _gen
    return _gen(*args, **kwargs)


def __getattr__(name):
    """Expose reference/diagnostics names lazily so RADIS and matplotlib import only when needed."""
    if name in ("ReferenceGenerationError", "gaussian", "get_reference_matrix"):
        from . import reference
        return getattr(reference, name)
    if name == "diagnostics":
        import importlib
        return importlib.import_module(".diagnostics", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # conventions
    "ABSORBANCE_CONVENTION", "LN10", "from_radis_absorbance", "number_density",
    "cross_section_to_absorbance", "pnnl_to_absorbance",
    # registry
    "SPECIES_REGISTRY", "XSEC_SPECIES", "CORE_SPECIES", "MOLAR_MASS",
    "CARBON_NUMBER", "CARBON_FRACTIONS", "get_registry", "build_compounds",
    "carbon_number", "molar_mass",
    # preprocessing
    "get_baseline", "estimate_background", "process_spectra", "create_smoother",
    "penalty_matrix", "penalty_eigenvalues", "build_A_matrix",
    # cross-sections
    "CrossSectionBand", "CrossSectionError", "read_hitran_xsc", "read_pnnl",
    "load_cross_section", "select_state", "instrument_kernel",
    "apply_instrument_lineshape", "band_to_absorbance", "cross_section_reference",
    "file_checksum",
    # reference
    "generate_reference", "ReferenceGenerationError", "gaussian",
    # species selection
    "lasso_inversion", "fit_lasso", "filter_compounds",
    # packing
    "pack_burn", "pack_directory_tree", "read_packed", "fitted_channel_mask",
    # inversion
    "temporally_regularised_inversion", "classical_least_squares", "l_curve",
    "inversion_residual", "RetrievalResult",
    # emissions
    "emission_factors", "emission_ratio", "excess_mixing_ratios",
    "background_mixing_ratio", "modified_combustion_efficiency", "total_carbon",
    "summarise",
    # io
    "read_data", "read_spectra", "read_spectrum", "get_compounds", "get_pt",
    "save_results", "align_datetime",
    # v2.0 optically-thick retrieval
    "thick_smoke", "retrieve_thick", "thick_reference", "thick_emission_ratios",
    "is_detected", "diagnostics",
]
