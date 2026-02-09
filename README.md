![439965275-fa9d8640-c4ef-44e6-84ef-7c18750c825e](https://github.com/user-attachments/assets/8f95a25d-2626-409e-bbff-8081ebb8af92)

# PyroSpectra: Biomass Burning Emission Factors From FTIR Time Series Spectra

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python package for analysing biomass burning emissions using closed-path Fourier Transform Infrared (FTIR) spectroscopy.

## Overview

<!--- This package implements the methodology described in **Richardson-Foulger et al. (2026)** for retrieving fire emission factors from laboratory FTIR measurements. Key features include: --->

- **Spectral preprocessing** with optimized asymmetric least squares (O-ALS) baseline correction
- **Reference spectra generation** using RADIS and HITRAN molecular databases
- **Automated species identification** via L1-regularized (Lasso) regression
- **Temporally regularised retrievals** using Tikhonov regularization
- **Uncertainty quantification** through posterior covariance matrices
- **Emission factor calculations** using carbon mass balance method

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install from source

```bash
git clone https://github.com/your-repo/ftir-fire-emissions.git
cd ftir-fire-emissions
pip install -e .
```

### Dependencies

The package automatically installs:
```
numpy>=1.20.0
scipy>=1.7.0
matplotlib>=3.4.0
scikit-learn>=0.24.0
radis>=0.12.0
pandas>=1.3.0
tqdm>=4.62.0
joblib>=1.0.0
```

## Quick Start

### Complete Analysis Pipeline

```python
from ftir_fire_emissions import (
    read_data,
    get_compounds,
    generate_reference,
    process_spectra,
    lasso_inversion,
    temporally_regularised_inversion,
    save_results
)

# 1. Load spectral data from MATRIX-MG5 FTIR
spectra, wavenumbers, pressure, temperature, datetime = read_data('path/to/data')

# 2. Load compound definitions (species and spectral windows)
emission_species = get_compounds('compounds.pkl')

# 3. Generate reference spectra using RADIS/HITRAN
reference_spectra, full_reference, mask = generate_reference(
    result_dir='./results',
    emission_species=emission_species,
    w=wavenumbers,
    P=pressure,  # bar
    T=temperature,  # K
    sigma=0.5  # Instrumental broadening (cm⁻¹)
)

# 4. Process observed spectra (baseline correction + absorbance conversion)
observed_spectra, full_observed = process_spectra(spectra, mask, './results')

# 5. Identify present species using Lasso regression
(ref_filtered, full_ref_filtered, obs_filtered, 
 species_filtered, lasso_scores) = lasso_inversion(
    reference_spectra,
    full_reference,
    observed_spectra,
    emission_species
)

# 6. Perform temporally regularized inversion
concentrations, uncertainties = temporally_regularised_inversion(
    reference_spectra=ref_filtered,
    residual_spectra=obs_filtered,
    lambda_=1e-3,  # Regularization parameter (optimize with L-curve)
    result_dir='./results',
    compound_list=list(species_filtered.keys())
)

# 7. Save results
save_results(concentrations, uncertainties, 
             list(species_filtered.keys()), datetime,
             './results')
```

### Optimize Regularization Parameter

Use the L-curve method to find optimal λ:

```python
from ftir_fire_emissions.examples.l_curve_optimization import (
    compute_l_curve, find_corner
)

# Test range of λ values
lambda_range = np.logspace(-6, -1, 50)
residual_norms, solution_norms = compute_l_curve(
    ref_filtered, obs_filtered, lambda_range
)

# Find optimal value
corner_idx = find_corner(residual_norms, solution_norms)
lambda_optimal = lambda_range[corner_idx]
print(f"Optimal λ = {lambda_optimal:.2e}")
```

## Data Format

### Input Files

The package expects data from the Bruker MATRIX-MG5 FTIR spectrometer:

```
experiment_directory/
├── Spectra/
│   ├── spectrum_0000.prn
│   ├── spectrum_0001.prn
│   └── ...
└── PT_Log.txt  # Optional: pressure/temperature log
```

Each `.prn` file contains two columns:
```
wavenumber (cm⁻¹)    intensity
800.0                0.9234
800.5                0.9187
...
```

### Compound Definitions

Create a pickle file defining target species and spectral windows:

```python
import pickle

compounds = {
    'CO2': {'bounds': [[2200, 2400], [3500, 3800]]},
    'CO': {'bounds': [[2000, 2250]]},
    'CH4': {'bounds': [[1150, 1400], [2800, 3100]]},
    'H2O': {'bounds': [[1300, 2000], [3400, 4000]]},
    # Add more species...
}

with open('compounds.pkl', 'wb') as f:
    pickle.dump(compounds, f)
```

## Package Structure

```
ftir_fire_emissions/
├── __init__.py              # Package initialization
├── preprocessing.py         # Baseline correction, absorbance conversion
├── reference.py            # RADIS-based reference spectra generation  
├── species_selection.py    # Lasso regression for species ID
├── inversion.py            # Tikhonov regularization, uncertainty quantification
├── io_utils.py             # File I/O for MATRIX-MG5 data
├── examples/
│   ├── example_workflow.py      # Complete analysis pipeline
│   └── l_curve_optimization.py  # Regularization parameter selection
├── tests/                  # Unit tests
├── requirements.txt        # Python dependencies
├── setup.py               # Installation script
├── LICENSE                # MIT License
├── CITATION.cff          # Software citation metadata
└── README.md             # This file
```

## Methodology

### 1. Baseline Correction (O-ALS)

Optimized Asymmetric Least Squares removes instrumental drift:
```
minimize: ||W(y - baseline)||² + λ||D²·baseline||²
```
where asymmetric weights W penalize points above the baseline more heavily.

### 2. Reference Spectra

Generated using RADIS line-by-line radiative transfer:
- HITRAN 2020 molecular database
- Voigt line profiles with Doppler + pressure broadening
- Gaussian convolution for instrumental broadening

### 3. Species Identification (Lasso)

L1-regularized regression automatically selects present species:
```
minimize: ||Ax - y||² + α||x||₁
```
Non-zero coefficients indicate detected species.

### 4. Regularized Inversion (Tikhonov)

Temporal smoothness constraint reduces noise:
```
minimize: ||Ax - y||² + λ||Dx||²
```
where D is first-order difference operator.

### 5. Uncertainty Quantification

Posterior covariance matrix:
```
C = (AᵀA + λDᵀD)⁻¹
```
Diagonal: variances. Off-diagonal: correlations (spectral interference).

## Examples

See `examples/` directory for:
- `example_workflow.py`: Complete analysis pipeline
- `l_curve_optimization.py`: Regularization parameter selection

Run examples:
```bash
cd examples
python example_workflow.py
```

## Citation

If you use this software, please cite:

```bibtex
@article{richardson2026ftir,
  title={Use of a Closed-Path 'Industrial Emissions' FTIR Spectrometer for 
         Close-to-Source Laboratory Sampling of Biomass Burning Smoke and 
         Retrieval of Fire Emission Factors},
  author={Richardson-Foulger, Luke and Wooster, Martin and 
          G{\'o}mez-Dans, Jos{\'e} and Grosvenor, Mark},
  year={2026},
  month={February}
}
```

Also cite the RADIS package used for reference spectra:
```bibtex
@article{pannier2019radis,
  title={RADIS: A nonequilibrium line-by-line radiative code for CO₂ and 
         HITRAN-like database species},
  author={Pannier, Erwan and Laux, Christophe O},
  journal={Journal of Quantitative Spectroscopy and Radiative Transfer},
  volume={222},
  pages={12--25},
  year={2019}
}
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Authors

**Luke Richardson-Foulger**
**José Gómez-Dans**

Leverhulme Centre for Wildfires, Environment and Society  
Department of Geography, King's College London  
London, UK

## Acknowledgments

- NERC National Centre for Earth Observation (NCEO)
- Leverhulme Centre for Wildfires, Environment and Society
- RADIS development team

## Support

For questions or issues:
- Open an issue on GitHub
- Contact: [luke.richardson-foulger@kcl.ac.uk]

## References

Key papers implementing similar methodology:

- Dong & Xu (2024). Baseline estimation using O-ALS. *Measurement*, 233, 114731.
- Pannier & Laux (2019). RADIS radiative transfer code. *JQSRT*, 222, 12-25.
- Eilers (2003). A perfect smoother. *Analytical Chemistry*, 75(14), 3631-3636.
- Calvetti et al. (2000). Tikhonov regularization and the L-curve. *J. Comp. Appl. Math.*
