# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-02-09

### Added
- Initial release of FTIR Fire Emissions Analysis Package
- Spectral preprocessing with O-ALS baseline correction
- Reference spectra generation using RADIS and HITRAN databases
- Automated species identification via Lasso regression with cross-validation
- Temporally regularized concentration retrieval using Tikhonov regularization
- L-curve method for optimal regularization parameter selection
- Uncertainty quantification via posterior covariance matrices
- Correlation matrix visualization for spectral interference analysis
- Data I/O utilities for MATRIX-MG5 FTIR spectrometer files
- Example workflows demonstrating complete analysis pipeline
- Comprehensive documentation and docstrings
- Unit tests for core functionality

### Features
- Supports analysis of biomass burning smoke from closed-path FTIR measurements
- Handles complex gas mixtures with overlapping spectral features
- Reduces noise through temporal regularization while maintaining time resolution
- Automatically identifies present species from candidate list
- Computes emission factors and modified combustion efficiency (MCE)
- Saves results in both NumPy and CSV formats for easy analysis

### Documentation
- README with installation instructions and quick start guide
- Detailed API documentation in module docstrings
- Example scripts for common use cases
- CITATION.cff file for proper software citation
- MIT License

### References
Richardson-Foulger, L., Wooster, M., Gómez-Dans, J., & Grosvenor, M. (2026).
Use of a Closed-Path 'Industrial Emissions' FTIR Spectrometer for Close-to-Source
Laboratory Sampling of Biomass Burning Smoke and Retrieval of Fire Emission Factors.
