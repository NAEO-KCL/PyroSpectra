# PyroSpectra

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Biomass burning emission factors from closed-path FTIR time series spectra.

Implements the methodology of Richardson-Foulger, Wooster, Gómez-Dans & Grosvenor (2026), *JGR: Biogeosciences* — from raw MATRIX-MG5 spectra through to emission factors by carbon mass balance.

**Version 2.0 adds an optically-thick, high-concentration retrieval (`thick_smoke`) for
close-to-source, undiluted smoke, a rigorous per-species detection framework, and a
publication-quality diagnostic-plot suite. The v1.1 temporal-regularisation retrieval is
unchanged and remains the default for diluted/stabilised time series. Read
[CHANGELOG.md](CHANGELOG.md).** (v1.1 corrected four v1.0 defects and added the reference
spectra v1.0 could not generate; emission ratios, factors and MCE were unchanged there.)

---

## Two complementary retrievals

| | `inversion` (v1.1, default) | `thick_smoke` (v2.0) |
|---|---|---|
| Best for | diluted / stabilised time series | close-to-source, **optically-thick** smoke |
| Estimator | CLS + **temporal** Tikhonov regularisation | **non-negative** bounded per-time-step LS |
| Interferents | windowed references | CO2/H2O/CO **full-spectrum** + per-window continuum term |
| Saturation | window choice | window choice + **optical-depth channel weighting** |
| Ratios | per-time-step | slope-based, with an optically-thin regime for N2O |
| Detection | lasso screening | lasso + **3σ detection limit** + carbon-plausibility |

---

## Installation

```bash
pip install -e .
```

Requires `numpy`, `scipy`, `scikit-learn`, `pandas`, `matplotlib`, `joblib`. `radis` is
needed only for line-by-line reference generation and is imported lazily — the
retrieval and emission factor code runs without it.

---

## Pipeline

```python
from pyrospectra import (read_data, get_compounds, generate_reference, process_spectra,
                         lasso_inversion, temporally_regularised_inversion, l_curve,
                         emission_factors, summarise, save_results, align_datetime)

# 1. Load a burn
spectra, w, P, T, dt = read_data('burns/peat_01')

# 2. Species, windows and databanks come from the registry (Table D1), not a pickle
compounds = get_compounds()

# 3. Reference matrix. Cross-section species need data files - see DATA_SOURCES.md
ref, full, mask, prov = generate_reference(
    'results/peat_01', compounds, w, P, T, sigma=0.5,
    xsec_paths={'C3H6O': 'data/xsec/acetone.xsc',
                'C5H8':  'data/xsec/isoprene.xsc'})

# 4. Observed absorbance, baseline from the stabilised pre-ignition block
obs, full_obs = process_spectra(spectra, mask, 'results/peat_01', n_preignition=20)

# 5. Which species are detectably present
ref, full, obs, species, score = lasso_inversion(ref, full, obs, compounds, seed=42)

# 6. Choose gamma from the L-curve - per fuel type, and per penalty form
gamma = l_curve(ref, obs, penalty='paper')['gamma_optimal']

# 7. Retrieve. Returns (Ns, Nt) arrays in ppm, already correctly shaped
result = temporally_regularised_inversion(ref, obs, gamma, 'results/peat_01',
                                          list(species), penalty='paper')

# 8. Emission factors by carbon mass balance
ef = emission_factors(result.concentrations, result.species, fuel='boreal_peat',
                      n_background=20, uncertainty=result.uncertainty)
print(summarise(ef))

save_results(result, datetime=align_datetime(dt, score),
             result_dir='results/peat_01', emission_result=ef)
```

Burn directories are large — ~375 MB of `.prn` text for 666 spectra. To move or archive
them, `examples/pack_for_transfer.py` writes a compressed `.npz` about 15x smaller,
keeping only the channels inside a Table D1 window; `read_data` accepts the archive
directly.

`examples/example_workflow.py` runs this end to end;
`examples/l_curve_optimization.py` compares the two penalty forms.

### Optically-thick retrieval (v2.0)

For close-to-source, undiluted smoke where CO2/H2O saturate and strong absorbers bleed into
the trace windows, use `thick_smoke`:

```python
import numpy as np
from pyrospectra import thick_smoke as TS
from pyrospectra import diagnostics
from pyrospectra.io_utils import read_spectra, get_pt

TS.configure(xsec_dir='data/xsec', refcache='results/_refcache')   # no hard-coded paths
spec, w = read_spectra('burns/corn_01/Spectra')
P, T, _ = get_pt('burns/corn_01')

rec = TS.retrieve(spec, w, P, T, fuel_cf=0.45, return_fit=True)     # one burn
for s in rec['species']:
    if TS.detected(rec, s):
        print(s, rec['EF'][s], 'g/kg')                             # quantitative
    elif TS.detection_limit_ok(rec, s):
        print(s, '<', rec['EF_DL'][s], 'g/kg (3-sigma)')           # upper bound

diagnostics.all_diagnostics(rec, 'results/corn_01')                 # spectral atlas, L-curve, ...
```

`examples/thick_smoke_workflow.py` runs this end to end for one burn.

---

## Methodology

**Baseline** — optimised asymmetric least squares (Dong & Xu 2024) on the stabilised
pre-ignition block, kept sparse so it scales to the ~29,900-channel MG5 grid.

**Reference spectra** — two pathways feeding one matrix. Line-by-line via RADIS from
HITRAN, HITEMP or GEISA; measured cross-sections via `xsections` for the heavy VOCs
that have no line list. Both are returned as decadic absorbance of 1 ppm over the 5 m
path, which is what makes them mixable. Every source file's SHA-256 is recorded.

**Species identification** — L1-regularised regression with 5-fold cross-validation,
non-negative, over 10–20 randomly sampled time-steps; a species zero at all of them is
judged undetectable and dropped.

**Retrieval** — Tikhonov regularisation on the temporal difference of the concentration
series, `argmin ||Ac − y||² + gamma||Dc||²`. Solved exactly by eigen-decoupling into Nt
independent Ns×Ns systems, so neither A nor a dense `(Ns·Nt)²` inverse is ever formed.

**Uncertainty** — `sigma_i = sigma_eps · sqrt([(AᵀA + gamma·DᵀD)⁻¹]_ii)`, in ppm, with
`sigma_eps` from the unregularised fit residuals. Reduces exactly to Eq. 2 at gamma = 0.

**Emission factors** — per-time-step emission ratios against a time-weighted
pre-ignition background, then carbon mass balance (Eqs. 4–7).

---

## Things to know before trusting the output

**γ does not carry across penalty forms.** `penalty='paper'` (Eq. 3, curvature²) and
`penalty='legacy'` (v1.0, first-order) put the L-curve corner in different places. The
values quoted in the manuscript were obtained under one convention; re-run `l_curve`.

**γ constrains species unequally.** A single scalar competes against `G_ii`, which
scales with the square of the reference amplitude and spans orders of magnitude between
CO₂ and a trace VOC. Check `result.effective_smoothing`; the retrieval warns when the
spread exceeds 100×.

**Cross-section species are extrapolated in temperature.** The cell runs at 448 K; the
libraries are measured near ambient. Contour comes from the nearest measurement and
band strength from the trend, but hot-band population is not corrected. Emission
factors for HNO₂, CH₃COOH, C₂H₆O, C₃H₆O, C₄H₄O, C₅H₈ and CH₃CHO carry a systematic that
the posterior covariance does not contain. See §4 of `DATA_SOURCES.md`.

**The smoothness prior rings at sharp transitions.** Where concentrations genuinely
change within the sampling interval — flaming ignition, not smouldering decay — the
curvature penalty smears the step and overshoots around it. Pinned by
`test_smoothness_prior_smears_a_step`.

**`emissions.py` is new in v1.1** and has not been cross-checked against the authors'
own EF pipeline. Verify one burn against a known result first.

**Two registry entries need your decision.** Table D1 attributes ethanol, acetone, furan
and isoprene to GEISA, which has no line lists for them; and CH₃CHO is reported in
Figures 8–9 but has no window in Table D1, so the registry carries a placeholder that
overlaps SO₂ and HNO₂. Both are flagged in `registry.py`.

---

## Tests

```bash
python -m pytest pyrospectra/tests -v
```

36 tests, each against an analytically known answer, so a regression in the conventions clearly fails rather than producing plausible numbers.

---

## Citation

```bibtex

 @misc{richardsonfoulger_gomezdans_2026, 
    title={{PyroSpectra}: Biomass Burning Emission Factors From {FTIR} Time Series Spectra}, 
    url={https://zenodo.org/records/18552195}, 
    DOI={10.5281/zenodo.18552195}, 
    abstractNote={A Python package for analysing biomass burning emissions using closed-path Fourier Transform Infrared (FTIR) spectroscopy, implementing temporally regularised concentration retrievals with automated species identification.}, 
    publisher={Zenodo}, 
    author={Richardson-Foulger, Luke and Gómez-Dans, José}, 
    year={2026}}

@article{richardsonfoulger2026ftir,
  title  = {Laboratory use of a Closed-Path 'Industrial Emissions' {FTIR} Spectrometer for High-Concentration Sampling of Biomass Burning Smoke and Retrieval of
            Fire Emission Factors},
  author = {Richardson-Foulger, Luke and Wooster, Martin and
            G{\'o}mez-Dans, Jos{\'e} and Grosvenor, Mark},
  journal= {Journal of Geophysical Research: Biogeosciences},
  year   = {2026}}

```

Also cite RADIS (Pannier & Laux 2019) and whichever spectroscopic databases you used — see `DATA_SOURCES.md`.

## Licence

MIT. Spectroscopic data are **not** redistributed with this package and carry their own terms.

## Authors

Luke Richardson-Foulger, José Gómez-Dans
Leverhulme Centre for Wildfires, Environment and Society / NERC National Centre for
Earth Observation, King's College London
