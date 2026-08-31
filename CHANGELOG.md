# Changelog

All notable changes to this project are documented in this file.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/);
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-08-29

Adds an **optically-thick, high-concentration retrieval** for close-to-source, undiluted
smoke (peak excess CO2 ~9,000-14,000 ppm; saturated H2O bands), together with a rigorous
per-species detection framework and a professional diagnostic-plot suite. The core
temporal-regularisation retrieval (`inversion.py`) is unchanged and remains the default for
diluted / stabilised time series; the new pathway (`thick_smoke.py`) is complementary.

### Added

- **`pyrospectra.thick_smoke`** - the optically-thick retrieval. Public entry points
  `retrieve_thick` (per-burn concentrations, emission ratios/factors, detection stats),
  `thick_reference` (reference matrix), `thick_emission_ratios`, `is_detected`. Paths are
  configured with `thick_smoke.configure(xsec_dir=..., refcache=...)`; no paths are hard-coded.
  Method elements:
  - **CO2 isotopologues** (`isotope='1,2,3'`) - 13-CO2 and hot bands in the N2O / CO windows.
  - **Pervasive absorbers CO2/H2O/CO modelled full-spectrum** - their columns are pinned by
    their reliable unsaturated bands and subtracted inside the crowded trace windows.
  - **Per-window non-negative continuum pseudo-species** in every trace window - absorbs the
    smoke-correlated broadband continuum that a species would otherwise fit (a high-R2 false
    detection); excluded from the N2O window (interference, not continuum).
  - **Non-negative bounded per-time-step least squares** (`scipy.optimize.lsq_linear`, `bvls`).
  - **Optical-depth channel weighting** - saturated channels (|A| above a linearity cap) are
    down-weighted.
  - **Slope-based emission ratios** over the plume, with an optically-thin (linear-regime)
    ratio for interference-limited species (N2O).
  - **C_T from CO2+CO+CH4 only** (>98% of emitted carbon), so over-retrieved VOCs do not
    inflate the total carbon and bias every EF low.
- **Per species-burn detection test** (`is_detected`, `detection_limit_ok`): a species is
  detected only if the emission-ratio slope exceeds 3x its standard error (with an R2 / SNR
  goodness gate) **and** its emitted carbon does not exceed CO's (a physical bound that rejects
  interferent-residual artefacts). Non-detections are reported as 3-sigma detection limits or
  indicative estimates, never as negative or spurious values.
- **`pyrospectra.diagnostics`** - publication-quality PDF diagnostics from a
  `retrieve_thick(..., return_fit=True)` record: `spectral_atlas` (per-species window
  decomposition), `hysteresis` (excess-X vs excess-CO2), `lcurve_residual` (Tikhonov L-curve +
  residual spectrum), `overview` (full-region fit + concentration time series), `all_diagnostics`.
- **12 heavy-VOC cross-section species** integrated via measured HITRAN `.xsc` files (acetone,
  isoprene, furan, ethanol, acetic acid, HONO, acetonitrile, methyl formate, propene, propane,
  toluene, acetaldehyde), state-selected to the cell (T, P).

### Changed / corrected windows

- **HCN** moved onto its nu1 Q-branch (3306-3325 cm-1; the old 3313-3330 sat on the R-branch
  shoulder and missed the Q-branch). HCN is nonetheless below detection at this optical depth
  (its nu2 bend at 712 cm-1, ~10x stronger, is swamped by the CO2 bending band) - documented.
- **NO2** generated line-by-line at its nu3 band (~1617 cm-1); the commonly distributed NO2
  cross-section is the UV-visible spectrum (15,000-42,000 cm-1) and is unusable in the mid-IR.
- **Acetonitrile** moved off the CO2 nu3 wing (where it fit CO2 residual) to its nu4 band;
  **propane** moved off the CH4/H2O line forest.

### Notes

- The effective instrument lineshape measured from isolated ambient lines is ~0.8-0.95 cm-1
  FWHM (Gaussian sigma ~0.40); retrieving at 0.40 vs the default 0.50 changes emission factors
  by <2% (integrated-absorption retrieval + saturated-core down-weighting), so 0.50 is retained.

## [1.1.0] - 2026-08-26

Corrections to the retrieval, and reference spectra for the species v1.0 could not
generate. **Absolute retrieved mixing ratios change; emission ratios, emission factors
and MCE do not** — see "Effect on published results" below.

### Fixed

- **Absorbance convention (`conventions.py`).** RADIS defines its `'absorbance'` array
  as napierian optical depth — throughout `radis/spectrum/rescale.py` the canonical
  conversion is `transmittance_noslit = exp(-absorbance)`. v1.0 fitted it directly
  against observations computed as `-log10(I/I0)`, a decadic quantity, so every
  retrieved concentration was low by a factor ln(10) = 2.302585. The package now works
  in decadic absorbance throughout and converts RADIS output on the way in. Set
  `ABSORBANCE_CONVENTION = 'napierian'` to reproduce v1.0.

- **Penalty form (`inversion.py`, `preprocessing.py`).** Eq. 3 is `||Dc||²`, which
  contributes `gamma · DᵀD` to the normal equations. v1.0 assembled
  `AᵀA + lambda · D`. Since D is the path-graph Laplacian, `cᵀDc = Σ(c_{t+1} − c_t)²`,
  so v1.0 in fact applied a first-order smoothness penalty — which is what its README
  describes, and is a reasonable prior, but is one power of D away from the manuscript.
  `penalty='paper'` (default) now implements Eq. 3; `penalty='legacy'` reproduces v1.0.
  **gamma is not transferable between the two forms**; re-run the L-curve.

- **Uncertainty scale (`inversion.py`).** v1.0 returned
  `sqrt(diag((AᵀA + lambda·D)⁻¹))` with no `sigma_eps` factor, so the reported 1-sigma
  was in inverse-absorbance units rather than ppm and did not depend on how well the
  spectra actually fitted. It is now scaled by the residual noise level, is in ppm, and
  reduces exactly to Eq. 2 when gamma = 0.

- **How `sigma_eps` is estimated (`inversion.py`).** Taking the RMS of the *regularised*
  fit residuals, the literal reading of Section 2.3.1, conflates measurement noise with
  smoothing bias: on a test case with 3×10⁻⁵ absorbance noise it returns 3.0×10⁻⁵ at
  gamma = 0 but 2.4×10⁻² at gamma = 10⁻³, a factor of 800. Since the reported
  uncertainty is proportional to `sigma_eps`, that makes uncertainty *grow* with
  regularisation, inverting the effect reported in Section 3.2. The default
  `noise_estimate='cls'` uses the unregularised residuals — a property of the data, not
  of the prior. `'rms'`, `'reduced_chi2'` and a supplied float remain available.

- **Concentration vector ordering (`io_utils.py`, `inversion.py`).** The solution vector
  is species-major (`i·Nt + t`). v1.0's `save_results`, `inversion_residual` and example
  plotting reshaped it with `order='F'`, transposing species against time: the saved CSV
  carried correct column names over scrambled data. Results are now returned already
  shaped `(Ns, Nt)`.

- **Double square root (`inversion.py`).** `inversion_residual` took `np.sqrt(sigma)` of
  a quantity that was already 1-sigma.

- **Datetime alignment (`io_utils.align_datetime`).** `lasso_inversion` removes sampled
  time-steps from the middle of the series; v1.0's example truncated the *tail* of the
  datetime array instead, mislabelling every spectrum after the first removed one.

- **Unseeded species selection (`species_selection.py`).** `random.sample` was unseeded,
  so the species set — and which spectra entered the retrieval — differed between runs
  of the same data. Seeded (default 42) and recorded in the diagnostics.

- **Natural sort of spectra (`io_utils.read_spectra`).** Lexical sorting places
  `spectrum_10` before `spectrum_9` whenever the index is not zero-padded, shuffling the
  time axis and silently invalidating the temporal constraint.

### Added

- **`xsections.py` — measured cross-section pathway.** HITRAN `.xsc` and PNNL
  two-column readers; (T, P) state selection with linear interpolation, band-strength
  extrapolation and refusal of implausible rescalings; instrument lineshape applied in
  quadrature against the library resolution rather than as if monochromatic; conversion
  to absorbance at 1 ppm over the cell path via the ideal gas law. This supplies
  reference spectra for HNO₂, CH₃COOH, C₂H₆O, C₃H₆O, C₄H₄O, C₅H₈ and CH₃CHO, none of
  which has a line list. **The data files are not bundled** — see `DATA_SOURCES.md`.

- **`registry.py` — species registry.** Table D1 windows, databank assignment,
  databank-specific molecule names, molar masses, carbon numbers and the Table 1 fuel
  carbon fractions, all under version control. Replaces `compounds.pkl`, which could
  drift from the manuscript unobserved.

- **`emissions.py` — emission factors.** Eqs. 4–7: per-time-step emission ratios against
  a time-weighted pre-ignition background, total carbon, carbon mass balance emission
  factors and MCE, with a minimum-excess-CO₂ threshold so the smouldering tail does not
  contribute ratios of noise to noise. v1.0's changelog listed emission factors as a
  feature but shipped no such code. **This is new and has not been cross-checked against
  the authors' own EF pipeline.**

- **HITEMP and GEISA support** in the reference generator, per Table D1 (NO₂ from
  HITEMP). Local `.par` paths are accepted in place of a databank name, for offline use.

- **Exact fast solver.** `AᵀA = kron(RRᵀ, I_Nt)` and `DᵀD = U diag(nu²) Uᵀ`, so the
  normal equations decouple in the eigenbasis of D into Nt independent Ns×Ns systems.
  Gives the exact solution and the exact posterior diagonal in O(Nt·Ns³). Replaces the
  dense `(Ns·Nt)²` inverse, which needed 1 GB at Ns=16, Nt=700 and 4.6 GB at Nt=1500.
  A is never formed.

- **`classical_least_squares`** — the Section 3.2 benchmark, implemented as gamma = 0 of
  the same estimator so a comparison isolates the temporal constraint and nothing else.

- **`l_curve`** — corner location by maximum distance from the endpoint chord.

- **Provenance.** `reference_information/reference_provenance.json` records, per species,
  the databank, source file and SHA-256, the cell state used, any temperature
  extrapolation and rescaling, and the lineshape treatment.

- **Reference caching**, keyed on species, window, cell state and source checksum.

- **`effective_smoothing` diagnostic.** A single scalar gamma does not constrain all
  species equally: it competes against `G_ii`, which scales with the square of the
  reference amplitude and spans orders of magnitude between CO₂ and a trace VOC. The
  retrieval now reports `gamma·<mu>/G_ii` per species and warns when the spread exceeds
  100×.

- **`packing.py` — compact burn archives.** A 666-spectrum burn is ~375 MB of `.prn`
  text, of which about half the channels lie outside every Table D1 window, the
  wavenumber column is repeated in every file, and the intensities carry far more digits
  than detector noise supports. `pack_burn` writes a single compressed `.npz` of the
  retained channels as float32 — ~15x smaller, round-trip verified — and `read_data`
  accepts the archive wherever it accepts a directory.

- **Test suite** (38 tests) pinning the conventions against analytically known answers:
  ordering, penalty form, uncertainty scaling and its reduction to Eq. 2, fast-solver
  equivalence to the explicit normal equations, absorbance base, Loschmidt number,
  cross-section unit conversion, band-strength conservation under convolution.

### Changed

- **Failures raise.** `generate_reference` collects every species it could not build and
  raises listing them. `on_error='warn_and_drop'` restores v1.0's behaviour of
  continuing, but warns rather than staying silent.

- **Sparse O-ALS baseline.** v1.0 called `.toarray()` on the second-difference matrix and
  used `np.diag(w)`, allocating three dense `(L, L)` arrays: ~7 GB each on the MG5 grid
  (~29,900 channels), so the call could not complete. The arithmetic is unchanged.

- **Baseline from the pre-ignition block.** Section A1 takes I₀ from the stabilised
  pre-ignition spectra; v1.0 used `spectra[0]` alone, carrying one scan's detector noise
  into every absorbance in the burn. Pass `n_preignition=` to average. Defaults to 1, so
  nothing changes unless you say how many scans you have.

- **QC plots are opt-in.** v1.0 wrote one PDF per time-step — 666 files for one burn.

- **Non-negative lasso coefficients** by default: they stand for concentrations, and a
  negative coefficient lets one species' noise cancel another's, which is the
  misattribution the step exists to prevent.

- **`spilu` removed.** v1.0 used an incomplete LU factorisation as though it were a
  direct solver. On the cases tested it happened to agree with `spsolve`, but the
  approximation is not guaranteed. The solver is now exact.

- Package imports RADIS lazily, so the retrieval and emission factor code is usable on a
  machine with no spectroscopic databases installed.

### Effect on published results

| Quantity | Affected? |
|---|---|
| Emission ratios, emission factors, MCE, total carbon | **No** — all are ratios of retrieved concentrations, so the ln(10) factor cancels exactly |
| Absolute mixing ratios in ppm | **Yes** — v1.0 values are low by 2.302585× |
| Detection limits, absolute comparisons against LGR/Aeris/OPUS GA | **Yes** |
| Reported 1-sigma retrieval uncertainties | **Yes** — v1.0 values are not in ppm |
| Species retrieved | **Yes** — eight were silently dropped |

The `order='F'` reshape affects anything that passed through `save_results` or the
example plotting; results taken directly from the solver's return value are unaffected.

---

## [1.0.0] - 2026-02-09

- Initial release. Spectral preprocessing with O-ALS baseline correction; reference
  spectra generation via RADIS and HITRAN; automated species identification by lasso
  regression with cross-validation; temporally regularised retrieval; L-curve parameter
  selection; uncertainty quantification via posterior covariance; correlation matrix
  visualisation; MATRIX-MG5 I/O; example workflows.
