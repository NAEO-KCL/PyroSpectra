# Optically-thick retrieval — methods (`pyrospectra.thick_smoke`, v2.0)

*This document describes the algorithm implemented in `pyrospectra/thick_smoke.py` and
`pyrospectra/diagnostics.py`, as applied to a 19-burn agricultural-residue / tropical-wood
validation dataset. It is the reference description for the v2.0 optically-thick pathway.*

*An improved trace-gas retrieval for optically-thick closed-path FTIR biomass-burning spectra.*
This describes the retrieval as actually run on the 19-burn agricultural-residue / tropical-wood
dataset. It builds on the PyroSpectra v1.1 framework (Richardson-Foulger & Gómez-Dans) but departs
from the Table D1 windowed classical-least-squares/Tikhonov retrieval in several respects, made
necessary by the high optical depth of this smoke (peak excess CO2 ≈ 9,000–14,000 ppm; the H2O
bands are saturated, decadic absorbance |A| ≈ 6–7). Each departure is flagged **[NEW]**.

## 1. Spectra and cell state
Time-resolved single-beam spectra were recorded with a Bruker MATRIX-MG5 closed-path FTIR
(0.241 cm⁻¹ sampling, ~4 s per scan) over 800–7898 cm⁻¹ (channels below 800 cm⁻¹ discarded).
Cell pressure and temperature were read from the per-burn `*_PT_Log.txt` (median T ≈ 440.5 K,
P ≈ 0.84 bar); these feed the reference generation. [Note: the log columns are
pressure(mbar), temperature(°C); this was corrected relative to the shipped code.]

## 2. Absorbance and background
Single-beam spectra I(ν,t) were converted to decadic absorbance A(ν,t) = −log₁₀(I/I₀). The
background I₀ was the optimised asymmetric-least-squares (O-ALS) lower envelope of the median of
the stabilised pre-ignition scans; the number of pre-ignition scans was determined per burn from
the onset of a strong CO2-band (3650–3760 cm⁻¹) integrated-absorbance metric (§A1).
**[NEW]** Because the chamber was only partially flushed between burns, the pre-ignition state is
often not clean air; the initial (pre-ignition) retrieved concentration was taken as the
per-burn baseline as-is, and emission ratios formed against it (§6).

## 3. Reference spectra
Reference absorbances (1 ppmv over the 5 m path, at the cell T,P, in decadic units) were generated
per species. Line-by-line species used HITRAN through RADIS 0.17.1 with a Gaussian instrument
lineshape (σ = 0.5 cm⁻¹). The effective instrument FWHM measured from isolated unsaturated ambient
lines is ≈ 0.8–0.95 cm⁻¹ (implying σ ≈ 0.40); retrieving at σ = 0.40 vs 0.50 changes the reported
emission factors by < 2 % (the retrieval is driven by integrated band absorption, and saturated
line cores are down-weighted, §4), so σ = 0.50 was retained. Cross-section species (heavy VOCs with
no line list) used measured
HITRAN composite cross-sections, state-selected to the cell (T,P) with the band strength
extrapolated linearly in T where multiple temperatures were available.
- **[NEW] CO2 modelled with isotopologues 1,2,3** (rather than the principal isotopologue only):
  ¹³CO2 and hot-band absorption in the N2O (2206–2251 cm⁻¹) and CO (2115–2130 cm⁻¹) windows is
  otherwise unmodelled and misattributed to those trace species.
- **[NEW] CO2, H2O and CO modelled full-spectrum** (their references are non-zero across every
  fitting window, not only their own): these three pervasive absorbers otherwise leave residual
  absorption inside the narrow trace windows. In particular the CO fundamental R-branch
  (2140–2230 cm⁻¹) was the dominant interferent in the N2O window.
- Cross-section VOCs are all measured at 194–323 K and extrapolated ≥125 K to the 448 K cell; high-
  temperature Ar-broadened combustion sets were excluded. Their band contours are not corrected for
  hot-band population — a systematic that the posterior does not contain.

## 4. Retrieval (per time step)
Concentrations were retrieved on the union of the fitting windows (Table D1 windows for the
manuscript species; microwindows selected from established band positions for the additional
cross-section species). The forward model is linear, A(ν,t) = Σⱼ cⱼ(t) Rⱼ(ν), solved per time step by
**[NEW] bounded least squares** with gas concentrations constrained non-negative (they are physical)
and the baseline terms (below) free:
- **[NEW] Per-window continuum pseudo-species.** A single **non-negative constant** per fitting
  window was added to the reference matrix, in **every trace window** (both the sharp-line gas
  windows and the cross-section VOC windows), to absorb the smoke-correlated broadband continuum
  (aerosol scattering + pseudo-continuum + imperfect I₀ between strong bands; ~0.02–0.05 decadic
  absorbance at peak smoke). Non-negativity (a continuum is positive absorbance) removes the
  sign degeneracy that otherwise let a species inflate against a compensating negative offset; a
  constant (not a tilt) keeps it from absorbing band shape. This term is essential and not optional:
  **without a continuum term in a window, that window's species simply fits the continuum, yielding
  an inflated emission factor that still correlates with ΔCO2 (a high-R² false detection)** — e.g.
  omitting it inflated HCOOH from 0.5 to 9 g kg⁻¹ and C2H2 from 0.3 to 11 g kg⁻¹ on a test burn.
  The terms are excluded from all reported quantities. They are NOT added to the broad H2O windows
  or the CO window (band fills the window), nor to the **N2O** window: N2O's over-retrieval is
  CO2/CO interference rather than continuum, and a constant there wrongly floors it (§5, thin regime).
- **[NEW] Optical-depth channel weighting.** Channels whose 90th-percentile |A| over the burn
  exceeds a linearity cap (|A| > 2, i.e. transmission < 1%) were down-weighted, since the linear
  Beer–Lambert model is invalid in saturated line cores.

All configured species are carried in the joint fit (so each absorber's contribution is modelled
rather than misattributed); detectability is decided *after* the fit, per species and per burn, from
the emission-ratio significance and a physical-plausibility test (§6), not by dropping species a priori.

## 5. Emission ratios, emission factors and MCE
Excess mixing ratios ΔX(t) = X(t) − X_background were formed per species. Emission ratios were the
**[NEW] slope of ΔX vs ΔCO2** through the origin (robust to a constant background offset), with the
coefficient of determination R² retained as a quality metric. Two regimes are computed:
- **full plume** (ΔCO2 > 200 ppm) — used for the well-retrieved species;
- **[NEW] optically-thin / linear regime** (ΔCO2 < ~3000 ppm) — used for species whose full-plume
  ratio is inflated by optical thickness. The ΔX–ΔCO2 relationship is linear at low ΔCO2 and curves
  upward as the smoke becomes optically thick (H2O/CO2 saturation and super-linear interference);
  the linear-regime slope is the physical emission ratio. This matters most for **N2O**, whose
  full-plume ratio is inflated several-fold on the optically-thickest burns (peak ΔCO2 ~10,000 ppm)
  but which retrieves near-literature on the optically-thin burns and from the linear regime of the
  thick ones. The cap is fixed a priori from the observed linearity extent, not tuned to literature.
Emission factors follow the carbon mass balance,
  EF_X = F_C · 1000 · (M_X/12) · ER(X/CO2) / C_T ,
with fuel carbon fraction F_C (below) and **[NEW] total carbon C_T = 1 + ER(CO/CO2) + ER(CH4/CO2)**,
i.e. from CO2+CO+CH4 only (>98 % of emitted carbon). The minor carbon-bearing VOCs are
over-retrieved on this smoke and, if included in C_T (as in the strict Eq. 6), inflate it and bias
every EF low. MCE = 1/(1 + ER(CO/CO2)).

## 6. Detection, quality control and aggregation
**[NEW] Detection limit.** The emission ratio is the slope of ΔX vs ΔCO2 through the origin; its
standard error σ_slope = √(Σresid²/(n−1)) / √(Σ ΔCO2²) propagates the fit-residual scatter. A
species is *detected* on a burn only when the slope is significant and the fit is physical:
1. positive EF (a negative/anti-correlated slope is a non-detection, not a negative EF);
2. slope signal-to-noise SNR = |ER|/σ_slope ≥ 3, together with (R² ≥ 0.25 **or** SNR ≥ 10) — the R²
   arm requires the linear model to explain the excess variance, the SNR-override retains unambiguous
   species (CO, CH4, CH2O: SNR 13–22) whose R² is depressed by optical-depth scatter on the thickest
   plumes only, while a moderate-SNR / low-R² combination (a weak cross-section fitting interferent
   residual) fails both arms;
3. physical plausibility: the species' emitted carbon may not exceed CO's (CO is the dominant
   incomplete-combustion product). This rejects interferent-residual artefacts inflated to hundreds
   of g kg⁻¹ that no statistical test alone catches (e.g. a cross-section sitting in the CO2 ν3 wing).

Where a species is not detected, an **indicative estimate** is still reported — the maximum-likelihood
retrieved EF (median over the real-plume burns) with a 1σ uncertainty floored at the detection-limit
scale — so every species gets a best number with honest error bars. Sub-cases: *indicative ≈ 0* (the
value is consistent with zero); *upper limit* (the value sits well above the DL, i.e. the fit found
real absorption that failed the detection gate — residual from an overlapping absorber, so the true
value is ≤ the estimate); and *not constrained* where the band is too weak/overlapped to estimate at
all (DL undefined, ≈0, or > 10 g kg⁻¹ — e.g. propane, whose strongest available cross-section band
gives only ~3×10⁻⁴ absorbance per ppm). A cross-burn high-side
outlier guard rejects interference artefacts (a detected value > 10× the species' dataset median;
spectral interference inflates, never deflates, a retrieval — this removed an N2O value of 25 g kg⁻¹
on one long, high-CO2 burn). Emission factors are averaged per fuel over the surviving detected burns.
One burn (banana2) had no usable plume over a heavily smoky background and was dropped. Reported EF
and ER values are magnitudes (a negative, anti-correlated slope is a non-detection, not a negative EF).

**[NEW] Window selection was checked per species against the spectral fit** (diagnostic atlas,
`results/<burn>/diag_spectral_atlas.pdf`). Windows corrected as a result: HCN moved onto its ν1
Q-branch (3311.5 cm⁻¹); acetonitrile moved off the CO2 ν3 wing (where it had fitted 26,000 ppm of CO2
residual) to its ν4 CH3-rock band; propane moved off the CH4/H2O line forest. **HCN's stronger ν2
band (712.6 cm⁻¹, ~10× ν3) was also tested** by reading below the 800 cm⁻¹ cut, but it sits inside
the CO2 ν2 bending band whose hot-band/Fermi-resonance structure at 440 K cannot be modelled to
better than ~0.18 absorbance residual (≫ HCN's ~0.02 signal), so HCN floors to zero there too — HCN
is below detection via every accessible band. **NO2** is generated
line-by-line at its ν3 band (~1617 cm⁻¹): the downloaded HITRAN cross-section proved to be the
UV–visible NO2 spectrum (15,000–42,000 cm⁻¹), unusable in the mid-IR. NO2's band overlaps the H2O
bending fingerprint, so it is below detection on most burns (upper bound).

## 7. Fuel carbon fractions (F_C, mass C of dry fuel)
Crop residues 0.45 (Ma et al. 2018, herbaceous); straw/cowstraw/banana 0.44; hedgerow/hillside 0.47
(IPCC 2006 default); mahogany/wood 0.46 (Doraisami et al. 2024, tropical angiosperm 0.456).

## 8. Systematics NOT contained in the reported uncertainties
Optical thickness at peak smoke (CO2/H2O saturation; the down-weighting mitigates but does not
remove it); the departure of this retrieval from the Table D1 windowed method; cross-section
temperature extrapolation (≥125 K) for the VOCs; the elevated and variable pre-ignition background
(partial chamber flushing); adsorptive losses on the sampling train; principal-isotopologue-only
line lists for species other than CO2.

## 9. Performance (all 19 burns vs Andreae 2019: Agricultural-residue & Tropical-forest categories)
Species fall into three detectability tiers (geometric-mean EF ratio to Andreae across fuels):

- **Quantitative (within a factor of two).** CO2 1.01×, CO 0.93×, CH4 1.07×, CH2O 1.54×, HCOOH 0.97×,
  and **MCE** matches every fuel (crops 0.92–0.95, wood/tropical forest 0.90–0.92 vs Andreae 0.92 /
  0.906). These are the carbon-budget species (CO2+CO+CH4 carry >98 % of emitted carbon) plus two
  oxygenates with clean isolated bands. This is the reliable output of the retrieval.
- **Semi-quantitative (detected and ΔCO2-correlated, but biased).** C2H4 2.3× (≈1× in optically-thin
  conditions, inflating to several-× on the thickest crop burns); acetic acid (CH3COOH) 2.7× — the
  one heavy VOC with a genuine, if marginal, retrieval (excess-vs-CO2 R² up to 0.44); C2H2 ~7×; N2O
  approaches literature only in the thinnest conditions (linear-regime ER) and is over-retrieved on
  the optically-thick plumes; NH3 ≈ 0.5× — consistent across burns and biased *low*, as expected
  from ammonia's well-known adsorptive losses on the sampling train. Report with these caveats.
- **Below detection (upper bounds).** Acetone, isoprene, furan, ethanol, HONO, SO2, C2H6, HCN and the
  additional cross-section VOCs (acetonitrile, methyl formate, propene, propane, toluene, acetaldehyde)
  have **no ΔCO2-correlated excess** once the per-window continuum term (§4) absorbs the smoke-correlated
  pseudo-continuum: their retrieved excess is ≈ 0 (R² ≈ 0) or, where a weak cross-section forces an
  absurd concentration to fit residual structure, an obviously non-physical value (e.g. ethanol 80×,
  acetonitrile 10³×). Their bands lie under the saturated H2O bending fingerprint (1200–2200 cm⁻¹) and
  are below the residual floor at this optical depth. **NB:** this is the *corrected* picture. An
  earlier configuration that lacked a continuum term inside the VOC windows reported these species at
  25–70× literature; that over-retrieval was the VOC cross-sections fitting the window continuum, not
  a real signal — adding the continuum term (§4) removes it, and the honest result is non-detection.

The optical-depth dependence of the semi-quantitative species was established by comparing emission
ratios across the burns' optically-thin and optically-thick time steps.
