# Reference spectroscopic data

PyroSpectra does not redistribute spectroscopic data. This file records what each
species needs, where it comes from, and how to point the package at it.

**No cross-section data are bundled with this release, and none have been synthesised.**
The files under `tests/` generate a Gaussian band with a known integral in the HITRAN
and PNNL formats purely to exercise the parsers and the unit conversions. They are
labelled synthetic in the test module and must never be used as reference data.

---

## 1. Why two pathways

`radis.calc_spectrum` performs line-by-line synthesis and needs a **line list**. Light
molecules have one. Heavy or floppy molecules — acetone, furan, isoprene, ethanol,
acetaldehyde, acetic acid, nitrous acid — do not: at 1 bar their room-temperature
spectra are unresolved band contours rather than assignable transitions, and the
databases distribute them as **measured composite absorption cross-sections**.

This is the same split the established codes use. MALT and Calcmet both fit light
molecules from line parameters and heavy VOCs from PNNL-style cross-sections, and it is
the arrangement behind the biomass-burning library of Johnson et al. (2010), which
extended the PNNL quantitative library with exactly these species for fire emissions.

In v1.0 all eight of the affected species raised inside `calc_spectrum`, were swallowed
by a bare `except`, and left an all-zero reference row that the non-zero-column mask
then removed — so they vanished from the retrieval with no error reported. v1.1 routes
them through `pyrospectra.xsections` and raises if the data are absent.

---

## 2. What each species needs

| Species | Name | Table D1 window (cm⁻¹) | Pathway | Source |
|---|---|---|---|---|
| CO₂ | carbon dioxide | 3650–3760 | line-by-line | HITRAN |
| CO | carbon monoxide | 2115–2130 | line-by-line | HITRAN |
| CH₄ | methane | 3111–3125 | line-by-line | HITRAN |
| N₂O | nitrous oxide | 2206–2251 | line-by-line | HITRAN |
| NH₃ | ammonia | 960–975 | line-by-line | HITRAN |
| HCN | hydrogen cyanide | 3313–3330 | line-by-line | HITRAN |
| SO₂ | sulphur dioxide | 1350–1362.5 | line-by-line | HITRAN |
| HCOOH | formic acid | 1116–1130 | line-by-line | HITRAN (mol. 32) or GEISA |
| CH₂O | formaldehyde | 2770–2810 | line-by-line | HITRAN, **as `H2CO`** |
| C₂H₂ | acetylene | 3250–3305 | line-by-line | HITRAN |
| C₂H₄ | ethylene | 940–951 | line-by-line | HITRAN |
| H₂O | water vapour | 1200–2200; 3400–4000; 5000–5650; 6600–7600 | line-by-line | HITRAN |
| NO₂ | nitrogen dioxide | 1618–1630 | line-by-line | **HITEMP** (per Table D1) |
| HNO₂ | nitrous acid | 1250–1368 | **cross-section** | PNNL / HITRAN-xsc |
| CH₃COOH | acetic acid | 1170–1200 | **cross-section** | PNNL / HITRAN-xsc |
| C₂H₆O | ethanol | 1055–1100 | **cross-section** | PNNL / HITRAN-xsc |
| C₃H₆O | acetone | 1205–1245 | **cross-section** | PNNL / HITRAN-xsc |
| C₄H₄O | furan | 990–1010 | **cross-section** | PNNL / HITRAN-xsc |
| C₅H₈ | isoprene | 885–905 | **cross-section** | PNNL / HITRAN-xsc |
| CH₃CHO | acetaldehyde | *not in Table D1* | **cross-section** | PNNL / HITRAN-xsc |

Three notes on the table.

* **CH₂O was a naming problem, not a missing-data problem.** HITRAN indexes
  formaldehyde as `H2CO`. v1.0 asked for `CH2O` and got a zero row. The registry now
  carries the databank name separately from the reporting name.
* **Table D1 attributes HCOOH, C₂H₆O, C₃H₆O, C₄H₄O and C₅H₈ to GEISA.** GEISA 2020 has
  a line list for formic acid, but not for ethanol, acetone, furan or isoprene — those
  four are cross-section species in GEISA as in HITRAN. Worth checking against your
  own build before the manuscript goes further.
* **CH₃CHO is reported in Figures 8 and 9 but absent from Table D1.** The window in the
  registry is a placeholder over the CH₃ deformation region and overlaps the SO₂ and
  HNO₂ windows. Set it deliberately before using it.

---

## 3. Obtaining the data

### Line lists (RADIS)

On a networked machine RADIS fetches and caches these itself:

```python
generate_reference(..., emission_species=build_compounds())   # databank='hitran'/'hitemp'
```

To run offline, download the line lists once and point the registry at the local files.
RADIS accepts a path in place of a databank name:

```python
compounds = build_compounds()
compounds['NO2']['databank'] = '/data/hitemp/NO2.par'
```

### Cross-sections

`fetch_reference_data.py` in `examples/` attempts the HITRAN cross-section index and
reports what it finds. PNNL requires accepting its terms of use through the
Northwest-Infrared portal, so it cannot be scripted — download those manually.

Either format works:

* **HITRAN `.xsc`** — fixed-width 100-character header, values in cm² molecule⁻¹. One
  file may hold several (T, P) bands; all are read and used for state selection.
* **PNNL two-column ASCII** — wavenumber and decadic absorbance per ppm per metre at
  296 K, 1 atm. Headers and comment lines are skipped. If your copy is already in
  cm² molecule⁻¹, pass `units='cm2/molecule'`.

Then:

```python
xsec_paths = {
    'C3H6O':    'data/xsec/acetone.xsc',
    'C5H8':     'data/xsec/isoprene.xsc',
    'C4H4O':    'data/xsec/furan.xsc',
    'C2H6O':    'data/xsec/ethanol.txt',
    'CH3COOH':  'data/xsec/acetic_acid.txt',
    'HNO2':     'data/xsec/nitrous_acid.txt',
    'CH3CHO':   'data/xsec/acetaldehyde.txt',
}
ref, full, mask, prov = generate_reference(out, compounds, w, P, T, sigma=0.5,
                                           xsec_paths=xsec_paths)
```

Every file's SHA-256 is recorded in `reference_information/reference_provenance.json`,
so a retrieval can be traced back to the exact data that produced it.

---

## 4. The temperature problem — read this before quoting these EFs

The MG5 cell runs at **448 K**. The cross-section libraries are measured near ambient:
PNNL at 278, 298 and 323 K, HITRAN-xsc typically over a similar range. Every
cross-section species is therefore **extrapolated by 125 K or more**, well outside the
measured range.

The package handles this as follows, and records what it did:

* the band **contour** is taken from the nearest measured temperature;
* where two or more temperatures exist, the integrated band **strength** is
  extrapolated linearly in T and the contour renormalised to it, because integrated
  intensity behaves far better under extrapolation than the rotational envelope;
* a rescaling outside 0.25–4× is refused rather than applied;
* extrapolations beyond 50 K raise a `RuntimeWarning` naming the species.

What this does **not** correct is hot-band population. At 448 K, vibrationally excited
states are appreciably populated and the band contour genuinely broadens and shifts —
an effect no amount of rescaling reproduces. **Treat emission factors for the
cross-section species as carrying a systematic uncertainty that the posterior
covariance does not contain**, and say so in the manuscript. The retrieval uncertainty
reported for HNO₂, CH₃COOH, C₂H₆O, C₃H₆O, C₄H₄O, C₅H₈ and CH₃CHO is a precision, not an
accuracy.

Two ways to do better, both worth considering:

1. **Measure them.** You have a heated cell and calibration gas capability. Cross-sections
   for even two or three of these species at 448 K would remove the largest systematic
   on those EFs outright, and would be publishable in its own right.
2. **Bracket it.** Run the retrieval at the nearest measured temperature and at the
   extrapolated one and report the spread as a systematic term. Cheap, and honest.

---

## 5. Citing the data

Cite whichever you use, alongside the package:

* HITRAN — Gordon et al., *JQSRT* (current edition)
* HITRAN cross-sections — Kochanov et al., *JQSRT* 177 (2016)
* HITEMP — Rothman et al., *JQSRT* 111 (2010)
* GEISA — Delahaye et al., *JQSRT* (2021)
* PNNL — Sharpe et al., *Applied Spectroscopy* 58 (2004)
* Biomass-burning IR library — Johnson et al., *Vibrational Spectroscopy* 53 (2010)
* RADIS — Pannier & Laux, *JQSRT* 222 (2019)
