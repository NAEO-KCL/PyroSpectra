"""
Optically-thick, high-concentration retrieval for close-to-source biomass-burning smoke
(PyroSpectra v2.0).

This module extends the core PyroSpectra retrieval (:mod:`pyrospectra.inversion`) for the
optically-thick regime encountered when sampling *undiluted, close-to-source* smoke, where peak
excess CO2 reaches 9,000-14,000 ppm, the H2O bands saturate (decadic absorbance |A| ~ 6-7), and
strong absorbers bleed into the narrow trace windows.  It departs from the single-window classical
least squares / Tikhonov retrieval in several respects, each motivated by the high optical depth:

1. **CO2 isotopologues** (``isotope='1,2,3'``) so that 13-CO2 and hot-band absorption in the N2O
   and CO windows is modelled rather than misattributed to those trace species.
2. **Pervasive absorbers CO2, H2O and CO modelled full-spectrum** (their references are non-zero in
   every fitting window): the column of each is pinned by its reliable unsaturated bands and its
   absorption is subtracted inside the crowded trace windows where its own cores are saturated.
3. **Per-window, non-negative continuum pseudo-species** in every trace window, absorbing the
   smoke-correlated broadband continuum (aerosol scattering + pseudo-continuum + imperfect I0
   between strong bands).  Without it a window's species fits that continuum and returns an inflated
   emission factor that still correlates with CO2 (a high-R2 false detection).  Excluded from the
   N2O window, whose over-retrieval is CO2/CO interference (handled by the thin-regime ratio) rather
   than continuum.
4. **Non-negative gas concentrations** via bounded per-time-step least squares.
5. **Optical-depth channel weighting**: channels whose 90th-percentile |A| exceeds a linearity cap
   are down-weighted, since Beer-Lambert linearity fails in saturated line cores.
6. **Slope-based emission ratios** (excess-X vs excess-CO2 through the origin) over the plume, with
   an **optically-thin / linear-regime** ratio for interference-limited species such as N2O.
7. **Per species-burn detection test**: the emission-ratio slope must exceed 3x its standard error
   *and* pass a carbon-plausibility check (a minor product's carbon cannot exceed CO's); otherwise a
   3-sigma detection-limit / indicative estimate is reported instead of a value.

The temporal-regularisation retrieval of :func:`pyrospectra.temporally_regularised_inversion` and
this optically-thick retrieval are complementary: the former stabilises the concentration *time
series*; the latter isolates weak trace species from the *dominant absorbers* at high optical depth.

Paths (the cross-section directory and the reference cache) are configured through
:data:`PATHS` or :func:`configure`; nothing is hard-coded.

References
----------
Richardson-Foulger, Wooster, Gomez-Dans & Grosvenor (2026), JGR: Biogeosciences.
"""
from __future__ import annotations
import os, glob as _glob, re as _re, hashlib, pickle as _pkl
import numpy as np

from .registry import MOLAR_MASS as _MM0, CARBON_NUMBER as _CN0
from .preprocessing import estimate_background
from .xsections import (load_cross_section, select_state,
                        apply_instrument_lineshape, band_to_absorbance, CrossSectionBand)

__all__ = ["PATHS", "configure", "CONFIG", "CFG", "MOLAR_MASS", "CARBON_NUMBER",
           "build_reference", "retrieve", "emission_ratios", "detected",
           "detection_limit_ok", "carbon", "n_preignition"]

# --------------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------------
PATHS = {
    # directory of HITRAN .xsc cross-section files for the heavy-VOC species (required only if such
    # species are retrieved); if None, defaults to ``<cwd>/data/xsec`` at first use.
    "xsec_dir": None,
    # on-disk cache of line-by-line reference pieces (keyed by species, window, T, P, sigma, isotope).
    # If None, defaults to ``<cwd>/pyrospectra_refcache``.
    "refcache": None,
}

class _Config:
    """Retrieval hyper-parameters (all fixed a priori; none tuned to literature)."""
    sigma_cm1   = 0.5     # Gaussian instrument lineshape sigma (see docs: measured effective FWHM
                          #   ~0.8-0.95 cm-1 => sigma~0.40; 0.50 changes EFs <2% and is retained).
    widen_cm1   = 0.0     # symmetric widening added to each target window (0 = Table-D1 windows).
    od_cap      = 2.0     # |A| linearity cap: channels with 90th-percentile |A| above this are
    od_weight   = 0.1     #   down-weighted to this factor (saturated line cores).
    lincap_ppm  = 3000.0  # excess-CO2 cap defining the optically-thin / linear regime.
    bscale      = 0.05    # amplitude of the per-window continuum basis column.
    path_cm     = 500.0   # optical path length of the MG5 cell (cm).
    plume_dco2  = 200.0   # excess-CO2 threshold (ppm) defining a plume time step.
    snr_detect  = 3.0     # emission-ratio slope must exceed this multiple of its standard error.

CONFIG = _Config()

def configure(xsec_dir=None, refcache=None, **hyper):
    """Set the cross-section directory, reference cache and/or any :class:`_Config` hyper-parameter."""
    if xsec_dir is not None:
        PATHS["xsec_dir"] = xsec_dir
    if refcache is not None:
        PATHS["refcache"] = refcache
        os.makedirs(refcache, exist_ok=True)
    for k, v in hyper.items():
        if not hasattr(CONFIG, k):
            raise AttributeError("unknown hyper-parameter %r" % k)
        setattr(CONFIG, k, v)

def _xsec_dir():
    d = PATHS["xsec_dir"] or os.path.join(os.getcwd(), "data", "xsec")
    return d

def _refcache():
    d = PATHS["refcache"] or os.path.join(os.getcwd(), "pyrospectra_refcache")
    os.makedirs(d, exist_ok=True)
    return d

# --------------------------------------------------------------------------------------------------
# Molar masses / carbon numbers (core registry + the additional cross-section species)
# --------------------------------------------------------------------------------------------------
MOLAR_MASS = dict(_MM0)
MOLAR_MASS.update({"CH3CN": 41.053, "HCO2CH3": 60.052, "C3H6": 42.081, "C3H8": 44.097,
                   "C6H5CH3": 92.141, "NO2": 46.006})
CARBON_NUMBER = dict(_CN0)
CARBON_NUMBER.update({"CH3CN": 2, "HCO2CH3": 2, "C3H6": 3, "C3H8": 3, "C6H5CH3": 7, "NO2": 0})

# --------------------------------------------------------------------------------------------------
# Species configuration:  name -> (radis_name, isotope, [ [lo, hi], ... ], role)
#   role "perv"  : pervasive absorber, modelled full-spectrum (over the merged fitted region)
#   role "targ"  : line-by-line species fit in its own window(s)
#   role "xsec"  : heavy VOC retrieved from a measured HITRAN cross-section (.xsc)
# Windows follow Table D1 where applicable; the corrections documented in v2.0 are noted inline.
# --------------------------------------------------------------------------------------------------
CFG = {
    "CO2":  ("CO2", "1,2,3", [[3650, 3760]], "perv"),
    "H2O":  ("H2O", "1", [[1200, 2200], [3400, 4000], [5000, 5650], [6600, 7600]], "perv"),
    "CO":   ("CO", "1,2,3", [[2115, 2130]], "perv"),
    "CH4":  ("CH4", "1", [[3111, 3125]], "targ"),
    "N2O":  ("N2O", "1", [[2200, 2235]], "targ"),
    "NH3":  ("NH3", "1", [[960, 975]], "targ"),
    "HCN":  ("HCN", "1", [[3306, 3325]], "targ"),   # nu1 Q-branch 3311.5 (old 3313-3330 missed it)
    "SO2":  ("SO2", "1", [[1350, 1362.5]], "targ"),
    "NO2":  ("NO2", "1", [[1600, 1640]], "targ"),   # nu3 ~1617 via line-by-line (the UV-Vis .xsc is
                                                    #   unusable in the IR); overlaps the H2O bend
    "HCOOH": ("HCOOH", "1", [[1116, 1130]], "targ"),
    "CH2O": ("H2CO", "1", [[2770, 2810]], "targ"),
    "C2H2": ("C2H2", "1", [[3250, 3305]], "targ"),
    "C2H4": ("C2H4", "1", [[940, 951]], "targ"),
    "C2H6": ("C2H6", "1", [[2975, 2995]], "targ"),
    # ---- heavy VOCs via measured HITRAN cross-sections ----
    "C3H6O":   ("acetone", "xsec", [[1205, 1245]], "xsec"),
    "C5H8":    ("isoprene", "xsec", [[885, 905]], "xsec"),
    "C4H4O":   ("furan", "xsec", [[990, 1010]], "xsec"),
    "C2H6O":   ("ethanol", "xsec", [[1055, 1100]], "xsec"),
    "CH3COOH": ("aceticacid", "xsec", [[1170, 1200]], "xsec"),
    "HNO2":    ("HNO2", "xsec", [[1250, 1368]], "xsec"),
    "CH3CN":   ("acetonitrile", "xsec", [[1035, 1050]], "xsec"),  # nu4 CH3-rock (old 2255-2285 sat in
                                                                  #   the CO2 nu3 wing -> fitted CO2)
    "C3H6":    ("propene", "xsec", [[906, 924]], "xsec"),
    "C6H5CH3": ("toluene", "xsec", [[1025, 1045]], "xsec"),
    "CH3CHO":  ("acetaldehyde", "xsec", [[1400, 1440]], "xsec"),
    "HCO2CH3": ("methylformate", "xsec", [[1150, 1172]], "xsec"),
    "C3H8":    ("propane", "xsec", [[1465, 1480]], "xsec"),       # off the CH4/H2O forest
}

# filename prefix for each cross-section species (the .xsc header molecule is verified on load)
XSEC_PREFIX = {"C3H6O": "CH3COCH3", "C5H8": "C5H8", "C4H4O": "C4H4O", "C2H6O": "C2H6O",
               "CH3COOH": "C2-H4-O2", "HNO2": "HNO2", "CH3CN": "CH3CN", "C3H6": "C3H6",
               "C6H5CH3": "C6H5CH3", "CH3CHO": "CH3CHO", "HCO2CH3": "HCO2CH3", "C3H8": "C3H8"}

PERVASIVE = {"CO2", "H2O", "CO"}
NO_BASELINE = {"N2O"}   # windows that do NOT receive a continuum term (see module docstring)

def baseline_windows(config=CFG):
    """Trace windows that receive a per-window continuum term (all targ+xsec except N2O)."""
    return [[float(w[0]), float(w[1])]
            for s, (rn, iso, wl, role) in config.items()
            if role in ("targ", "xsec") and s not in NO_BASELINE for w in wl]

# --------------------------------------------------------------------------------------------------
# Cross-section loading
# --------------------------------------------------------------------------------------------------
_BAND_CACHE = {}

def _load_bands(name, lo, hi):
    """Load and cache the measured cross-section bands covering ``[lo, hi]`` for one species."""
    ck = os.path.join(_refcache(), "xsecbands_%s.pkl" % name)
    if os.path.exists(ck):
        try:
            return [CrossSectionBand(w, v, Tk, Pb, u, molecule=m, resolution_cm1=r)
                    for (w, v, Tk, Pb, u, m, r) in _pkl.load(open(ck, "rb"))]
        except Exception:
            pass
    cands = []
    for f in _glob.glob(os.path.join(_xsec_dir(), "%s_*.xsc" % XSEC_PREFIX[name])):
        b = os.path.basename(f)
        if "_Ar_" in b:                        # skip Ar-broadened combustion sets
            continue
        tt = _re.search(r'_(\d{2,4}\.?\d*)K?[-_]', b)
        Tf = float(tt.group(1)) if tt else 300.0
        rg = _re.search(r'(\d+\.\d+)-(\d+\.\d+)', b)
        if rg and not (float(rg.group(1)) <= lo and float(rg.group(2)) >= hi):
            continue                            # file must cover the window
        if Tf > 360:
            continue                            # atmospheric-temperature measurements only
        span = (float(rg.group(2)) - float(rg.group(1))) if rg else 0
        cands.append((span, Tf, f))
    cands.sort(reverse=True)                     # widest coverage first
    bands = []
    for span, Tf, f in cands[:8]:               # cap for speed; keeps broadband + a temperature spread
        try:
            bs = load_cross_section(f)
        except Exception:
            continue
        bands.extend(b for b in bs if b.temperature_k <= 360)
    try:
        _pkl.dump([(b.wavenumber, b.values, b.temperature_k, b.pressure_bar, b.units,
                    b.molecule, b.resolution_cm1) for b in bands], open(ck, "wb"))
    except Exception:
        pass
    return bands

def _xsec_row(name, wv, P, T, config=CFG):
    lo, hi = config[name][2][0]
    if name not in _BAND_CACHE:
        _BAND_CACHE[name] = _load_bands(name, lo, hi)
    bands = _BAND_CACHE[name]
    if not bands:
        raise RuntimeError("no usable .xsc for %s (prefix %s, window %g-%g)"
                           % (name, XSEC_PREFIX[name], lo, hi))
    band, _ = select_state(bands, T, P, window=[lo, hi], strength_extrapolation=True)
    band, _ = apply_instrument_lineshape(band, CONFIG.sigma_cm1)
    ab = band_to_absorbance(band, P, T, path_length_cm=CONFIG.path_cm, mole_fraction=1e-6)
    out = np.interp(wv, band.wavenumber, ab, left=0, right=0)
    out[(wv < lo) | (wv > hi)] = 0.0
    return np.nan_to_num(out)

# --------------------------------------------------------------------------------------------------
# Pre-ignition detection and reference matrix
# --------------------------------------------------------------------------------------------------
def n_preignition(spec, wv):
    """Number of stabilised pre-ignition scans and the ignition index, from a CO2-band metric."""
    I0 = np.median(spec[:8], axis=0)
    band = (wv >= 3650) & (wv <= 3760)
    with np.errstate(divide="ignore", invalid="ignore"):
        A = -np.log10(np.where(spec <= 0, np.nan, spec) / np.where(I0 <= 0, np.nan, I0))
    m = np.nanmean(np.abs(A[:, band]), axis=1)
    base = np.median(m[:15]); rsig = 1.4826 * np.median(np.abs(m[:15] - base)) + 1e-12
    ign = None
    for i in range(3, len(m) - 3):
        if all(m[i + k] > base + 8 * rsig for k in range(3)):
            ign = i; break
    if ign is None:
        ign = int(np.argmax(m))
    npre = 1
    for i in range(1, ign):
        if abs(m[i] - m[i - 1]) <= 6 * rsig:
            npre = i + 1
        else:
            break
    return max(5, min(npre, ign - 2)), ign

def _lbl_window(*a, **k):
    from .reference import _lbl_window as f
    return f(*a, **k)

def build_reference(wv, P, T, config=CFG):
    """Build the reference matrix over the merged fitted region.

    Returns ``(names, full, mask, merged, target_windows)`` where ``full`` is ``(N_species, N_k)``
    with pervasive absorbers non-zero across the whole fitted region and every other species
    non-zero only inside its own window(s).
    """
    from .reference import ReferenceGenerationError
    W = CONFIG.widen_cm1
    tw = {}
    for s, (rn, iso, wl, role) in config.items():
        tw[s] = ([[max(wv.min(), lo - W), min(wv.max(), hi + W)] for lo, hi in wl]
                 if role == "targ" else [list(map(float, w)) for w in wl])
    allw = sorted(w for s in config for w in tw[s])
    merged = [list(allw[0])]
    for lo, hi in allw[1:]:
        if lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])

    def ck(n, lo, hi, iso):
        key = "%s|%.3f|%.3f|%.4f|%.5f|%.3f|%s" % (n, lo, hi, T, P, CONFIG.sigma_cm1, iso)
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    names = list(config)
    full = np.zeros((len(names), wv.size))
    for i, name in enumerate(names):
        rn, iso, wl, role = config[name]
        if role == "xsec":
            try:
                full[i] = _xsec_row(name, wv, P, T, config)
            except Exception as e:
                print("  cross-section %s failed: %s" % (name, e))
            continue
        row = np.zeros_like(wv)
        for lo, hi in (merged if role == "perv" else tw[name]):
            cf = os.path.join(_refcache(), "%s_%s.npy" % (name, ck(name, lo, hi, iso)))
            piece = None
            if os.path.exists(cf):
                try:
                    p = np.load(cf); piece = p if p.shape == wv.shape else None
                except Exception:
                    piece = None
            if piece is None:
                try:
                    piece, _ = _lbl_window(rn, [lo, hi], T, P, wv, CONFIG.sigma_cm1, "hitran", isotope=iso)
                except ReferenceGenerationError:
                    piece = np.zeros_like(wv)
                except Exception:
                    piece = np.zeros_like(wv)
                np.save(cf, piece)
            row = row + piece
        full[i] = np.nan_to_num(row)
    mask = np.zeros(wv.size, bool)
    for lo, hi in merged:
        mask |= (wv >= lo) & (wv <= hi)
    return names, full, mask, merged, tw

# --------------------------------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------------------------------
def retrieve(spec, wv, P, T, fuel_cf, config=CFG, npre=None, return_fit=False):
    """Retrieve concentrations, emission ratios, factors and detection statistics for one burn.

    Parameters
    ----------
    spec : (N_t, N_k) single-beam spectra;  wv : (N_k,) wavenumber grid (cm^-1)
    P, T : gas-cell pressure (bar) and temperature (K)
    fuel_cf : fuel carbon mass fraction F_C for the carbon mass-balance EF (Eq. 6)
    config : species configuration (defaults to :data:`CFG`)
    npre : number of pre-ignition scans (auto-detected if None)

    Returns a dict with per-species EF / EF_thin, ER / ER_thin and their standard errors, R2,
    3-sigma detection-limit EFs, SNR, MCE, total carbon, the retrieved concentration array and the
    species order.  See the module docstring for the method.
    """
    from scipy.optimize import lsq_linear
    if npre is None:
        npre, _ = n_preignition(spec, wv)
    names, full, mask, merged, tw = build_reference(wv, P, T, config)
    wvf = wv[mask]; gas = full[:, mask]

    baseline = estimate_background(spec, n_preignition=npre)
    with np.errstate(divide="ignore", invalid="ignore"):
        obs = np.nan_to_num(-np.log10(np.where(spec <= 0, np.nan, spec)
                                      / np.where(baseline <= 0, np.nan, baseline)))[:, mask]

    # per-window non-negative continuum columns
    bl = []
    for lo, hi in baseline_windows(config):
        w = (wvf >= lo) & (wvf <= hi)
        if w.sum() < 4:
            continue
        c = np.zeros(wvf.size); c[w] = CONFIG.bscale; bl.append(c)
    bl = np.array(bl) if bl else np.zeros((0, wvf.size))
    R = np.vstack([gas, bl]); allnames = names + ["BL%d" % k for k in range(len(bl))]

    # optical-depth channel weights
    q90 = np.percentile(np.abs(obs), 90, axis=0)
    sw = np.sqrt(np.where(q90 <= CONFIG.od_cap, 1.0, CONFIG.od_weight))
    Aw = (R * sw[None, :]).T
    lb = np.zeros(len(allnames)); ub = np.full(len(allnames), np.inf)   # gas + continuum non-negative
    conc = np.zeros((len(allnames), obs.shape[0]))
    for tt in range(obs.shape[0]):
        conc[:, tt] = lsq_linear(Aw, obs[tt] * sw, bounds=(lb, ub), method="bvls",
                                 max_iter=150, tol=1e-8).x

    gi = {s: allnames.index(s) for s in names}
    bg = np.median(conc[:, :npre], axis=1, keepdims=True)
    exc = conc - bg; dco2 = exc[gi["CO2"]]
    plume = dco2 > CONFIG.plume_dco2
    thin = plume & (dco2 <= CONFIG.lincap_ppm)

    ER, ERt, ER_se, ERt_se, R2 = emission_ratios(exc, dco2, gi, names, plume, thin)
    CT  = 1.0 + sum(CARBON_NUMBER.get(s, 0) * ER.get(s, 0)  for s in ("CO", "CH4") if _f(ER.get(s)))
    CTt = 1.0 + sum(CARBON_NUMBER.get(s, 0) * ERt.get(s, 0) for s in ("CO", "CH4") if _f(ERt.get(s)))

    def ef(er, ct, s):
        v = er.get(s, float("nan"))
        return (fuel_cf * 1000 * (MOLAR_MASS[s] / 12.0) * v / ct) if (s in MOLAR_MASS and _f(ct) and _f(v)) else float("nan")
    def efv(x, ct, s):
        return (fuel_cf * 1000 * (MOLAR_MASS[s] / 12.0) * x / ct) if (s in MOLAR_MASS and _f(ct) and _f(x)) else float("nan")

    EF   = {s: ef(ER, CT, s)   for s in ER}
    EFt  = {s: ef(ERt, CTt, s) for s in ERt}
    EF_DL  = {s: efv(3.0 * ER_se[s], CT, s)  for s in ER_se}
    EFt_DL = {s: efv(3.0 * ERt_se[s], CTt, s) for s in ERt_se}
    SNR  = {s: (abs(ER[s]) / ER_se[s]  if _pos(ER_se.get(s)) else float("nan")) for s in ER}
    SNRt = {s: (abs(ERt[s]) / ERt_se[s] if _pos(ERt_se.get(s)) else float("nan")) for s in ERt}
    MCE = 1.0 / (1.0 + ER.get("CO", 0)) if _f(ER.get("CO")) else float("nan")

    if return_fit:   # fit internals for the diagnostics module (spectral atlas, residuals, ...)
        self_fit = {"R": R, "obs": obs, "wvf": wvf, "allnames": allnames, "conc_all": conc,
                    "gi": gi, "exc": exc, "dco2": dco2, "plume": plume, "thin": thin, "sw": sw}
    return {"T_K": T, "P_bar": P, "npre": int(npre), "fuel_cf": fuel_cf, "MCE": MCE, "CT": CT,
            "EF": EF, "EF_thin": EFt, "ER": ER, "ER_thin": ERt, "ER_se": ER_se, "ER_thin_se": ERt_se,
            "EF_DL": EF_DL, "EF_thin_DL": EFt_DL, "SNR": SNR, "SNR_thin": SNRt, "R2": R2,
            "n_plume": int(plume.sum()), "n_thin": int(thin.sum()), "lincap": CONFIG.lincap_ppm,
            "peak_excess_CO2": float(np.nanmax(dco2)) if dco2.size else float("nan"),
            "species": names, "concentrations": conc[[gi[s] for s in names]],
            **(self_fit if return_fit else {})}

def emission_ratios(exc, dco2, gi, names, plume, thin):
    """Slope of excess-X vs excess-CO2 (through the origin) over the plume and thin subsets.

    Returns ``(ER, ER_thin, ER_se, ER_thin_se, R2)``.  The standard error propagates the fit-residual
    scatter and yields the detection limit; a negative slope (anti-correlation) is a non-detection.
    """
    def slope(sel, s):
        x = exc[gi[s]][sel]; d = dco2[sel]
        ok = np.isfinite(x) & np.isfinite(d) & (d > 0); n = int(ok.sum())
        if n <= 2:
            return float("nan"), float("nan"), float("nan")
        sl = float(np.sum(x[ok] * d[ok]) / np.sum(d[ok] ** 2))
        res = x[ok] - sl * d[ok]
        ss = float(np.sum((x[ok] - x[ok].mean()) ** 2))
        r2 = float(1 - np.sum(res ** 2) / ss) if ss > 0 else float("nan")
        se = float(np.sqrt(np.sum(res ** 2) / max(n - 1, 1)) / np.sqrt(np.sum(d[ok] ** 2)))
        return sl, r2, se
    ER, ERt, ER_se, ERt_se, R2 = {}, {}, {}, {}, {}
    for s in names:
        if s == "H2O":
            continue
        ER[s], R2[s], ER_se[s] = slope(plume, s)
        ERt[s], _, ERt_se[s] = slope(thin, s)
    return ER, ERt, ER_se, ERt_se, R2

# --------------------------------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------------------------------
def carbon(ef, s):
    """Relative emitted carbon of species ``s`` at emission factor ``ef`` (common factors dropped)."""
    return ef * CARBON_NUMBER.get(s, 0) / MOLAR_MASS[s] if (s in MOLAR_MASS and _f(ef)) else 0.0

def detected(rec, s, thin_species=("N2O",), min_plume=30, min_r2=0.25, snr_override=10.0):
    """Whether species ``s`` is detected on the burn summarised by ``rec`` (a :func:`retrieve` dict).

    Detected iff: real plume (>= ``min_plume`` steps); positive EF; slope SNR >= ``CONFIG.snr_detect``
    with (R2 >= ``min_r2`` OR [SNR >= ``snr_override`` AND R2 > 0]); and the species' emitted carbon
    does not exceed CO's (a physical bound that rejects interferent-residual artefacts).  CO2 is the
    reference species and is always detected.
    """
    if rec.get("n_plume", 0) < min_plume:
        return False
    thin = s in thin_species
    v = (rec.get("EF_thin", {}) if thin else rec.get("EF", {})).get(s)
    if s == "CO2":
        return _pos(v)
    if not _pos(v):
        return False
    snr = (rec.get("SNR_thin", {}) if thin else rec.get("SNR", {})).get(s)
    r2 = (rec.get("R2", {}) or {}).get(s)
    if not (snr is not None and np.isfinite(snr) and snr >= CONFIG.snr_detect):
        return False
    r2ok = r2 is not None and np.isfinite(r2)
    if not ((r2ok and r2 >= min_r2) or (snr >= snr_override and r2ok and r2 > 0)):
        return False
    co = rec.get("EF", {}).get("CO")
    if _pos(co) and s not in ("CO2", "CO", "CH4") and carbon(v, s) > carbon(co, "CO"):
        return False
    return True

def detection_limit_ok(rec, s, thin_species=("N2O",), ceiling=10.0):
    """True if the 3-sigma detection limit is in a usable range (a bound can be reported)."""
    dl = (rec.get("EF_thin_DL", {}) if s in thin_species else rec.get("EF_DL", {})).get(s)
    return dl is not None and np.isfinite(dl) and 1e-3 <= dl <= ceiling

def _f(x):   return x is not None and x == x                       # finite / not-None / not-NaN
def _pos(x): return x is not None and np.isfinite(x) and x > 0

# Public, package-qualified aliases (distinct from the core single-window retrieval names).
retrieve_thick = retrieve
thick_reference = build_reference
thick_emission_ratios = emission_ratios
is_detected = detected
__all__ += ["retrieve_thick", "thick_reference", "thick_emission_ratios", "is_detected"]
