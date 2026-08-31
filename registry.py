"""
Species registry.

A single, explicit table mapping each target species to

  * the spectral fitting window(s) of Table D1 of the manuscript,
  * the spectroscopic databank its reference spectrum is generated from,
  * the name that databank knows it by,
  * molar mass and carbon number, for the carbon mass balance of Eq. 5.

WHY THE DATABANK COLUMN MATTERS
-------------------------------
``radis.calc_spectrum`` performs line-by-line synthesis and can only handle molecules
that have a *line list*. Eight of the species reported in Table 3 of the manuscript do
not: HNO2, CH3COOH, C2H6O (ethanol), C3H6O (acetone), C4H4O (furan), C5H8 (isoprene)
and CH3CHO (acetaldehyde) are heavy or floppy molecules whose room-temperature
Q-branches are unresolvable congestion rather than assignable lines, and are therefore
distributed as *measured composite absorption cross-sections* rather than line
parameters. CH2O is a line-by-line molecule but HITRAN indexes it as ``H2CO``.

In v1.0 every one of these raised inside ``calc_spectrum`` and was swallowed by a bare
``except``, leaving an all-zero reference column that the non-zero-column mask then
removed - so the species disappeared with no error. v1.1 routes them through the
cross-section pathway in :mod:`pyrospectra.xsections` and raises if the data are absent.

This is the same division of labour used by the established codes: MALT and Calcmet
both fit light molecules from line parameters and heavy VOCs from PNNL-style measured
cross-sections, and it is the arrangement behind the biomass-burning spectral library of
Johnson et al. (2010), which extended the PNNL quantitative library for exactly these
species.

The ``databank`` field takes one of:
  'hitran'  - line-by-line, fetched or read locally by RADIS
  'hitemp'  - line-by-line, HITEMP (hot bands; used for NO2 per Table D1)
  'geisa'   - line-by-line, GEISA 2020
  'xsec'    - measured cross-section, read by :mod:`pyrospectra.xsections`
"""

from copy import deepcopy

# ---------------------------------------------------------------------------
# Molar masses (g/mol) and carbon numbers, for Eqs. 5 and 6
# ---------------------------------------------------------------------------
MOLAR_MASS = {
    "CO2": 44.009, "CO": 28.010, "CH4": 16.043, "H2O": 18.015,
    "NO": 30.006, "NO2": 46.006, "N2O": 44.013, "NH3": 17.031,
    "HNO2": 47.013, "HNO3": 63.012, "HCN": 27.026, "SO2": 64.066,
    "HCl": 36.461, "COF2": 66.007, "HC3N": 51.047,
    "HCOOH": 46.025, "CH2O": 30.026, "CH3OH": 32.042, "CH3CHO": 44.053,
    "CH3COOH": 60.052, "C2H2": 26.038, "C2H4": 28.054, "C2H6": 30.069,
    "C2H6O": 46.069, "C3H6O": 58.080, "C4H4O": 68.075, "C5H8": 68.119,
}

CARBON_NUMBER = {
    "CO2": 1, "CO": 1, "CH4": 1, "H2O": 0,
    "NO": 0, "NO2": 0, "N2O": 0, "NH3": 0,
    "HNO2": 0, "HNO3": 0, "HCN": 1, "SO2": 0,
    "HCl": 0, "COF2": 1, "HC3N": 3,
    "HCOOH": 1, "CH2O": 1, "CH3OH": 1, "CH3CHO": 2,
    "CH3COOH": 2, "C2H2": 2, "C2H4": 2, "C2H6": 2,
    "C2H6O": 2, "C3H6O": 3, "C4H4O": 4, "C5H8": 5,
}

#: Fuel carbon mass fractions, Table 1 of the manuscript.
CARBON_FRACTIONS = {
    "boreal_peat": 0.44,   # Hu et al. (2018)
    "wheat": 0.46,         # Di Gruttola & Borello (2021); Fang et al. (2026)
    "rice": 0.41,
    "corn": 0.45,
    "rapeseed": 0.44,
}

# ---------------------------------------------------------------------------
# Species registry - windows are those of Table D1
# ---------------------------------------------------------------------------
# 'isotope' is passed straight to RADIS. v1.0 used '1' (principal isotopologue only)
# throughout; that is retained as the default so v1.1 does not silently change the
# line-by-line references. Note that for CO2 the principal isotopologue carries ~98.4%
# of terrestrial abundance, so a '1'-only reference is ~1.6% weak and biases retrieved
# CO2 correspondingly high. Because CO2 is the denominator of every emission ratio the
# effect largely divides out of the emission factors, but set isotope='1,2,3' if you
# want absolute CO2 right. Untested here (no network for line lists) - verify before use.
SPECIES_REGISTRY = {
    # --- line-by-line, HITRAN ------------------------------------------------
    "CO2":      {"databank": "hitran", "name": "CO2",   "isotope": "1",
                 "windows": [[3650, 3760]],
                 "note": "avoids the saturating nu3 band near 2350 cm-1"},
    "CO":       {"databank": "hitran", "name": "CO",    "isotope": "1",
                 "windows": [[2115, 2130]]},
    "CH4":      {"databank": "hitran", "name": "CH4",   "isotope": "1",
                 "windows": [[3111, 3125]]},
    "N2O":      {"databank": "hitran", "name": "N2O",   "isotope": "1",
                 "windows": [[2206, 2251]]},
    "NH3":      {"databank": "hitran", "name": "NH3",   "isotope": "1",
                 "windows": [[960, 975]]},
    "HCN":      {"databank": "hitran", "name": "HCN",   "isotope": "1",
                 "windows": [[3313, 3330]]},
    "SO2":      {"databank": "hitran", "name": "SO2",   "isotope": "1",
                 "windows": [[1350, 1362.5]]},
    "HCOOH":    {"databank": "hitran", "name": "HCOOH", "isotope": "1",
                 "windows": [[1116, 1130]],
                 "note": "Table D1 cites GEISA; HITRAN molecule 32 also carries it. "
                         "Set databank='geisa' to follow the manuscript exactly."},
    "CH2O":     {"databank": "hitran", "name": "H2CO",  "isotope": "1",
                 "windows": [[2770, 2810]],
                 "note": "HITRAN indexes formaldehyde as H2CO; v1.0 requested 'CH2O' "
                         "and silently produced a zero reference."},
    "C2H2":     {"databank": "hitran", "name": "C2H2",  "isotope": "1",
                 "windows": [[3250, 3305]]},
    "C2H4":     {"databank": "hitran", "name": "C2H4",  "isotope": "1",
                 "windows": [[940, 951]]},
    "H2O":      {"databank": "hitran", "name": "H2O",   "isotope": "1",
                 "windows": [[1200, 2200], [3400, 4000], [5000, 5650], [6600, 7600]]},

    # --- line-by-line, HITEMP (hot bands matter at 448 K cell temperature) ----
    "NO2":      {"databank": "hitemp", "name": "NO2",   "isotope": "1",
                 "windows": [[1618, 1630]],
                 "note": "Table D1 specifies HITEMP. Weakest-constrained window in the "
                         "set: sits inside the 1200-2200 cm-1 H2O envelope."},

    # --- measured cross-sections (no line list exists) ------------------------
    "HNO2":     {"databank": "xsec", "name": "HNO2", "windows": [[1250, 1368]],
                 "aliases": ["HONO", "Nitrous acid", "nitrous_acid"]},
    "CH3COOH":  {"databank": "xsec", "name": "CH3COOH", "windows": [[1170, 1200]],
                 "aliases": ["Acetic acid", "acetic_acid", "AceticAcid"]},
    "C2H6O":    {"databank": "xsec", "name": "C2H6O", "windows": [[1055, 1100]],
                 "aliases": ["Ethanol", "ethanol", "C2H5OH"]},
    "C3H6O":    {"databank": "xsec", "name": "C3H6O", "windows": [[1205, 1245]],
                 "aliases": ["Acetone", "acetone", "CH3COCH3"]},
    "C4H4O":    {"databank": "xsec", "name": "C4H4O", "windows": [[990, 1010]],
                 "aliases": ["Furan", "furan"]},
    "C5H8":     {"databank": "xsec", "name": "C5H8", "windows": [[885, 905]],
                 "aliases": ["Isoprene", "isoprene", "2-methyl-1,3-butadiene"]},
    "CH3CHO":   {"databank": "xsec", "name": "CH3CHO", "windows": [[1330, 1390]],
                 "aliases": ["Acetaldehyde", "acetaldehyde"],
                 "note": "Not listed in Table D1 although reported in Figs 8-9. Window "
                         "given here is the CH3 deformation region; CHECK AND ADJUST - "
                         "it overlaps the SO2 and HNO2 windows."},
}

#: Species whose reference spectrum requires a measured cross-section file.
XSEC_SPECIES = tuple(k for k, v in SPECIES_REGISTRY.items() if v["databank"] == "xsec")

#: Species that :func:`pyrospectra.species_selection.lasso_inversion` never drops.
CORE_SPECIES = ("CO2", "CO", "CH4", "N2O")


def get_registry(species=None, databank_overrides=None):
    """
    Return a deep copy of the registry, optionally restricted and overridden.

    Parameters
    ----------
    species : sequence of str, optional
        Restrict to these species. Default: all.
    databank_overrides : dict, optional
        e.g. ``{'HCOOH': 'geisa', 'NO2': 'hitran'}``.

    Returns
    -------
    dict
    """
    keys = list(SPECIES_REGISTRY) if species is None else list(species)
    unknown = [k for k in keys if k not in SPECIES_REGISTRY]
    if unknown:
        raise KeyError(
            f"Unknown species {unknown}. Known: {sorted(SPECIES_REGISTRY)}"
        )
    reg = {k: deepcopy(SPECIES_REGISTRY[k]) for k in keys}
    for key, bank in (databank_overrides or {}).items():
        if key not in reg:
            raise KeyError(f"Cannot override databank for absent species {key!r}")
        reg[key]["databank"] = bank
    return reg


def build_compounds(species=None, databank_overrides=None):
    """
    Build the ``compounds`` dictionary consumed by the reference generator.

    This replaces the hand-pickled ``compounds.pkl`` of v1.0 and keeps the windows,
    the databank and the databank-specific molecule name together, so that Table D1 is
    reproducible from the code rather than from a binary blob.

    Returns
    -------
    dict
        ``{species: {'bounds': [[wmin, wmax], ...], 'databank': ..., 'name': ...}}``
    """
    reg = get_registry(species, databank_overrides)
    return {
        k: {
            "bounds": [list(map(float, w)) for w in v["windows"]],
            "databank": v["databank"],
            "name": v.get("name", k),
            "isotope": v.get("isotope", "1"),
            "aliases": v.get("aliases", []),
            "note": v.get("note", ""),
        }
        for k, v in reg.items()
    }


def carbon_number(species):
    """Carbon atoms per molecule; raises for unregistered species."""
    try:
        return CARBON_NUMBER[species]
    except KeyError as exc:
        raise KeyError(
            f"No carbon number registered for {species!r}. Add it to "
            "registry.CARBON_NUMBER before computing a carbon mass balance."
        ) from exc


def molar_mass(species):
    """Molar mass in g/mol; raises for unregistered species."""
    try:
        return MOLAR_MASS[species]
    except KeyError as exc:
        raise KeyError(
            f"No molar mass registered for {species!r}. Add it to "
            "registry.MOLAR_MASS before computing an emission factor."
        ) from exc


__all__ = [
    "SPECIES_REGISTRY", "XSEC_SPECIES", "CORE_SPECIES", "MOLAR_MASS",
    "CARBON_NUMBER", "CARBON_FRACTIONS", "get_registry", "build_compounds",
    "carbon_number", "molar_mass",
]
