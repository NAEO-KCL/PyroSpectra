"""
Locate the cross-section data PyroSpectra needs, and report what is missing.

Run this on a machine with network access. It checks a local data directory first, then
queries the HITRAN cross-section index for anything absent. PNNL requires accepting its
terms of use through the Northwest-Infrared portal and cannot be scripted - this script
tells you which species still need a manual download.

    python fetch_reference_data.py --data-dir data/xsec
"""

import argparse
import os
import sys

from pyrospectra.registry import SPECIES_REGISTRY, XSEC_SPECIES

HITRAN_XSC_INDEX = "https://hitran.org/xsc/"
PNNL_PORTAL = "https://nwir.pnnl.gov/"

EXTENSIONS = (".xsc", ".txt", ".dat", ".csv")


def find_local(species, data_dir):
    """Match a species to a local file by name or by any registered alias."""
    if not os.path.isdir(data_dir):
        return None
    names = [species] + list(SPECIES_REGISTRY[species].get("aliases", []))
    wanted = {n.lower().replace("-", "").replace("_", "").replace(",", "")
              for n in names}
    for fname in sorted(os.listdir(data_dir)):
        stem, ext = os.path.splitext(fname)
        if ext.lower() not in EXTENSIONS:
            continue
        key = stem.lower().replace("-", "").replace("_", "").replace(",", "")
        if any(w and (w == key or w in key) for w in wanted):
            return os.path.join(data_dir, fname)
    return None


def query_hitran(species, timeout=30):
    """Ask the HITRAN cross-section index whether it lists this species."""
    try:
        import urllib.request
        req = urllib.request.Request(HITRAN_XSC_INDEX,
                                     headers={"User-Agent": "pyrospectra"})
        with urllib.request.urlopen(req, timeout=timeout) as fh:
            page = fh.read().decode("utf-8", errors="replace").lower()
    except Exception as exc:                                   # noqa: BLE001
        return None, f"could not reach hitran.org ({type(exc).__name__})"

    for name in [species] + list(SPECIES_REGISTRY[species].get("aliases", [])):
        if name.lower() in page:
            return True, f"listed on {HITRAN_XSC_INDEX} as {name!r}"
    return False, "not found in the HITRAN cross-section index"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/xsec",
                    help="directory holding .xsc / .txt cross-section files")
    ap.add_argument("--no-network", action="store_true",
                    help="only check locally; do not query HITRAN")
    args = ap.parse_args()

    print(f"Cross-section species required: {len(XSEC_SPECIES)}")
    print(f"Looking in: {os.path.abspath(args.data_dir)}\n")

    found, missing, snippet = {}, [], []
    for species in XSEC_SPECIES:
        window = SPECIES_REGISTRY[species]["windows"][0]
        path = find_local(species, args.data_dir)
        if path:
            print(f"  [ok]      {species:<9} {window[0]:>7.1f}-{window[1]:<7.1f} {path}")
            found[species] = path
            snippet.append(f"    {species!r:<11}: {path!r},")
            continue
        note = ""
        if not args.no_network:
            listed, note = query_hitran(species)
            note = f"  ({note})" if note else ""
        print(f"  [MISSING] {species:<9} {window[0]:>7.1f}-{window[1]:<7.1f}{note}")
        missing.append(species)

    print()
    if found:
        print("Pass these to generate_reference():\n")
        print("xsec_paths = {")
        print("\n".join(snippet))
        print("}\n")

    if missing:
        print(f"{len(missing)} species still need data: {', '.join(missing)}")
        print(f"  HITRAN cross-sections: {HITRAN_XSC_INDEX}")
        print(f"  PNNL (manual, terms of use apply): {PNNL_PORTAL}")
        print("\nSpecies with no file are excluded from the retrieval. Say so in the "
              "paper\nrather than letting them disappear silently, as v1.0 did.")
        return 1

    print("All cross-section species have data. See DATA_SOURCES.md section 4 on the "
          "448 K\nextrapolation before quoting these emission factors.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
