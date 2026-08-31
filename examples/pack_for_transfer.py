"""
Pack burn directories into compact archives for transfer or archival.

A 666-spectrum burn on the MG5 grid is ~375 MB of .prn text. Roughly half the channels
lie outside every Table D1 window, the wavenumber column is repeated in all 666 files,
and the intensities are written with far more digits than the detector noise supports.
Removing all three gives ~15x, which brings one burn under the 30 MB per-file limit of
the Claude web interface.

    python pack_for_transfer.py --root data/burns --out data/packed
    python pack_for_transfer.py --root data/burns --out data/packed --no-windows
    python pack_for_transfer.py --root data/burns --out data/packed --drop-h2o-nir

Unpack with pyrospectra.read_packed(), or pass the .npz straight to read_data().
"""

import argparse
import sys

from pyrospectra import build_compounds, pack_directory_tree


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="directory containing burn folders")
    ap.add_argument("--out", required=True, help="destination for the .npz archives")
    ap.add_argument("--pattern", default="*", help="glob for burn folder names")
    ap.add_argument("--pad", type=float, default=2.0,
                    help="cm^-1 of headroom either side of each window (default 2)")
    ap.add_argument("--no-windows", action="store_true",
                    help="keep the whole grid; storage format only, nothing discarded")
    ap.add_argument("--drop-h2o-nir", action="store_true",
                    help="also drop the H2O 6600-7600 cm^-1 window, the weakest of the "
                         "four and the single largest contributor to the mask "
                         "(a further ~28%% smaller)")
    args = ap.parse_args()

    compounds = build_compounds()
    if args.drop_h2o_nir:
        compounds["H2O"] = dict(
            compounds["H2O"],
            bounds=[b for b in compounds["H2O"]["bounds"] if b[0] < 6000])
        print("Dropping the H2O 6600-7600 cm^-1 window from the retained channels.\n"
              "H2O is still fitted in the other three windows of Table D1.\n")

    manifests = pack_directory_tree(
        args.root, args.out, pattern=args.pattern, compounds=compounds,
        pad=args.pad, windows=not args.no_windows)

    print(f"\nWrote {len(manifests)} archives and a manifest.json recording each "
          "source\ndirectory, channel count, cell state and SHA-256.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
