"""
Example: optically-thick, high-concentration retrieval (PyroSpectra v2.0).

Runs the ``thick_smoke`` retrieval on one close-to-source biomass-burning burn, prints the
detected emission factors and 3-sigma detection limits, and writes the diagnostic-plot suite.

Usage
-----
    python thick_smoke_workflow.py <burn_dir> <fuel_carbon_fraction> [out_dir]

``<burn_dir>`` contains a ``Spectra/`` folder of ``.prn`` spectra and a ``*_PT_Log.txt`` giving
the gas-cell pressure and temperature.  Heavy-VOC cross-sections (if those species are wanted) are
read from the directory set via ``thick_smoke.configure(xsec_dir=...)`` below.
"""
import os, sys, json
import numpy as np

from pyrospectra import thick_smoke as TS
from pyrospectra.io_utils import read_spectra, get_pt


def main(burn_dir, fuel_cf, out_dir="thick_out"):
    os.makedirs(out_dir, exist_ok=True)
    # Configure paths (edit to your machine). The reference cache speeds up repeated runs; the
    # cross-section directory is only needed if heavy-VOC (xsec) species are retrieved.
    TS.configure(
        xsec_dir=os.environ.get("PYROSPECTRA_XSEC", os.path.join(os.getcwd(), "data", "xsec")),
        refcache=os.path.join(out_dir, "_refcache"),
    )

    spec, wv = read_spectra(os.path.join(burn_dir, "Spectra"))
    P, T, _ = get_pt(burn_dir)
    print(f"read {spec.shape[0]} spectra x {spec.shape[1]} channels; cell T={T:.1f} K, P={P:.3f} bar")

    rec = TS.retrieve(spec, wv, P, T, float(fuel_cf), return_fit=True)
    print(f"plume steps={rec['n_plume']}  MCE={rec['MCE']:.4f}  peak dCO2={rec['peak_excess_CO2']:.0f} ppm\n")

    print(f"{'species':8s} {'status':16s} {'EF or DL (g/kg)':>16s} {'SNR':>6s}")
    for s in rec["species"]:
        if s == "H2O":
            continue
        det = TS.detected(rec, s)
        thin = s in ("N2O",)
        ef = (rec["EF_thin"] if thin else rec["EF"]).get(s, float("nan"))
        dl = (rec["EF_thin_DL"] if thin else rec["EF_DL"]).get(s, float("nan"))
        snr = (rec["SNR_thin"] if thin else rec["SNR"]).get(s, float("nan"))
        if det:
            print(f"{s:8s} {'detected':16s} {ef:16.3f} {snr:6.1f}")
        elif TS.detection_limit_ok(rec, s):
            print(f"{s:8s} {'< detection lim':16s} {dl:16.3f} {snr:6.1f}")
        else:
            print(f"{s:8s} {'not constrained':16s} {'--':>16s} {snr:6.1f}")

    # machine-readable summary (drop the bulky fit arrays)
    summary = {k: v for k, v in rec.items()
               if k not in ("R", "obs", "wvf", "allnames", "conc_all", "gi", "exc",
                            "dco2", "plume", "thin", "sw", "concentrations")}
    json.dump(summary, open(os.path.join(out_dir, "ef_summary.json"), "w"), indent=2, default=str)

    # diagnostic-plot suite (PDF)
    from pyrospectra import diagnostics
    paths = diagnostics.all_diagnostics(rec, out_dir)
    print("\nwrote:", ", ".join(os.path.basename(p) for p in paths))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "thick_out")
