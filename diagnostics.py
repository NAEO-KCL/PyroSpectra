"""
Professional diagnostic plots for the optically-thick retrieval (:mod:`pyrospectra.thick_smoke`).

All functions take a retrieval record produced with ``thick_smoke.retrieve(..., return_fit=True)``
(which carries the fit internals: reference matrix ``R``, observed absorbance ``obs``, fitted
region ``wvf``, retrieved concentrations, etc.) and write a PDF:

- :func:`spectral_atlas`   - per-species window decomposition (observed, interferents, continuum,
                             species, total model) at each species' max-excess plume step;
- :func:`hysteresis`       - excess-X vs excess-CO2 with the emission-ratio slope (thin/thick);
- :func:`lcurve_residual`  - Tikhonov L-curve (linear-system conditioning) + full residual spectrum;
- :func:`overview`         - full-region fit at the thickest step + normalised concentration series;
- :func:`all_diagnostics`  - all of the above into ``<outdir>/diag_*.pdf``.

The plots are theme-neutral and vector (PDF, Type-42 fonts) for publication.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from . import thick_smoke as TS

plt.rcParams.update({"pdf.fonttype": 42, "font.size": 9, "axes.grid": True, "grid.alpha": 0.2})

SPEC_ORDER = ["CO2", "CO", "CH4", "NH3", "N2O", "NO2", "C2H2", "C2H4", "CH2O", "HCOOH", "HCN",
              "SO2", "C2H6", "C3H6O", "C5H8", "C4H4O", "C2H6O", "CH3COOH", "HNO2", "CH3CN",
              "C3H6", "C6H5CH3", "CH3CHO", "HCO2CH3", "C3H8"]
_PERV = ("CO2", "H2O", "CO", "CH4")


def _windows(s, config=TS.CFG):
    return [tuple(map(float, w)) for w in config[s][2]]

def _best_t(rec, s):
    gi = rec["gi"]
    if s not in gi:
        return int(np.argmax(rec["dco2"]))
    e = rec["exc"][gi[s]].copy(); e[~rec["plume"]] = -1e9
    return int(np.argmax(e)) if rec["plume"].any() else int(np.argmax(rec["dco2"]))

def plot_fit(ax, rec, s, t=None, pad=6.0, config=TS.CFG):
    """Draw the window decomposition for species ``s`` on ``ax``."""
    wins = _windows(s, config); lo = min(w[0] for w in wins) - pad; hi = max(w[1] for w in wins) + pad
    wvf = rec["wvf"]; w = (wvf >= lo) & (wvf <= hi); ws = wvf[w]
    if t is None:
        t = _best_t(rec, s)
    conc = rec["conc_all"][:, t]; R = rec["R"]; gi = rec["gi"]; alln = rec["allnames"]
    ax.plot(ws, rec["obs"][t, w], color="k", lw=1.5, label="observed", zorder=6)
    interf = np.zeros(int(w.sum()))
    for p in _PERV:
        if p in gi and p != s:
            interf = interf + conc[gi[p]] * R[gi[p]][w]
    ax.plot(ws, interf, color="#17becf", lw=1.0, label="CO2+H2O+CO+CH4")
    blm = sum(conc[alln.index(n)] * R[alln.index(n)][w] for n in alln if n.startswith("BL"))
    if np.any(blm):
        ax.plot(ws, blm, color="#9467bd", lw=1.0, ls=":", label="continuum")
    if s in gi:
        ax.plot(ws, conc[gi[s]] * R[gi[s]][w], color="#d62728", lw=1.8,
                label="%s (%.3g ppm)" % (s, conc[gi[s]]))
    ax.plot(ws, R[:, w].T @ conc, color="#7f7f7f", lw=1.1, ls="--", label="total model")
    for a, b in wins:
        ax.axvspan(a, b, color="gold", alpha=0.10)
    r2 = rec["R2"].get(s); ef = rec["EF"].get(s)
    ax.set_title("%s   EF=%.3g g/kg   R2=%.2f"
                 % (s, ef if ef is not None else float("nan"), r2 if r2 is not None else float("nan")),
                 fontsize=9)
    ax.set_xlabel("wavenumber (cm$^{-1}$)", fontsize=8); ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc="best", framealpha=0.6)


def spectral_atlas(rec, path, species=None, ncols=3, nrows=4, config=TS.CFG):
    """Multi-page PDF: window-decomposition spectral fit for every species."""
    sp = [s for s in (species or SPEC_ORDER) if s in config]
    per = ncols * nrows
    with PdfPages(path) as pdf:
        for pg in range(0, len(sp), per):
            fig, axs = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.0 * nrows))
            axs = np.atleast_1d(axs).ravel()
            for ax in axs:
                ax.axis("off")
            for ax, s in zip(axs, sp[pg:pg + per]):
                ax.axis("on")
                try:
                    plot_fit(ax, rec, s, config=config)
                except Exception as e:
                    ax.set_title("%s: %s" % (s, e), fontsize=7)
            fig.suptitle("Spectral-fit atlas (each species at its max-excess plume step)", fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.97]); pdf.savefig(fig); plt.close(fig)
    return path


def hysteresis(rec, path, species=None, ncols=3, nrows=4):
    """Multi-page PDF: excess-X vs excess-CO2 per species (optical-depth hysteresis + ER slope)."""
    gi = rec["gi"]; exc = rec["exc"]; dco2 = rec["dco2"]; plume = rec["plume"]; thin = rec["thin"]
    sp = [s for s in (species or SPEC_ORDER) if s in gi and s != "H2O"]
    ER = rec["ER"]; per = ncols * nrows
    with PdfPages(path) as pdf:
        for pg in range(0, len(sp), per):
            fig, axs = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.0 * nrows))
            axs = np.atleast_1d(axs).ravel()
            for ax in axs:
                ax.axis("off")
            for ax, s in zip(axs, sp[pg:pg + per]):
                ax.axis("on"); x = exc[gi[s]]; thick = plume & ~thin
                ax.scatter(dco2[thin], x[thin], s=8, c="#1f77b4", alpha=0.5, label="thin")
                ax.scatter(dco2[thick], x[thick], s=8, c="#d62728", alpha=0.5, label="thick")
                sl = ER.get(s)
                if sl is not None and np.isfinite(sl) and plume.any():
                    xs = np.array([0, float(np.nanmax(dco2[plume]))]); ax.plot(xs, sl * xs, "k-", lw=1.2)
                ax.set_title("%s  EF=%.3g  R2=%.2f" % (s, rec["EF"].get(s, float("nan")),
                             rec["R2"].get(s, float("nan"))), fontsize=8)
                ax.set_xlabel("$\\Delta$CO2 (ppm)", fontsize=7); ax.set_ylabel("$\\Delta$X (ppm)", fontsize=7)
                ax.tick_params(labelsize=6)
            fig.suptitle("Excess-vs-CO2 (emission-ratio) diagnostics", fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.97]); pdf.savefig(fig); plt.close(fig)
    return path


def lcurve_residual(rec, path, t=None):
    """Single-page PDF: Tikhonov L-curve (linear-system conditioning) + full residual spectrum."""
    if t is None:
        t = int(np.argmax(rec["dco2"]))
    R = rec["R"]; sw = rec["sw"]; y = rec["obs"][t] * sw; A = (R * sw[None, :]).T
    lams = np.logspace(-4, 2, 40); AtA = A.T @ A; Aty = A.T @ y; I = np.eye(A.shape[1])
    rn = np.array([np.linalg.norm(A @ np.linalg.solve(AtA + l ** 2 * I, Aty) - y) for l in lams])
    sn = np.array([np.linalg.norm(np.linalg.solve(AtA + l ** 2 * I, Aty)) for l in lams])
    lr, ls_ = np.log(rn + 1e-30), np.log(sn + 1e-30)
    d1r, d1s = np.gradient(lr), np.gradient(ls_); d2r, d2s = np.gradient(d1r), np.gradient(d1s)
    curv = np.abs(d1r * d2s - d1s * d2r) / (d1r ** 2 + d1s ** 2) ** 1.5
    k = int(np.nanargmax(curv))
    conc = rec["conc_all"][:, t]; resid = rec["obs"][t] - R.T @ conc; wvf = rec["wvf"]
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        ax[0].loglog(rn, sn, "o-", ms=3, color="#1f77b4")
        ax[0].plot(rn[k], sn[k], "rs", ms=9, label="corner $\\lambda$=%.2g" % lams[k])
        ax[0].set_xlabel("residual norm $\\|Ac-y\\|$"); ax[0].set_ylabel("solution norm $\\|c\\|$")
        ax[0].set_title("Tikhonov L-curve (linear-system conditioning, t=%d)" % t); ax[0].legend()
        ax[1].plot(wvf, rec["obs"][t], "k", lw=0.5, alpha=0.7, label="observed")
        ax[1].plot(wvf, resid, color="#d62728", lw=0.5, label="residual (obs-model)")
        ax[1].axhline(0, color="0.5", lw=0.6)
        ax[1].set_xlabel("wavenumber (cm$^{-1}$)"); ax[1].set_ylabel("decadic absorbance")
        ax[1].set_title("Fit residual over the fitted region (RMS=%.4f)" % np.sqrt(np.mean(resid ** 2)))
        ax[1].legend(fontsize=8)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    return path


def overview(rec, path):
    """Single-page PDF: full-region fit at the thickest step + normalised concentration series."""
    t = int(np.argmax(rec["dco2"])); wvf = rec["wvf"]; conc = rec["conc_all"][:, t]; R = rec["R"]
    gi = rec["gi"]; exc = rec["exc"]; xs = np.arange(exc.shape[1])
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(2, 1, figsize=(12, 8))
        ax[0].plot(wvf, rec["obs"][t], "k", lw=0.4, alpha=0.8, label="observed")
        ax[0].plot(wvf, R.T @ conc, color="#d62728", lw=0.4, alpha=0.8, label="total model")
        ax[0].set_title("Full fitted-region fit, thickest step ($\\Delta$CO2=%.0f ppm)" % rec["dco2"][t])
        ax[0].set_xlabel("wavenumber (cm$^{-1}$)"); ax[0].set_ylabel("decadic absorbance"); ax[0].legend(fontsize=8)
        for s, c in [("CO2", "#ff7f0e"), ("CO", "#8c564b"), ("CH4", "#2ca02c"),
                     ("NH3", "#1f77b4"), ("C2H4", "#9467bd"), ("HCOOH", "#e377c2")]:
            if s in gi:
                y = exc[gi[s]]; ax[1].plot(xs, y / (np.nanmax(np.abs(y)) or 1), lw=1.0, color=c, label=s)
        ax[1].axvline(rec["npre"], color="0.5", ls="--", lw=0.8)
        ax[1].set_title("Excess concentration time series (each normalised to its own max)")
        ax[1].set_xlabel("time step"); ax[1].set_ylabel("normalised excess"); ax[1].legend(fontsize=7, ncol=3)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    return path


def all_diagnostics(rec, outdir, prefix="diag_"):
    """Write the full diagnostic set to ``outdir`` and return the list of paths written."""
    os.makedirs(outdir, exist_ok=True)
    return [spectral_atlas(rec, os.path.join(outdir, prefix + "spectral_atlas.pdf")),
            hysteresis(rec, os.path.join(outdir, prefix + "hysteresis.pdf")),
            lcurve_residual(rec, os.path.join(outdir, prefix + "lcurve_residual.pdf")),
            overview(rec, os.path.join(outdir, prefix + "overview.pdf"))]
