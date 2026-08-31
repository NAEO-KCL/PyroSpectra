"""
Compact burn archives for transfer and storage.

A MATRIX-MG5 burn directory is enormous for what it contains. A 666-spectrum burn on
the ~29,900-channel grid is roughly 500 MB of ASCII, and almost all of that is waste:

  * the wavenumber column is identical in every ``.prn`` file and is stored 666 times;
  * around half the channels lie outside every fitting window in Table D1 and are
    discarded by the reference mask before the retrieval ever sees them;
  * intensities are written as ~25 bytes of text per value to hold a number that
    carries far fewer significant figures than that.

:func:`pack_burn` removes all three. It writes a single compressed ``.npz`` holding the
retained channels as float32, the wavenumber axis once, the cell state, the timestamps
and a manifest. :func:`read_packed` restores it, and :func:`pyrospectra.read_data`
accepts a packed file wherever it accepts a directory.

Packing is lossy only in the sense that discarded channels are gone. Set
``windows=None`` to keep the full grid and lose nothing but the storage format.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .registry import build_compounds

#: float32 carries ~7 significant decimal digits, far more than the ~3-4 that survive
#: FTIR detector noise. The round-trip error is verified against this at pack time.
_MAX_ACCEPTABLE_RELATIVE_ERROR = 1e-5


def fitted_channel_mask(wavenumbers, compounds=None, pad=0.0):
    """
    Boolean mask of channels lying inside at least one fitting window.

    Computed from the window bounds alone, so it needs no spectroscopic data and can be
    run before the reference matrix exists. Uses the same Table D1 windows the retrieval
    will use, so nothing that would have been fitted is discarded.

    Parameters
    ----------
    wavenumbers : np.ndarray
    compounds : dict, optional
        As from :func:`pyrospectra.build_compounds`. Default: the full registry.
    pad : float
        Extra cm^-1 either side of each window, kept as headroom in case a window is
        later widened. A few cm^-1 costs very little.
    """
    w = np.asarray(wavenumbers, dtype=float)
    compounds = build_compounds() if compounds is None else compounds
    mask = np.zeros(w.size, dtype=bool)
    for spec in compounds.values():
        for lo, hi in spec["bounds"]:
            mask |= (w >= lo - pad) & (w <= hi + pad)
    return mask


def pack_burn(directory, output, compounds=None, pad=2.0, windows=True,
              dtype=np.float32, verify=True):
    """
    Pack a burn directory into a single compressed archive.

    Parameters
    ----------
    directory : path
        Burn directory, containing ``Spectra/*.prn`` and a ``*PT_Log.txt``.
    output : path
        Destination ``.npz``.
    windows : bool
        Keep only channels inside a Table D1 window (plus ``pad``). Set False to keep
        the whole grid.
    verify : bool
        Check that the dtype round-trip is negligible against the data's own precision,
        and raise if not.

    Returns
    -------
    dict
        Manifest, including the compression achieved.
    """
    from .io_utils import read_spectra, get_pt

    directory = Path(directory)
    output = Path(output)
    spectra, w = read_spectra(directory / "Spectra")
    P, T, datetime = get_pt(directory)

    original_bytes = sum(f.stat().st_size for f in (directory / "Spectra").glob("*.prn"))

    if windows:
        mask = fitted_channel_mask(w, compounds, pad=pad)
        if not mask.any():
            raise ValueError(
                "no channel falls inside any fitting window - are these wavenumbers "
                f"({w.min():.1f}-{w.max():.1f} cm-1) what you expect?")
        kept_w, kept = w[mask], spectra[:, mask]
    else:
        mask = np.ones(w.size, dtype=bool)
        kept_w, kept = w, spectra

    packed = kept.astype(dtype)

    if verify:
        scale = np.maximum(np.abs(kept), np.finfo(float).tiny)
        rel = float(np.max(np.abs(packed.astype(np.float64) - kept) / scale))
        if rel > _MAX_ACCEPTABLE_RELATIVE_ERROR:
            raise ValueError(
                f"{np.dtype(dtype).name} round-trip changes intensities by up to "
                f"{rel:.2e} relative, above the {_MAX_ACCEPTABLE_RELATIVE_ERROR:.0e} "
                "threshold. Pack with dtype=np.float64.")
    else:
        rel = None

    manifest = {
        "source_directory": str(directory.resolve()),
        "n_spectra": int(kept.shape[0]),
        "n_channels_kept": int(mask.sum()),
        "n_channels_original": int(w.size),
        "wavenumber_range": [float(kept_w.min()), float(kept_w.max())],
        "windowed": bool(windows),
        "window_pad_cm1": float(pad) if windows else None,
        "dtype": np.dtype(dtype).name,
        "max_relative_roundtrip_error": rel,
        "pressure_bar": float(P),
        "temperature_K": float(T),
        "original_bytes": int(original_bytes),
        "pyrospectra_version": _version(),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        spectra=packed,
        wavenumbers=kept_w.astype(np.float64),
        channel_mask=mask,
        datetime=np.asarray(datetime).astype("datetime64[ns]")
        if datetime is not None and np.asarray(datetime).dtype.kind == "M"
        else np.arange(kept.shape[0]),
        pressure_bar=np.array([P]),
        temperature_K=np.array([T]),
        manifest=json.dumps(manifest),
    )

    packed_bytes = output.stat().st_size
    manifest["packed_bytes"] = int(packed_bytes)
    manifest["compression_factor"] = (round(original_bytes / packed_bytes, 1)
                                      if packed_bytes else None)
    manifest["sha256"] = _checksum(output)

    print(f"  {directory.name}: {_human(original_bytes)} -> {_human(packed_bytes)} "
          f"({manifest['compression_factor']}x), "
          f"{manifest['n_channels_kept']} of {manifest['n_channels_original']} channels")
    return manifest


def read_packed(path):
    """
    Restore a packed burn.

    Returns
    -------
    spectra, wavenumbers, pressure, temperature, datetime
        The same tuple as :func:`pyrospectra.read_data`.
    """
    with np.load(path, allow_pickle=False) as z:
        manifest = json.loads(str(z["manifest"]))
        spectra = z["spectra"].astype(np.float64)
        w = z["wavenumbers"]
        P = float(z["pressure_bar"][0])
        T = float(z["temperature_K"][0])
        datetime = z["datetime"]

    print(f"Loaded packed burn: {manifest['n_spectra']} spectra x "
          f"{manifest['n_channels_kept']} channels "
          f"({w.min():.1f}-{w.max():.1f} cm-1), T = {T:.1f} K, P = {P:.5f} bar")
    if manifest["windowed"]:
        print(f"  NOTE: windowed archive - only channels within {manifest['window_pad_cm1']} "
              "cm-1 of a\n  Table D1 window are present. Widening a fitting window "
              "beyond that requires\n  re-packing from the original .prn files.")
    return spectra, w, P, T, datetime


def pack_directory_tree(root, output_dir, pattern="*", **kw):
    """Pack every burn subdirectory of ``root``. Returns a list of manifests."""
    root, output_dir = Path(root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    burns = sorted(d for d in root.glob(pattern)
                   if d.is_dir() and (d / "Spectra").is_dir())
    if not burns:
        raise FileNotFoundError(f"no burn directories (with a Spectra/ subfolder) in {root}")

    print(f"Packing {len(burns)} burns from {root}")
    manifests = [pack_burn(d, output_dir / f"{d.name}.npz", **kw) for d in burns]

    total_in = sum(m["original_bytes"] for m in manifests)
    total_out = sum(m["packed_bytes"] for m in manifests)
    print(f"\nTotal: {_human(total_in)} -> {_human(total_out)} "
          f"({total_in / total_out:.1f}x)")

    index = output_dir / "manifest.json"
    index.write_text(json.dumps(manifests, indent=2))
    over = [m for m in manifests if m["packed_bytes"] > 30 * 1024 ** 2]
    if over:
        print(f"\n{len(over)} archive(s) still exceed 30 MB and cannot be uploaded to "
              "the Claude\nweb interface as single files. Split them by time range, or "
              "drop the H2O\n6600-7600 cm-1 window, which is the largest single "
              "contributor to the mask.")
    return manifests


def _checksum(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024


def _version():
    from . import __version__
    return __version__


__all__ = ["fitted_channel_mask", "pack_burn", "read_packed", "pack_directory_tree"]
