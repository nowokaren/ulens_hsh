"""
image_qc.py
===========
Quality control for FITS images using SExtractor.

Pipeline:
    fits_path -> SExtractor -> catalog -> metrics -> decision -> plots -> FITS header

Public API
----------
run_image_qc(fits_path, qc_cfg, base_dir)  ->  (metrics, use_image, flags_str)
plot_qc_panel(fits_path, cat_path, metrics, outdir)
plot_qc_advanced(fits_path, cat_path, astrom_cat_path, obj_radec, outdir)
write_qc_to_header(fits_path, metrics, use_image, flags_str)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from shutil import copyfile
import subprocess

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from astropy.io import fits
from astropy.visualization import ImageNormalize, ZScaleInterval
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u


# =============================================================================
# SExtractor default configuration
# Override individual keys via qc_cfg["sex_cfg"] — no need to repeat them all.
# =============================================================================

DEFAULT_SEX_CFG = {
    # Detection
    "DETECT_MINAREA":   10,      # minimum pixel area — filters hot pixels / cosmic rays
    "DETECT_THRESH":    2.5,     # detection threshold in sigma above background
    "ANALYSIS_THRESH":  1.5,     # threshold for shape/area measurements

    # Deblending
    "DEBLEND_NTHRESH":  32,
    "DEBLEND_MINCONT":  0.01,

    # Photometry
    "PHOT_APERTURES":   "3,5,8",
    "PHOT_AUTOPARAMS":  "2.5,3.5",
    "PHOT_FLUXFRAC":    0.5,
    "SATUR_LEVEL":      60000,
    "MAG_ZEROPOINT":    25.0,

    # Instrument (HSH/CASLEO)
    "GAIN":             2.0,     # e-/ADU
    "PIXEL_SCALE":      0.53,    # arcsec/px
    "SEEING_FWHM":      4.0,     # estimated seeing [px] — used for star/galaxy separation

    # Background
    "BACK_SIZE":        64,
    "BACK_FILTERSIZE":  3,
}

# Catalog column names — must match default.param exactly
CATALOG_COLUMNS = [
    "NUMBER", "X_IMAGE", "Y_IMAGE",
    "FLUX_AUTO", "FLUXERR_AUTO",
    "MAG_AUTO", "MAGERR_AUTO",
    "FWHM_IMAGE", "ELLIPTICITY", "THETA_IMAGE",
    "FLAGS", "SNR_WIN",
    "FLUX_APER_1", "FLUXERR_APER_1",
]

# FWHM physical bounds for "good" sources:
# < 1.5 px: unresolved artifacts (hot pixels / cosmic rays that survived MINAREA)
# > 15 px:  extended objects, galaxies, heavily blended pairs
FWHM_MIN_STAR = 1.5
FWHM_MAX_STAR = 15.0

# Plate scale for seeing conversion
PLATE_SCALE = 0.53   # arcsec/px


# =============================================================================
# SExtractor configuration files
# =============================================================================

def _write_param_file(path: Path) -> None:
    """Write default.param — columns SExtractor will measure and output."""
    path.write_text(
        "NUMBER\n"
        "X_IMAGE\n"
        "Y_IMAGE\n"
        "FLUX_AUTO\n"
        "FLUXERR_AUTO\n"
        "MAG_AUTO\n"
        "MAGERR_AUTO\n"
        "FWHM_IMAGE\n"
        "ELLIPTICITY\n"
        "THETA_IMAGE\n"
        "FLAGS\n"
        "SNR_WIN\n"
        "FLUX_APER(1)\n"
        "FLUXERR_APER(1)\n"
    )


def _write_conv_file(path: Path) -> None:
    """Write default.conv — use official SExtractor file if available."""
    official = Path(
        "/home/knowogrodski/miniconda3/envs/hsh_images/share/sextractor/default.conv"
    )
    if official.exists():
        copyfile(official, path)
    else:
        path.write_text(
            "CONV NORM\n"
            "# 3x3 Gaussian convolution mask\n"
            "3 3\n"
            "1 2 1\n"
            "2 4 2\n"
            "1 2 1\n"
        )


def _write_sex_file(path: Path, sex_cfg: dict) -> None:
    """Write default.sex from a configuration dict."""
    lines = [f"{key:<20} {val}" for key, val in sex_cfg.items()]
    lines += [
        "CATALOG_TYPE       ASCII_HEAD",
        "DETECT_TYPE        CCD",
        "CLEAN              Y",
        "CLEAN_PARAM        1.0",
        "CHECKIMAGE_TYPE    NONE",
        "MEMORY_OBJSTACK    3000",
        "MEMORY_PIXSTACK    300000",
        "MEMORY_BUFSIZE     1024",
        "VERBOSE_TYPE       QUIET",
    ]
    path.write_text("\n".join(lines) + "\n")


def ensure_sextractor_files(outdir: Path, sex_cfg: dict) -> None:
    """Create all SExtractor configuration files in outdir."""
    outdir.mkdir(exist_ok=True)
    _write_param_file(outdir / "default.param")
    _write_conv_file(outdir / "default.conv")
    _write_sex_file(outdir / "default.sex", sex_cfg)


# =============================================================================
# Run SExtractor
# =============================================================================

def run_sextractor(fits_path: Path, outdir: Path, sex_cfg: dict) -> Path:
    """
    Run SExtractor on fits_path and return the path to the generated catalog.
    Raises RuntimeError with stdout/stderr on failure.
    """
    ensure_sextractor_files(outdir, sex_cfg)

    cat_path = outdir / f"{fits_path.stem}_cat.txt"
    cmd = [
        "sex", str(fits_path),
        "-c",               str(outdir / "default.sex"),
        "-CATALOG_NAME",    str(cat_path),
        "-PARAMETERS_NAME", str(outdir / "default.param"),
        "-FILTER",          "Y",
        "-FILTER_NAME",     str(outdir / "default.conv"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("CMD:", " ".join(cmd))
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise RuntimeError("SExtractor failed — see messages above")

    return cat_path


# =============================================================================
# Read catalog
# =============================================================================

def read_sex_catalog(cat_path: Path) -> pd.DataFrame:
    """Read SExtractor ASCII catalog and assign column names."""
    df = pd.read_csv(cat_path, comment="#", sep=r"\s+", header=None)

    if len(df.columns) != len(CATALOG_COLUMNS):
        print(
            f"WARNING: catalog has {len(df.columns)} columns, "
            f"expected {len(CATALOG_COLUMNS)}"
        )
        df = df.iloc[:, : len(CATALOG_COLUMNS)]

    df.columns = CATALOG_COLUMNS
    return df


# =============================================================================
# Background gradient
# =============================================================================

def compute_background_gradient(fits_path: Path) -> dict:
    """
    Measure background uniformity by dividing the image into a 3x3 grid
    and comparing median values across blocks.

    A strong gradient indicates stray light (moon, nearby bright source).

    Returns
    -------
    dict with:
        bkg_median   : float -- overall background level [ADU]
        bkg_gradient : float -- (max - min) / median across 9 blocks
                                > 0.05 suspicious, > 0.10 is a problem
    """
    with fits.open(fits_path) as hdul:
        data = hdul[0].data if hdul[0].data is not None else hdul[1].data
    data = data.astype(float)
    h, w = data.shape
    bh, bw = h // 3, w // 3

    block_medians = [
        float(np.median(data[i * bh : (i + 1) * bh, j * bw : (j + 1) * bw]))
        for i in range(3)
        for j in range(3)
    ]

    bkg_median   = float(np.median(block_medians))
    bkg_gradient = float(
        (max(block_medians) - min(block_medians)) / (bkg_median + 1e-6)
    )

    return {"bkg_median": bkg_median, "bkg_gradient": bkg_gradient}


# =============================================================================
# Compute QC metrics from catalog
# =============================================================================

def compute_metrics(df: pd.DataFrame, fits_path: Path = None) -> dict:
    """
    Compute QC metrics from a SExtractor catalog DataFrame.

    Good sources criteria:
      - FLAGS == 0
      - SNR > 5
      - FWHM_MIN_STAR < FWHM < FWHM_MAX_STAR

    Falls back to all detected sources if no good sources pass the filter.

    Returns a dict with scalar metrics plus internal keys
    (_df_good, _df_all, _snr_all) used by the plot functions.
    """
    flags_col = df.get("FLAGS", pd.Series(np.zeros(len(df), dtype=int)))

    if "SNR_WIN" in df.columns:
        snr = df["SNR_WIN"]
    elif "FLUX_AUTO" in df.columns and "FLUXERR_AUTO" in df.columns:
        snr = df["FLUX_AUTO"] / df["FLUXERR_AUTO"].replace(0, np.nan)
    else:
        snr = pd.Series(np.nan, index=df.index)

    good_mask = (
        (flags_col == 0)
        & (snr > 5)
        & snr.notna()
        & (df["FWHM_IMAGE"] > FWHM_MIN_STAR)
        & (df["FWHM_IMAGE"] < FWHM_MAX_STAR)
    )
    good = df[good_mask]

    if len(good) == 0:
        print("WARNING: no good sources (FLAG=0, SNR>5, valid FWHM). Using all detected.")
        good = df

    fwhm  = good["FWHM_IMAGE"]
    ellip = good["ELLIPTICITY"]
    theta = good["THETA_IMAGE"]

    # Circular statistics for orientation
    # THETA_IMAGE has 180 deg periodicity: multiply by 2 before averaging
    theta_rad           = np.deg2rad(2.0 * theta)
    mean_sin            = float(np.sin(theta_rad).mean())
    mean_cos            = float(np.cos(theta_rad).mean())
    theta_mean          = float(np.rad2deg(np.arctan2(mean_sin, mean_cos)) / 2.0)
    theta_concentration = float(np.sqrt(mean_sin**2 + mean_cos**2))  # 0=random, 1=all aligned

    snr_good = snr[good_mask] if good_mask.any() else snr

    metrics = {
        # Source counts
        "n_sources":           len(good),
        "n_detected":          len(df),

        # PSF / seeing
        "fwhm_median":         float(np.median(fwhm)),
        "fwhm_p25":            float(np.percentile(fwhm, 25)),
        "fwhm_p75":            float(np.percentile(fwhm, 75)),
        "fwhm_std":            float(fwhm.std()),
        "seeing_arcsec":       round(float(np.median(fwhm)) * PLATE_SCALE, 2),

        # Shape
        "ellip_median":        float(np.median(ellip)),
        "ellip_p75":           float(np.percentile(ellip, 75)),
        "ellip_std":           float(ellip.std()),

        # Orientation (circular statistics)
        "theta_mean":          theta_mean,
        "theta_concentration": theta_concentration,

        # Signal
        "snr_median":          float(np.nanmedian(snr_good)),
        "snr_max":             float(np.nanmax(snr)) if not snr.isna().all() else np.nan,

        # Magnitude (completeness proxy)
        "mag_median":          float(np.median(good["MAG_AUTO"])),

        # Internal -- used by plot functions, not written to header
        "_df_good": good,
        "_df_all":  df,
        "_snr_all": snr,
    }
    if fits_path is not None:
        try:
            with fits.open(fits_path) as hdul:
                hdr = hdul[0].header
                if hdr.get("CRCLEAN", False):
                    metrics["n_cosmic_rays"] = hdr.get("NCOSMIC", 0)
                    metrics["cr_fraction"] = hdr.get("CRFRAC", 0.0)
        except Exception:
            pass
    
    return metrics

# =============================================================================
# Evaluate image: is it usable for photometry?
# =============================================================================

def evaluate_image(metrics: dict, qc_thresholds: dict) -> tuple:
    """
    Decide whether the image is usable based on threshold values.

    qc_thresholds keys
    ------------------
    min_sources              : minimum number of good sources
    max_fwhm                 : maximum acceptable median FWHM [px]
    max_ellipticity          : maximum acceptable median ellipticity
    max_ellip_for_trailing   : ellipticity above which trailing is checked
    max_theta_concentration  : theta_concentration threshold for trailing
                               (checked only when ellipticity also exceeds
                                max_ellip_for_trailing)
    max_bkg_gradient         : maximum relative background gradient (optional)

    Returns
    -------
    (use_image: bool, flags: list[str])
    """
    rejection_flags = []

    if metrics["n_sources"] < qc_thresholds.get("min_sources", 50):
        rejection_flags.append(f"few_sources ({metrics['n_sources']})")

    if metrics["fwhm_median"] > qc_thresholds.get("max_fwhm", 12.0):
        rejection_flags.append(f"bad_seeing ({metrics['fwhm_median']:.1f}px)")

    if metrics["ellip_median"] > qc_thresholds.get("max_ellipticity", 0.45):
        rejection_flags.append(f"high_ellipticity ({metrics['ellip_median']:.3f})")

    # Trailing: high ellipticity AND systematic orientation
    ellip_thresh = qc_thresholds.get("max_ellip_for_trailing", 0.35)
    theta_thresh = qc_thresholds.get("max_theta_concentration", 0.85)
    if (
        metrics["ellip_median"]        > ellip_thresh
        and metrics["theta_concentration"] > theta_thresh
    ):
        rejection_flags.append(
            f"trailing (ellip={metrics['ellip_median']:.2f}, "
            f"theta_c={metrics['theta_concentration']:.2f})"
        )

    # Background gradient / stray light -- only checked if measured
    if "bkg_gradient" in metrics:
        if metrics["bkg_gradient"] > qc_thresholds.get("max_bkg_gradient", 0.08):
            rejection_flags.append(
                f"stray_light (gradient={metrics['bkg_gradient']:.3f})"
            )

    use_image = len(rejection_flags) == 0
    return use_image, rejection_flags


# =============================================================================
# Main entry point
# =============================================================================

def run_image_qc(fits_path: Path, qc_cfg: dict, base_dir: Path) -> tuple:
    """
    Run full QC pipeline on a single FITS image.

    Parameters
    ----------
    fits_path : Path
    qc_cfg    : dict
        Accepted keys (all optional, sensible defaults apply):
          sex_cfg               -- dict overriding DEFAULT_SEX_CFG values
          measure_bkg_gradient  -- bool (default True)
          min_sources           -- see evaluate_image()
          max_fwhm              -- see evaluate_image()
          max_ellipticity       -- see evaluate_image()
          max_ellip_for_trailing-- see evaluate_image()
          max_theta_concentration-- see evaluate_image()
          max_bkg_gradient      -- see evaluate_image()
    base_dir  : Path
        qc_outputs/ subdirectory will be created here.

    Returns
    -------
    (metrics: dict, use_image: bool, flags_str: str)
    """
    outdir  = base_dir / "qc_outputs"
    sex_cfg = {**DEFAULT_SEX_CFG, **qc_cfg.get("sex_cfg", {})}

    cat_path = run_sextractor(fits_path, outdir, sex_cfg)
    df       = read_sex_catalog(cat_path)
    metrics  = compute_metrics(df, fits_path=fits_path)

    if qc_cfg.get("measure_bkg_gradient", True):
        metrics.update(compute_background_gradient(fits_path))

    use_image, flags = evaluate_image(metrics, qc_cfg)
    return metrics, use_image, ",".join(flags)


# =============================================================================
# QC plots
# =============================================================================

def plot_qc_panel(
    fits_path: Path,
    cat_path: Path,
    metrics: dict,
    outdir: Path,
) -> Path:
    """
    Standard QC panel. Saves to outdir/<stem>_qc.png and returns the path.

    Layout
    ------
    Top-left (2 rows) : FITS image with good sources overlaid
    Top-right row 0   : metrics summary text
    Top-right row 1   : SNR histogram
    Middle row        : ellipticity histogram | FWHM histogram
    Bottom row        : orientation histogram | log(flux) histogram
    """
    df_good = metrics["_df_good"]
    df_all  = metrics["_df_all"]
    snr     = metrics["_snr_all"]

    fig = plt.figure(figsize=(10, 12), constrained_layout=True)
    gs  = gridspec.GridSpec(4, 2, figure=fig, height_ratios=[0.6, 1.0, 1.0, 1.0])

    # Image
    ax_img = fig.add_subplot(gs[:2, 0])
    with fits.open(fits_path) as hdul:
        img_data = hdul[0].data.astype(float)
    norm = ImageNormalize(img_data, interval=ZScaleInterval())
    im   = ax_img.imshow(img_data, norm=norm, cmap="gray", origin="lower")
    fig.colorbar(im, ax=ax_img, location="left", fraction=0.046,
                 pad=0.02, shrink=0.8, anchor=(0.0, 0.5))
    ax_img.set_title(fits_path.stem, fontsize=12)
    ax_img.axis("off")
    ax_img.scatter(
        df_good["X_IMAGE"], df_good["Y_IMAGE"],
        s=40, facecolors="none", edgecolors="lime", lw=0.8,
        label=f"Good sources: {len(df_good)}",
    )
    ax_img.legend(loc=(0.3, -0.06), fontsize=8)

    # Metrics text
    ax_txt = fig.add_subplot(gs[0, 1])
    ax_txt.axis("off")
    lines = [
        "Good sources (FLAG=0, SNR>5, valid FWHM):",
        "-" * 36,
        f"  n detected : {metrics['n_detected']:>6d}",
        f"  n good     : {metrics['n_sources']:>6d}",
        f"  FWHM median: {metrics['fwhm_median']:>6.2f} px"
        f"  ({metrics['seeing_arcsec']:.2f}\")",
        f"  Ellip med  : {metrics['ellip_median']:>6.3f}",
        f"  Theta conc : {metrics['theta_concentration']:>6.3f}",
        f"  SNR median : {metrics['snr_median']:>6.1f}",
        f"  Mag median : {metrics['mag_median']:>6.2f}",
    ]
    if "bkg_gradient" in metrics:
        lines.append(f"  Bkg grad   : {metrics['bkg_gradient']:>6.3f}")
    ax_txt.text(
        0.04, 0.95, "\n".join(lines),
        transform=ax_txt.transAxes,
        fontsize=10, va="top", ha="left", family="monospace",
    )

    # SNR histogram
    ax_snr = fig.add_subplot(gs[1, 1])
    snr_good = snr[df_all.index.isin(df_good.index)].dropna()
    if len(snr_good) > 3:
        p1, p99 = np.percentile(snr_good, [1, 99])
        ax_snr.hist(snr_good.to_numpy(), bins=25, range=[p1, p99], alpha=0.8, log=True)
        ax_snr.axvline(metrics["snr_median"], color="red", ls="--",
                       label=f"median={metrics['snr_median']:.1f}")
        ax_snr.set_title("SNR (log scale)")
        ax_snr.set_xlabel("SNR")
        ax_snr.set_ylabel("N sources")
        ax_snr.legend(fontsize=8)
        ax_snr.grid(alpha=0.2)

    # Ellipticity histogram
    ax_ell = fig.add_subplot(gs[2, 0])
    e = df_good["ELLIPTICITY"]
    p1e, p99e = np.percentile(e, [1, 99])
    ax_ell.hist(e.to_numpy(), bins=24, alpha=0.8, range=[p1e, p99e])
    ax_ell.axvline(metrics["ellip_median"], color="red", ls="--",
                   label=f"median={metrics['ellip_median']:.3f}")
    ax_ell.set_title("Ellipticity")
    ax_ell.set_xlabel("e")
    ax_ell.legend(fontsize=8)
    ax_ell.grid(alpha=0.2)

    # FWHM histogram
    ax_fwhm = fig.add_subplot(gs[2, 1])
    fw = df_good["FWHM_IMAGE"]
    p1f, p99f = np.percentile(fw, [1, 99])
    ax_fwhm.hist(fw.to_numpy(), bins=24, alpha=0.8, range=[p1f, p99f])
    ax_fwhm.axvline(metrics["fwhm_median"], color="red", ls="--",
                    label=f"median={metrics['fwhm_median']:.2f}px")
    ax_fwhm.set_title("FWHM")
    ax_fwhm.set_xlabel("px")
    ax_fwhm.legend(fontsize=8)
    ax_fwhm.grid(alpha=0.2)

    # Orientation histogram
    ax_theta = fig.add_subplot(gs[3, 0])
    ax_theta.hist(df_good["THETA_IMAGE"].to_numpy(), bins=36, alpha=0.8)
    ax_theta.set_title(
        f"Orientation  (theta_c={metrics['theta_concentration']:.2f})"
    )
    ax_theta.set_xlabel("degrees")
    ax_theta.set_xlim(-90, 90)
    ax_theta.grid(alpha=0.2)

    # log(flux) histogram
    ax_flux = fig.add_subplot(gs[3, 1])
    flux       = df_good["FLUX_AUTO"]
    flux_clean = flux[(flux > 0) & flux.notna()]
    if len(flux_clean) > 3:
        logf = np.log10(flux_clean)
        p1l, p99l = np.percentile(logf, [1, 99])
        ax_flux.hist(logf.to_numpy(), bins=32, alpha=0.8, range=[p1l, p99l])
        ax_flux.axvline(float(np.median(logf)), color="red", ls="--",
                        label=f"median={float(np.median(logf)):.2f}")
        ax_flux.set_title("log(FLUX_AUTO)")
        ax_flux.set_xlabel("log10(flux)")
        ax_flux.legend(fontsize=8)
        ax_flux.grid(alpha=0.2)

    out_png = outdir / f"{fits_path.stem}_qc.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


def plot_qc_advanced(
    fits_path: Path,
    cat_path: Path,
    astrom_cat_path=None,
    obj_radec=None,
    outdir: Path = None,
    verbose: bool = False,
) -> Path:
    """
    Advanced QC panel. Saves to outdir/<stem>_qc_advanced.png.

    Panels
    ------
    1. FITS image: matched sources (green), reference catalog (red), target (purple)
    2. Ellipticity vs orientation scatter — trailing diagnostic
    3. Astrometric residuals Delta-RA vs Delta-DEC [arcsec]
    """
    df_cat = pd.read_csv(cat_path, comment="#", sep=r"\s+", header=None)
    df_cat.columns = CATALOG_COLUMNS[: df_cat.shape[1]]
    good = df_cat[df_cat["FLAGS"] == 0].copy()

    df_astrom = None
    if astrom_cat_path is not None and Path(astrom_cat_path).exists():
        try:
            df_astrom = pd.read_csv(astrom_cat_path)
            if not {"RA", "DEC"}.issubset(df_astrom.columns):
                if verbose:
                    print("   WARNING: RA/DEC not found in astrometric catalog")
                df_astrom = None
            elif verbose:
                print(f"   Astrometric catalog: {len(df_astrom)} entries")
        except Exception as exc:
            if verbose:
                print(f"   Could not load astrometric catalog: {exc}")

    fig = plt.figure(figsize=(12, 4), constrained_layout=True)
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.32)

    # Panel 1: image
    ax_img = fig.add_subplot(gs[0])
    with fits.open(fits_path) as hdul:
        img_data = hdul[0].data.astype(float).squeeze()
        wcs = WCS(hdul[0].header)

    norm = ImageNormalize(img_data, interval=ZScaleInterval())
    ax_img.imshow(img_data, origin="lower", cmap="gray_r", norm=norm)
    fig.colorbar(
        ax_img.imshow(img_data, origin="lower", cmap="gray_r", norm=norm),
        ax=ax_img, location="left", fraction=0.02, pad=0.01, shrink=0.5,
    )
    ax_img.set_title(fits_path.stem, fontsize=11)
    ax_img.axis("off")

    if obj_radec is not None:
        x_obj, y_obj = wcs.all_world2pix(obj_radec[0], obj_radec[1], 0)
        ax_img.scatter(x_obj, y_obj, s=120, facecolor="none",
                       edgecolor="purple", lw=2, label="Target")

    has_match = None
    coord_sex = coord_ref = idx = None

    if df_astrom is not None:
        ra_sex, dec_sex = wcs.all_pix2world(
            good["X_IMAGE"].values, good["Y_IMAGE"].values, 0
        )
        coord_sex = SkyCoord(ra=ra_sex * u.deg, dec=dec_sex * u.deg)
        coord_ref = SkyCoord(
            ra=df_astrom["RA"].values * u.deg,
            dec=df_astrom["DEC"].values * u.deg,
        )
        idx, sep, _ = coord_sex.match_to_catalog_sky(coord_ref)
        has_match = sep < 1.0 * u.arcsec

        matched = good[has_match]
        if not matched.empty:
            ax_img.scatter(
                matched["X_IMAGE"], matched["Y_IMAGE"],
                s=70, facecolor="none", edgecolor="lime", lw=0.9,
                label=f"Matched ({len(matched)})",
            )

        x_ref, y_ref = wcs.all_world2pix(
            df_astrom["RA"].values, df_astrom["DEC"].values, 0
        )
        ax_img.scatter(
            x_ref, y_ref, s=90, facecolor="none", edgecolor="red",
            lw=0.8, alpha=0.6, label=f"Ref catalog ({len(df_astrom)})",
        )
        ax_img.legend(loc=(-0.1, -0.08), fontsize=7, ncol=2)

    # Panel 2: ellipticity vs orientation
    ax_eo = fig.add_subplot(gs[1])
    ax_eo.scatter(
        good["THETA_IMAGE"].to_numpy(), good["ELLIPTICITY"].to_numpy(),
        s=15, alpha=0.6, color="steelblue", edgecolor="none",
    )
    ax_eo.set_xlabel("Orientation (deg)")
    ax_eo.set_ylabel("Ellipticity")
    ax_eo.set_xlim(-90, 90)
    ax_eo.set_title("Ellipticity vs orientation\n(trailing -> cluster at one angle)")
    ax_eo.grid(True, alpha=0.25, ls="--")

    # Panel 3: astrometric residuals
    if df_astrom is not None and has_match is not None and has_match.any():
        ra_diff  = (coord_sex.ra  - coord_ref[idx].ra ).to(u.arcsec).value[has_match]
        dec_diff = (coord_sex.dec - coord_ref[idx].dec).to(u.arcsec).value[has_match]

        ax_res = fig.add_subplot(gs[2])
        ax_res.scatter(ra_diff, dec_diff, s=20, color="tomato",
                       alpha=0.7, edgecolor="none")
        lim = 1.5
        for v in [-lim, 0, lim]:
            ax_res.axvline(v, color="gray", ls="--" if v else "-",
                           lw=0.8, alpha=0.5)
            ax_res.axhline(v, color="gray", ls="--" if v else "-",
                           lw=0.8, alpha=0.5)
        ax_res.set_aspect("equal")
        ax_res.set_xlim(-lim * 1.2, lim * 1.2)
        ax_res.set_ylim(-lim * 1.2, lim * 1.2)
        ax_res.set_xlabel("Delta-RA (arcsec)")
        ax_res.set_ylabel("Delta-DEC (arcsec)")
        ax_res.set_title("Astrometric residuals")
        ax_res.grid(True, alpha=0.25, ls="--")

    out_png = outdir / f"{fits_path.stem}_qc_advanced.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


# =============================================================================
# Write QC results to FITS header
# =============================================================================

def write_qc_to_header(
    fits_path: Path, metrics: dict, use_image: bool, flags: str
) -> None:
    """
    Write QC keywords to the FITS header.
    Call after run_image_qc().
    """
    try:
        with fits.open(fits_path, mode="update") as hdul:
            h = hdul[0].header
            h["QC_USE"]   = (bool(use_image),                            "Image approved for combination?")
            h["QC_FLAGS"] = (flags or "OK",                              "Rejection flags")
            h["QC_NSRCS"] = (int(metrics.get("n_sources",        0)),    "N good sources (FLAG=0, SNR>5)")
            h["QC_FWHM"]  = (float(metrics.get("fwhm_median",    0.0)),  "Median FWHM [px]")
            h["QC_SEE"]   = (float(metrics.get("seeing_arcsec",  0.0)),  "Seeing [arcsec]")
            h["QC_ELLIP"] = (float(metrics.get("ellip_median",   0.0)),  "Median ellipticity")
            h["QC_THETC"] = (float(metrics.get("theta_concentration", 0.0)),
                             "Theta concentration (0=random,1=aligned)")
            if "bkg_gradient" in metrics:
                h["QC_BKGG"] = (float(metrics["bkg_gradient"]),
                                "Relative background gradient")
            h["QC_DATE"]  = (
                datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), "QC timestamp"
            )
            hdul.flush()
    except Exception as exc:
        print(f"   ERROR writing QC header for {fits_path.name}: {exc}")