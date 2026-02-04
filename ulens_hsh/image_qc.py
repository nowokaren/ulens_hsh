import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from shutil import copyfile
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ZScaleInterval, ImageNormalize
from astropy.visualization.stretch import LogStretch



# =====================================================
# Always create valid default SExtractor files
# =====================================================

def ensure_sextractor_files(outdir: Path):
    outdir.mkdir(exist_ok=True)

    param_file = outdir / "default.param"
    sex_file   = outdir / "default.sex"
    conv_file  = outdir / "default.conv"

    # ---------------- default.param ----------------
    param_file.write_text(
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

    # ---------------- default.conv ----------------
    sextractor_share = Path("/home/knowogrodski/miniconda3/envs/hsh_images/share/sextractor")
    official_conv = sextractor_share / "default.conv"

    if official_conv.exists():
        copyfile(official_conv, conv_file)
        # print(f"Using official SExtractor default.conv: {official_conv}")
    else:
        conv_file.write_text("""CONV NORM
# 3x3 Gaussian-like convolution mask (standard for star detection)
3 3
1 2 1
2 4 2
1 2 1
""")

    # ---------------- default.sex ----------------
    sex_file.write_text(
        "CATALOG_TYPE       ASCII_HEAD\n"
        "DETECT_TYPE        CCD\n"
        "DETECT_MINAREA     5\n"
        "DETECT_THRESH      1.5\n"
        "ANALYSIS_THRESH    1.5\n"
        "DEBLEND_NTHRESH    32\n"
        "DEBLEND_MINCONT    0.005\n"
        "CLEAN              Y\n"
        "CLEAN_PARAM        1.0\n"
        "PHOT_APERTURES     3,5,8\n"           # multiple apertures for robustness
        "PHOT_AUTOPARAMS    2.5,3.5\n"
        "PHOT_FLUXFRAC      0.5\n"
        "SATUR_LEVEL        50000\n"
        "MAG_ZEROPOINT      25.0\n"
        "GAIN               1.0\n"
        "PIXEL_SCALE        1.0\n"
        "SEEING_FWHM        2.0\n"
        "BACK_SIZE          64\n"
        "BACK_FILTERSIZE    3\n"
        "CHECKIMAGE_TYPE    NONE\n"
        "MEMORY_OBJSTACK    3000\n"
        "MEMORY_PIXSTACK    300000\n"
        "MEMORY_BUFSIZE     1024\n"
        "VERBOSE_TYPE       QUIET\n"           # change to NORMAL for more info
    )


# =====================================================
# Run SExtractor
# =====================================================

def run_sextractor(fits_path: Path, outdir: Path) -> Path:
    ensure_sextractor_files(outdir)

    sex_file   = outdir / "default.sex"
    param_file = outdir / "default.param"
    conv_file  = outdir / "default.conv"
    cat_path   = outdir / f"{fits_path.stem}_cat.txt"

    cmd = [
        "sex", str(fits_path),
        "-c", str(sex_file),
        "-CATALOG_NAME", str(cat_path),
        "-PARAMETERS_NAME", str(param_file),
        "-FILTER", "Y",
        "-FILTER_NAME", str(conv_file),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("\n===== Command executed =====")
        print(" ".join(cmd))
        print("\n===== Content of default.conv =====")
        print(conv_file.read_text())
        print("\n===== SExtractor STDOUT =====")
        print(result.stdout)
        print("\n===== SExtractor STDERR =====")
        print(result.stderr)
        raise RuntimeError("SExtractor failed")

    return cat_path


# =====================================================
# Quality metrics (with robust fallback)
# =====================================================

def compute_metrics(df: pd.DataFrame) -> dict:
    df.columns = df.columns.str.strip()  # just in case

    print(f"      Detected objects: {len(df)}")

    flags = df.get("FLAGS", pd.Series(np.zeros(len(df), dtype=int)))

    # SNR: prefer SNR_WIN if it exists and is not all NaN
    if "SNR_WIN" in df.columns:
        snr = df["SNR_WIN"]
        print(f"      SNR_WIN - min: {snr.min():.2f}, max: {snr.max():.2f}, mean: {snr.mean():.2f}, NaNs: {snr.isna().sum()}")
    elif "FLUX_AUTO" in df.columns and "FLUXERR_AUTO" in df.columns:
        snr = df["FLUX_AUTO"] / df["FLUXERR_AUTO"].replace(0, np.nan)
        print("Using FLUX_AUTO / FLUXERR_AUTO for SNR")
    else:
        snr = pd.Series(np.nan, index=df.index)
        print("WARNING: No SNR available")

    # Good objects: FLAGS == 0 and SNR > 5 (lower to 3 if you want more lenient)
    good_mask = (flags == 0) & (snr > 5) & snr.notna()
    good = df[good_mask]

    print(f"      Objects with FLAGS==0: { (flags == 0).sum() }")
    print(f"      Objects with SNR > 5: { (snr > 5).sum() }")
    print(f"      Objects good (FLAGS=0 and SNR>5): {len(good)}")

    if len(good) == 0:
        # If no good objects, use all detected for FWHM/ellipticity estimation (fallback)
        print("DEBUG: Using fallback: all detected for metrics")
        good = df  # or df[df['FLAGS'] < 4] to allow low flags

    fwhm_med = np.median(good["FWHM_IMAGE"]) if "FWHM_IMAGE" in good else np.nan
    ellip_med = np.median(good["ELLIPTICITY"]) if "ELLIPTICITY" in good else np.nan
    bkg_med = np.median(good.get("FLUX_AUTO", good.get("FLUX_APER_1", np.nan)))

    return dict(
        n_sources=len(good),
        fwhm=fwhm_med,
        ellipticity=ellip_med,
        background=bkg_med
    )


# =====================================================
# Main function
# =====================================================

def run_image_qc(fits_path: Path, qc_cfg: dict, base_dir: Path):
    outdir = base_dir / "qc_outputs"
    cat_path = run_sextractor(fits_path, outdir)

    print(f"Catalog generated: {cat_path}")

    # Read without assuming header
    df = pd.read_csv(cat_path, comment="#", delim_whitespace=True, header=None)

    # Assign EXACT names according to default.param order
    column_names = [
        "NUMBER",
        "X_IMAGE",
        "Y_IMAGE",
        "FLUX_AUTO",
        "FLUXERR_AUTO",
        "MAG_AUTO",
        "MAGERR_AUTO",
        "FWHM_IMAGE",
        "ELLIPTICITY",
        "THETA_IMAGE",
        "FLAGS",
        "SNR_WIN",              # if not present, will be NaN later
        "FLUX_APER_1",
        "FLUXERR_APER_1"
    ]

    # If number of columns doesn't match, warn
    if len(df.columns) != len(column_names):
        print(f"WARNING: Catalog has {len(df.columns)} columns, but we expected {len(column_names)}")
        # Truncate or extend if needed (rare)
        df = df.iloc[:, :len(column_names)]

    df.columns = column_names
    
    metrics = compute_metrics(df)

    use_image = (
        metrics["n_sources"] >= qc_cfg.get("min_sources", 50) and  # fallback if not in cfg
        metrics["fwhm"] <= qc_cfg.get("max_fwhm", 10.0)
    )

    flags = []
    if metrics["n_sources"] < qc_cfg.get("min_sources", 50):
        flags.append("few_sources")
    if metrics["fwhm"] > qc_cfg.get("max_fwhm", 10.0):
        flags.append("bad_seeing")

    return metrics, use_image, ",".join(flags)


# =====================================================
# QC visualization panel
# =====================================================
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from astropy.io import fits
from astropy.visualization import ImageNormalize, ZScaleInterval, LogStretch
from pathlib import Path
import pandas as pd
import numpy as np

def plot_qc_panel(fits_path: Path, cat_path: Path, metrics: dict, outdir: Path):


    df = pd.read_csv(cat_path, comment="#", delim_whitespace=True, header=None)
    
    column_names = [
        "NUMBER", "X_IMAGE", "Y_IMAGE", "FLUX_AUTO", "FLUXERR_AUTO",
        "MAG_AUTO", "MAGERR_AUTO", "FWHM_IMAGE", "ELLIPTICITY", "THETA_IMAGE",
        "FLAGS", "SNR_WIN", "FLUX_APER_1", "FLUXERR_APER_1", 
    ]
    n_cols = df.shape[1]
    df.columns = column_names[:n_cols]

    good = df[(df["FLAGS"] == 0) & (df["FWHM_IMAGE"] > 0)].copy()


    fig = plt.figure(figsize=(10, 12), constrained_layout=True)
    gs = GridSpec(4, 2, figure=fig,
                height_ratios=[0.6, 1.0, 1.0, 1.0],
                )


    ax_img = fig.add_subplot(gs[:2, 0])
    with fits.open(fits_path) as hdul:
        data = hdul[0].data.astype(float)
    
    norm = ImageNormalize(data, interval=ZScaleInterval())
    im = ax_img.imshow(data, norm=norm, cmap='gray', origin='lower')
    #fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04, shrink=0.78)
    fig.colorbar(im, ax=ax_img, location='left', fraction=0.046, 
                 pad=0.02, shrink=0.8, anchor=(0.0, 0.5))
    ax_img.set_title(f"{fits_path.stem}", fontsize=13)
    ax_img.axis('off')

    ax_img.scatter(good["X_IMAGE"], good["Y_IMAGE"],
                   s=50, facecolors='none', edgecolors='lime', lw=1,
                   label=f"Good: {len(good)}")
    ax_img.legend(loc=(0.45,-0.06), fontsize=9)


    ax_text = fig.add_subplot(gs[0, 1])
    ax_text.axis('off')

    text_lines = [
        "Metrics",
        "─────────────────",
        f"n total:      {len(df):>6}",
        f"n good:       {len(good):>6}",
        f"FWHM median:  {metrics.get('fwhm', good['FWHM_IMAGE'].median() if len(good)>0 else np.nan):>6.2f} pix",
        f"Ellip median: {metrics.get('ellipticity', good['ELLIPTICITY'].median() if len(good)>0 else np.nan):>6.3f}",
        f"SNR_WIN med:  {good['SNR_WIN'].median() if len(good)>0 else np.nan:>6.1f}",
        f"Background:   {metrics.get('background', 0):>6.1f} ADU",
    ]
    textstr = "\n".join(text_lines)
    ax_text.text(0.05, 0.95, textstr, transform=ax_text.transAxes,
                 fontsize=11, va='top', ha='left', family='monospace')

    ax_snr = fig.add_subplot(gs[1, 1])
    if len(good) > 3:
        data = good["SNR_WIN"]
        p99 = np.percentile(data, 99)
        p1 = np.percentile(data, 1)
        ax_snr.hist(good["SNR_WIN"], bins=25, range=[p1,p99],
                    alpha=0.8, log=True)
        ax_snr.axvline(good["SNR_WIN"].median() if len(good)>0 else np.nan, 
                    color='red', ls='--', label =f"median = {data.median():.4f}")
        ax_snr.axvline(good["SNR_WIN"].mean() if len(good)>0 else np.nan, 
                    color='k', ls='--', label =f"mean = {data.mean():.4f}")
        ax_snr.set_title("SNR_WIN (escala log)")
        ax_snr.set_xlabel("SNR")
        ax_snr.set_ylabel("N fuentes")
        ax_snr.grid(alpha=0.25)
        ax_snr.legend()

    # ── Fila 2: Ellipticidad + FWHM ──────────────────────────────────────
    data = good["ELLIPTICITY"]
    p99 = np.percentile(data, 99)
    p1 = np.percentile(data, 1)
    ax_ell = fig.add_subplot(gs[2, 0])
    ax_ell.hist(good["ELLIPTICITY"], bins=22, alpha=0.8, range=[p1,p99])
    ax_ell.axvline(good["ELLIPTICITY"].median() if len(good)>0 else np.nan, 
                   color='red', ls='--', label =f"median = {data.median():.4f}")
    ax_ell.axvline(good["ELLIPTICITY"].mean() if len(good)>0 else np.nan, 
                color='k', ls='--', label =f"mean = {data.mean():.4f}")
    ax_ell.set_title("Ellipticidad")
    ax_ell.set_xlabel("e")
    ax_ell.grid(alpha=0.25)
    ax_ell.legend()

    data = good["FWHM_IMAGE"]
    p99 = np.percentile(data, 99)
    p1 = np.percentile(data, 1)
    ax_fwhm = fig.add_subplot(gs[2, 1])
    ax_fwhm.hist(good["FWHM_IMAGE"], bins=24, alpha=0.8, range=[p1,p99])
    ax_fwhm.axvline(good["FWHM_IMAGE"].median() if len(good)>0 else np.nan, 
                    color='red', ls='--', label = f"median = {data.median()}")
    ax_fwhm.axvline(good["FWHM_IMAGE"].mean() if len(good)>0 else np.nan, 
                color='k', ls='--', label = f"mean = {data.mean():.4f}")
    ax_fwhm.set_title("FWHM")
    ax_fwhm.set_xlabel("pix")
    ax_fwhm.grid(alpha=0.25)
    ax_fwhm.legend()

    # ── Fila 3: Orientación + Flujo ──────────────────────────────────────
    ax_theta = fig.add_subplot(gs[3, 0])
    ax_theta.hist(good["THETA_IMAGE"], bins=36, alpha=0.8)
    orient_std = good["THETA_IMAGE"].std()
    ax_theta.set_title(f"Orientación ($\sigma$ = {orient_std:.4f})")
    ax_theta.set_xlabel("Grados (°)")
    ax_theta.set_xlim(-90, 90)
    ax_theta.grid(alpha=0.25)


    ax_flux = fig.add_subplot(gs[3, 1])
    flux_clean = good["FLUX_AUTO"][(good["FLUX_AUTO"] > 0) & good["FLUX_AUTO"].notna()]
    data = np.log10(flux_clean)
    p99 = np.percentile(data, 99)
    p1 = np.percentile(data, 1)
    if len(flux_clean) > 3:
        ax_flux.hist(data, bins=32, alpha=0.8, range=[p1,p99])
        ax_flux.axvline(data.median() if len(good)>0 else np.nan, 
                    color='red', ls='--', label = f"median = {data.median():.4f}")
        ax_flux.axvline(data.mean() if len(good)>0 else np.nan, 
                color='k', ls='--', label = f"mean = {data.mean():.4f}")
        ax_flux.set_title("Flujo (log FLUX_AUTO)")
        ax_flux.set_xlabel("$log_{10}$(flux)")
        ax_flux.grid(alpha=0.25)
    ax_flux.legend()

    # ── Guardar ──────────────────────────────────────────────────────────
    #plt.tight_layout()
    output_png = outdir / f"{fits_path.stem}_qc.png"
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Guardado: {output_png}")
    
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from astropy.io import fits
from astropy.visualization import ImageNormalize, ZScaleInterval
import pandas as pd
import numpy as np
from pathlib import Path

def plot_qc_advanced(fits_path: Path, cat_path: Path, astrom_cat_path = None,
                     obj_radec = None, outdir: Path = None):
    """
    Panel QC avanzado con:
    - Imagen con fuentes matched (verde) y no matched (rojo)
    - Scatter Ellipticidad vs Orientación limpio
    - Residuales astrométricos en plot cuadrado simple
    """
    # ── Lectura de catálogos ─────────────────────────────────────────────
    df = pd.read_csv(cat_path, comment="#", delim_whitespace=True, header=None)
    column_names = [
        "NUMBER", "X_IMAGE", "Y_IMAGE", "FLUX_AUTO", "FLUXERR_AUTO",
        "MAG_AUTO", "MAGERR_AUTO", "FWHM_IMAGE", "ELLIPTICITY", "THETA_IMAGE",
        "FLAGS", "SNR_WIN", "FLUX_APER_1", "FLUXERR_APER_1",
    ]
    df.columns = column_names[:df.shape[1]]

    good = df[(df["FLAGS"] == 0) & (df["FWHM_IMAGE"] > 1)
              & (df["SNR_WIN"] > 4)].copy()

    df_astrom = None
    if astrom_cat_path and astrom_cat_path.exists():
        try:
            df_astrom = pd.read_csv(astrom_cat_path)
            if not {'RA', 'DEC'}.issubset(df_astrom.columns):
                print("   Columnas RA / DEC no encontradas → astrometría omitida")
                df_astrom = None
            else:
                print(f"   → Cargado catálogo astrométrico: {len(df_astrom)} entradas")
        except Exception as e:
            print(f"   Error astrom cat: {e}")
            df_astrom = None

    # ── Figura ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 4), constrained_layout=True)
    gs = GridSpec(1, 3, figure=fig, wspace=0.32)

    # ── Imagen principal con fuentes matched y no matched ─────────────────
    ax_img = fig.add_subplot(gs[0])
    with fits.open(fits_path) as hdul:
        data = hdul[0].data.astype(float).squeeze()
        w = WCS(hdul[0].header)

    norm = ImageNormalize(data, interval=ZScaleInterval())
    ax_img.imshow(data, origin='lower', cmap='gray_r', norm=norm)
    ax_img.set_title(f"{fits_path.stem}", fontsize=14)
    ax_img.axis('off')

    fig.colorbar(ax_img.imshow(data, origin='lower', cmap='gray_r', norm=norm),
                 ax=ax_img, location='left', fraction=0.02, pad=0.01, shrink=0.5)

    # Fuentes base (todas good) en gris tenue
    #ax_img.scatter(good["X_IMAGE"], good["Y_IMAGE"],
    #               s=30, facecolor='none', edgecolor='gray', lw=0.6, alpha=0.4)

    if obj_radec is not None:
        x_obj, y_obj = w.all_world2pix(obj_radec[0], obj_radec[1], 0)
        ax_img.scatter(x_obj, y_obj,
                           s=100, facecolor='none', edgecolor='purple', lw=2,
                           label=f"Object")
    # ── Astrometría: matching y marcación ────────────────────────────────
    if df_astrom is not None:
        ra_sex, dec_sex = w.all_pix2world(good["X_IMAGE"].values, good["Y_IMAGE"].values, 0)
        coord_sex = SkyCoord(ra=ra_sex * u.deg, dec=dec_sex * u.deg, frame='icrs')
        coord_ref = SkyCoord(
            ra=df_astrom["RA"].values * u.deg,
            dec=df_astrom["DEC"].values * u.deg,
            frame='icrs'
        )

        idx, sep, _ = coord_sex.match_to_catalog_sky(coord_ref)
        has_match = sep < 1.0 * u.arcsec

        matched = good[has_match].copy()
        unmatched = good[~has_match].copy()

        # Marcadores para matched (verde lima)
        if not matched.empty:
            ax_img.scatter(matched["X_IMAGE"], matched["Y_IMAGE"],
                           s=80, facecolor='none', edgecolor='lime', lw=1,
                           label=f"Matched catalog ({len(matched)})")
        
        x_ref, y_ref = w.all_world2pix(df_astrom["RA"].values, 
                                    df_astrom["DEC"].values, 0)
        
        ax_img.scatter(x_ref, y_ref,
                s=100, facecolor='none', edgecolor='red', lw=1, alpha=0.7,
                label=f"Cat. astro ({len(df_astrom)})")

        # Marcadores para no matched (rojo, más visibles)
        #if not unmatched.empty:
        #    ax_img.scatter(unmatched["X_IMAGE"], unmatched["Y_IMAGE"],
        #                   s=100, facecolor='none', edgecolor='red', lw=1, alpha=0.95,
        #                   label=f"Sin match ({len(unmatched)})")

        #ax_img.legend(loc='upper right', fontsize=10, framealpha=0.85)

  
        #ax_img.scatter(good["X_IMAGE"], good["Y_IMAGE"],
        #                s=60, facecolor='none', edgecolor='cyan', lw=10.5,
        #                label=f"Good  SExtractor sources ({len(good)})")
        ax_img.legend(loc=(-0.1, -0.1), fontsize=8, ncol=2)

    # ── Scatter Ellipticidad vs Orientación (mejorado) ────────────────────
    ax_ell_theta = fig.add_subplot(gs[1])
    ax_ell_theta.scatter(good["THETA_IMAGE"], good["ELLIPTICITY"],
                         s=20, alpha=0.7, color='C0', edgecolor='none')
    ax_ell_theta.set_xlabel("Orientation (°)")
    ax_ell_theta.set_ylabel("Ellipticidad")
    ax_ell_theta.set_xlim(-90, 90)
    #ax_ell_theta.set_ylim(0, 1)
    ax_ell_theta.set_title("Ellip vs orientation")
    ax_ell_theta.grid(True, alpha=0.3, ls='--')

    # ── Residuales astrométricos (plot cuadrado simple) ───────────────────
    if df_astrom is not None and len(matched) > 0:
        ra_diff = (coord_sex.ra - coord_ref[idx].ra).to(u.arcsec).value[has_match]
        dec_diff = (coord_sex.dec - coord_ref[idx].dec).to(u.arcsec).value[has_match]

        ax_res = fig.add_subplot(gs[2])
        ax_res.scatter(ra_diff, dec_diff,
                       s=25, color='C3', alpha=0.75, edgecolor='none')
        ax_res.set_xlabel(r"$\Delta$ RA (arcsec)")
        ax_res.set_ylabel(r"$\Delta$ DEC (arcsec)")
        ax_res.set_title("Residuales astrométricos")
        ax_res.grid(True, alpha=0.3, ls='--')

        # Líneas guía simétricas
        lim = 1.5
        ax_res.axvline(0, color='gray', ls='-', lw=1, alpha=0.5)
        ax_res.axhline(0, color='gray', ls='-', lw=1, alpha=0.5)
        ax_res.axvline(lim, color='gray', ls='--', lw=1, alpha=0.6)
        ax_res.axvline(-lim, color='gray', ls='--', lw=1, alpha=0.6)
        ax_res.axhline(lim, color='gray', ls='--', lw=1, alpha=0.6)
        ax_res.axhline(-lim, color='gray', ls='--', lw=1, alpha=0.6)

        # Hacerlo cuadrado
        ax_res.set_aspect('equal')
        ax_res.set_xlim(-lim*1.2, lim*1.2)
        ax_res.set_ylim(-lim*1.2, lim*1.2)

    # ── Guardar ───────────────────────────────────────────────────────────
    output_png = outdir / f"{fits_path.stem}_qc_advanced.png"
    plt.savefig(output_png, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Guardado: {output_png}")