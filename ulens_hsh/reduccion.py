from astropy.io import fits
from astropy.io.fits import getval
from astropy import stats
import ccdproc
from ccdproc import ImageFileCollection
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Backend no GUI, salva plots sin display
from pathlib import Path

from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
import warnings
from astropy.utils.exceptions import ErfaWarning
warnings.filterwarnings("ignore", category=ErfaWarning)
from fits_io import scan_dataset, imstats, read_fits_data, img_stats
from astropy import log
log.setLevel('ERROR')
from matplotlib.gridspec import GridSpec
from tqdm.auto import tqdm



# =============================================================================
# ZEROCOMBINE
# =============================================================================
def zerocombine(mybias, zerocorrection, plot=True):
    """
    Combine and process zero level images.
    
    Args:
        mybias (list): List of bias.
        zerocorrection (boolean): Apply zero level correction?
        
    Returns:
        master_bias: Output zero level.
    """
    
    if zerocorrection == True:
        ## Se verifica que haya bias a combinar.
        if len(mybias) == 0:
            print("      NO ZERO IMAGES TO COMBINE!")
            exit()
        night_dir = Path(mybias[0]).parent
        
        ## Se rechazan imágenes que no cumplen criterios.
        means = []

        for bias in mybias:
            with fits.open(bias) as hdu:
                data = hdu[0].data
                m = stats.sigma_clipped_stats(data, sigma=2, maxiters=5)[0]
                means.append((bias, m))

        # Estadísticos globales
        bias_values = np.array([m for _, m in means])
        mean_bias = bias_values.mean()
        std_bias = bias_values.std()

        print(
            "      Mean value and standard deviation of all zero images: "
            f"{mean_bias:0.1f}  {std_bias:0.1f}"
        )

        # Selección final
        bias_list = []
        for bias, m in means:
            if (m < mean_bias - 2 * std_bias) or (m > mean_bias + 2 * std_bias):
                print(f"      REJECTED: {bias}  MEAN: {m:0.3f}")
            else:
                bias_list.append(bias)


        print("      Zero images ready to combine: %s"%len(bias_list))
        if len(bias_list) == 0: 
            print("      ALL ZERO IMAGES WERE REJECTED!!")
            print("      THERE IS NOTHING MORE TO DO HERE")
            exit()

        ## zerocombine.
        bias_data = []
        for bias in bias_list:
            hdu = fits.open(bias)
            bias_data.append(ccdproc.CCDData(data=hdu[0].data, unit="adu"))
            meta = hdu[0].header
            
        zero_combiner = ccdproc.Combiner(bias_data)
        zero_combiner.sigma_clipping(low_thresh=2, high_thresh=5, func=np.ma.mean)
        del bias_data

        print("      Combining zero images")
        master_bias = zero_combiner.average_combine()
        master_bias.header = meta
        master_bias_filename = 'Zero.fits'
        master_bias_path = Path(night_dir,master_bias_filename)
        master_bias.write(master_bias_path, overwrite=True)
        fits.setval(master_bias_path, 'IMAGETYP', value='masterbias')
        fits.setval(master_bias_path, 'FILENAME', value=master_bias_filename)

        ## Plot
        if plot:
            bias_min, bias_max, bias_mean, bias_std = imstats(np.asarray(master_bias))
            plt.figure(dpi=250)
            plt.imshow(master_bias, vmax=bias_mean + 4*bias_std,
                    vmin=bias_mean - 4*bias_std)
            plt.colorbar()
            plt.savefig(Path(night_dir,"masterbias.pdf"), dpi='figure', format='pdf')
            plt.close()
    else:
        print("Zerocombine set to false")
        return
    return master_bias

def darkcombine(mydarks, darkcorrection):
    """
    NO ESTÁ TERMINADA
    
    Combine and process dark images.
    
    Args:
        mydarks (list): List of dark images.
        darkcorrection (boolean): Apply dark count correction?
        
    Returns:
        master_darks: Combined dark images. One image for each exposure time.
    """
    night_dir = Path(mydarks[0]).parent
    if darkcorrection == True:
        exposures = []
        for dark in mydarks:
            exposures.append(getval(dark, 'EXPTIME'))
        exposures = np.unique(np.asarray(exposures, dtype=str))
    
        master_darks = {}
        combiners = {}
        for exposure in exposures:
            dark_list = []
            for dark in mydarks:
                hdu = fits.open(dark)
                if hdu[0].header['exptime'] == float(exposure):
                    meta = hdu[0].header
                    dark_list.append(ccdproc.CCDData(data=hdu[0].data,
                                                     meta=meta, unit="adu"))
            
            ## get the exposure time as it appears in the fits file for use as a dictionary key
            exp_time_fits_file = dark_list[0].header['exptime']
            
            ## make a combiner for sigma clipping and average combine.
            print("Combining dark images with exposure time = %s s"%exposure)
            dark_combiner = ccdproc.Combiner(dark_list)
            dark_combiner.sigma_clipping(low_thresh=2, high_thresh=5, func=np.ma.mean)
            combiners[exp_time_fits_file] = dark_combiner
            master_darks[exp_time_fits_file] = dark_combiner.average_combine()
            master_darks[exp_time_fits_file].write('Dark%s.fits'%exposure, overwrite=True)
    
            ## Plot
            d_min, d_max, d_mean, d_std = imstats(np.asarray(master_darks[exp_time_fits_file]))
            plt.figure(dpi=250)
            plt.imshow(master_darks[exp_time_fits_file], vmax=d_mean + 4*d_std, vmin=d_mean - 4*d_std)
            plt.colorbar()
            plt.savefig("masterdark%s.pdf"%exp_time_fits_file, dpi='figure', format='pdf')
            plt.close()
    else:
        print("Darkcombine set to false")
        return
    return master_darks

def flatcombine(myflats, flatcorrection, plot=True):
    """
    Combine and process flat field images.
    
    Args:
        myflats (list): List of flat field images to combine.
        flatcorrection (boolean): Apply flat field correction?
        
    Returns:
        master_flat: Combined flat field images. One image for each filter.
    """

    if flatcorrection == True:
        ## se verifica que haya flat-field images a combinar.
        if len(myflats) == 0:
            print("      NO FLAT FIELD IMAGES TO COMBINE!")
            exit()
        night_dir = Path(myflats[0]).parent
        bands = []
        for flat in myflats:
            flat_path = Path(night_dir, flat)
            bands.append(getval(flat_path, 'FILTERS'))
        bands = np.unique(np.asarray(bands, dtype=str))
        
        ## Se rechazan imágenes que no cumplen criterios.
        flat_list = []
        for flat in myflats:
            flat_path = Path(night_dir, flat)
            hdu = fits.open(flat_path)
            data = hdu[0].data
            m = stats.sigma_clipped_stats(data, sigma=2, maxiters=5)[0]
            if m < 26000. or m > 55000.:
                print("      REJECTED: %s"%flat, " MEAN: %0.3f"%m)
            else:
                flat_list.append(flat)
        
        master_flat = {}
        for band in bands:
            flat_filter = []
            print("      Combining flat images in %s-band"%band)
            for flat in flat_list:
                flat_path = Path(night_dir, flat)
                hdu = fits.open(flat_path)
                if hdu[0].header['filters'] == band:
                    meta = hdu[0].header
                    flat_filter.append(ccdproc.CCDData(data=hdu[0].data,
                                                         meta=meta, unit="adu"))
            if len(flat_filter) == 0:
                print("      ALL %s-BAND FLAT FIELD IMAGES WERE REJECTED!"%band)
                exit()

            print("      Combining %i flat images in %s-band"%(len(flat_filter),band))

            flat_combiner = ccdproc.Combiner(flat_filter)
            flat_combiner.sigma_clipping(low_thresh=2, high_thresh=5, func=np.ma.mean)
            scaling_func = lambda arr: 1/np.ma.average(arr)
            flat_combiner.scaling = scaling_func
            master_flat[band] = flat_combiner.average_combine()
            master_flat[band].header = meta
            master_flat_filename = "skyflat%s.fits"%band
            master_flat_path = Path(night_dir, master_flat_filename)
            master_flat[band].write(master_flat_path, overwrite=True)
            fits.setval(master_flat_path, 'IMAGETYP', value='masterflat')
            fits.setval(master_flat_path, 'FILENAME', value=master_flat_filename)
            fits.setval(master_flat_path, 'FILTERS', value=band)

            del flat_filter

            ## Plot.
            if plot:
                f_min, f_max, f_mean, f_std = imstats(np.asarray(master_flat[band]))
                plt.figure(dpi=250)
                plt.imshow(master_flat[band], vmin=f_mean-5*f_std, vmax=f_mean+5*f_std)
                plt.colorbar()
                plt.savefig(Path(night_dir,"masterflat%s.pdf"%band), dpi='figure', format='pdf')
                plt.close()
    else:
        print("      Flatcombine set to false")
        return
    return master_flat


from pathlib import Path
from ccdproc import ImageFileCollection
import ccdproc

from fits_io import scan_dataset   # para reutilizar tu inventario


# =============================================================================
# ORQUESTADOR DE REDUCCIÓN
# =============================================================================

def run_reduction(dataset,
                  zero_correction=True,
                  flat_correction=True,
                  dark_correction=False):
    """
    Ejecuta la reducción completa usando el dataset de scan_dataset:

    - Combina bias
    - Resta bias a darks, flats e imágenes
    - (Opcional) combina darks
    - Combina flats
    - Aplica flat a imágenes

    Parameters
    ----------
    dataset : dict
        Salida de scan_dataset().
    zero_correction : bool
    flat_correction : bool
    dark_correction : bool

    Returns
    -------
    dict
        Archivos generados por la reducción.
    """

    outputs = {}
    data_dir = dataset["dir"]
    # -------------------------------------------------------------------------
    # 1) Inputs desde el dataset
    # -------------------------------------------------------------------------
    mybias   = dataset.get("bias", [])
    mydarks  = dataset.get("darks_raw", [])
    myflats  = dataset.get("flats_raw", [])
    myimages = dataset.get("images_raw", [])

    # -------------------------------------------------------------------------
    # 2) Zerocombine + subtract bias
    # -------------------------------------------------------------------------
    if zero_correction:
        print("   → Zero correction")
        master_bias = zerocombine(mybias, zero_correction)
        outputs["master_bias"] = master_bias

        # Darks
        if dark_correction and len(mydarks) > 0:
            print("   → Subtracting bias from darks")
            for dark in tqdm(mydarks, desc="      → Bias subtraction from darks"):
                ccd = ccdproc.CCDData.read(dark, unit="adu")
                ccd = ccdproc.subtract_bias(
                    ccd, master_bias,
                    add_keyword={'calibz': 'subtracted bias'}
                )
                ccd.write(data_dir / f"B{Path(dark).name}", overwrite=True)

        # Flats
        if flat_correction and len(myflats) > 0:
            print("   → Subtracting bias from flats")
            for flat in tqdm(myflats, desc="      → Bias subtraction from flats"):
                ccd = ccdproc.CCDData.read(flat, unit="adu")
                ccd = ccdproc.subtract_bias(
                    ccd, master_bias,
                    add_keyword={'calibz': 'subtracted bias'}
                )
                ccd.write(data_dir / f"B{flat.name}", overwrite=True)

        # Science images
        if len(myimages) > 0:
            print("   → Subtracting bias from science images")
            for image in tqdm(myimages, desc="      → Bias subtraction from science images"):
                ccd = ccdproc.CCDData.read(image, unit="adu")
                ccd = ccdproc.subtract_bias(
                    ccd, master_bias,
                    add_keyword={'calibz': 'subtracted bias'}
                )
                ccd.write(data_dir / f"B{image.name}", overwrite=True)

    else:
        print("   → Zero correction disabled")
        master_bias = None

    # -------------------------------------------------------------------------
    # 3) Dark combine (opcional)
    # -------------------------------------------------------------------------
    if dark_correction:
        print("   → Dark correction")
        master_darks = darkcombine(mydarks, dark_correction, data_dir)
        outputs["master_darks"] = master_darks

    # -------------------------------------------------------------------------
    # 4) Flat combine + flat correction
    # -------------------------------------------------------------------------
    if flat_correction:
        print("   → Flat correction")

        # Re-escanear para tomar los archivos bias-subtracted
        dataset = scan_dataset(data_dir)
        myflats  = dataset.get("flats_cal", [])
        myimages = dataset.get("images_cal", [])

        if len(myflats) == 0:
            print("   ⚠ No hay flats bias-subtracted para combinar")
        else:
            master_flat = flatcombine(myflats, flat_correction)
            outputs["master_flat"] = master_flat

            # Aplicar flat a imágenes científicas bias-subtracted
            for image in tqdm(myimages, desc="         → Applying flat-field correction to science images"):
                band = getval(image, 'FILTERS')
                ccd = ccdproc.CCDData.read(image, unit="adu")
                ccd = ccdproc.flat_correct(
                    ccd, master_flat[band],
                    add_keyword={'calibf': 'flat correction'}
                )
                ccd.write(data_dir / f"F{image.name}", overwrite=True)

    print("✓ Reduction finished")
    return outputs

# -----------------------------------------------------------------------------
# Plot de reducción
# -----------------------------------------------------------------------------
import pandas as pd
def plot_reduction(dataset, night_dir, objname, output_name=None, show=False,
                   overwrite=False, per_page=30):
    """
    Genera un plot de control de calidad de la reducción:
    RAW, Bias, RAW-Bias, FlatBias, Bias-FlatBias + histogramas.

    Parameters
    ----------
    dataset : dict
        Diccionario devuelto por scan_dataset().
    night_dir : Path
        Directorio de la noche.
    objname : str
        Nombre del objeto (para filtrar archivos).
    output_name : str or None
        Nombre del archivo de salida. Si None, se usa 'reduction_<obj>.png'.
    show : bool
        Si True, muestra el plot en pantalla.
    overwrite : bool
        Si False y la imágen existe, no hace nada.
    """

    output_name = f"reduction_{objname}.png"
    output_path = night_dir / output_name
    if output_path.exists() and not overwrite:
        print(f"      ✓ Plot already exists: {output_path} (overwrite=False). Skipping.")
        return output_path

    # ------------------------------------------------------------------
    # Selección de archivos desde el dataset
    # ------------------------------------------------------------------
    if isinstance(dataset, dict):

        raw_files = [f for f in dataset["images_raw"] if objname in f.name]
        bias_files = [f for f in dataset["images_cal"] if objname in f.name]
        flatbias_files = [f for f in dataset["images_flat"] if objname in f.name]

    elif isinstance(dataset, str) and dataset.endswith(".csv"):

        ds = pd.read_csv(Path(night_dir, dataset))
        ds_obj = ds[(ds["OBJECT"] == objname) & (ds["ASTROMET"].isna())]
        raw_files = [
            Path(night_dir, fname)
            for fname in ds_obj[ds_obj["CALIBZ"].isna()]["FILENAME"]
        ]
        bias_files = [
            Path(night_dir, fname)
            for fname in ds_obj[(ds_obj["CALIBZ"] == "subtracted bias")
                                & (ds_obj["CALIBF"].isna())]["FILENAME"]
        ]
        flatbias_files = [
            Path(night_dir, fname)
            for fname in ds_obj[ds_obj["CALIBF"] == "flat correction"]["FILENAME"]
        ]
    else:
        raise ValueError("dataset debe ser un dict o la ruta a un CSV de metadatos.")
        
        
    raw = sorted(raw_files)
    bias = sorted(bias_files)
    flatbias = sorted(flatbias_files)

    if not (len(raw) == len(bias) == len(flatbias)):
        print(len(raw))
        print(len(bias))
        print(len(flatbias))
        raise ValueError("La cantidad de RAW, Bias y FlatBias no coincide.")

    n = len(raw)
    
    if n==0:
        print(f"      No images for object {objname}")

    # Umbrales de control de calidad
    MEDIAN_TOL = 5.0
    NOISE_RATIO = 0.95
    NEG_FRAC_MAX = 0.01
    
    n_pages = int(np.ceil(n / per_page))

    saved_paths = []

    for page in range(n_pages):
        start = page * per_page
        end   = min((page + 1) * per_page, n)
        n_page = end - start

        print(f"      → Page {page+1}/{n_pages}  (images {start+1}–{end})")

        fig = plt.figure(figsize=(20, 5 * n_page))
        gs = GridSpec(
            nrows=2 * n_page, ncols=5,
            height_ratios=[4, 1] * n_page,
            hspace=0.35, wspace=0.15
        )

        for local_i, i in enumerate(range(start, end)):
            row_img  = 2 * local_i
            row_hist = 2 * local_i + 1

            raw_img = read_fits_data(raw[i])
            b_img   = read_fits_data(bias[i])
            fb_img  = read_fits_data(flatbias[i])

            diff_rb = raw_img - b_img
            diff_bf = b_img - fb_img

            vmin_raw = np.percentile(raw_img, 5)
            vmax_raw = np.percentile(raw_img, 99)
            vmin_b   = np.percentile(b_img, 5)
            vmax_b   = np.percentile(b_img, 99)
            vmin_fb  = np.percentile(fb_img, 5)
            vmax_fb  = np.percentile(fb_img, 99)

            dmax_rb = np.percentile(np.abs(diff_rb), 99)
            dmax_bf = np.percentile(np.abs(diff_bf), 99)

            images = [raw_img, b_img, diff_rb, fb_img, diff_bf]
            titles = [
                "RAW: " + raw[i].stem,
                "Bias: " + bias[i].stem,
                "RAW − Bias",
                "FlatBias: " + flatbias[i].stem,
                "Bias − FlatBias"
            ]
            cmaps  = ["gray", "gray", "seismic", "gray", "seismic"]
            vmins  = [vmin_raw, vmin_b, -dmax_rb, vmin_fb, -dmax_bf]
            vmaxs  = [vmax_raw, vmax_b,  dmax_rb, vmax_fb,  dmax_bf]

            s_raw = img_stats(raw_img)
            s_rb  = img_stats(diff_rb)

            flags = []
            if s_rb["std"] > NOISE_RATIO * s_raw["std"]:
                flags.append("NOISE↑")
            if s_rb["neg_frac"] > NEG_FRAC_MAX:
                flags.append("NEG")

            qc_label = "OK" if len(flags) == 0 else "QC: " + ", ".join(flags)

            for j in range(5):
                ax_img = fig.add_subplot(gs[row_img, j])
                im = ax_img.imshow(images[j], origin="lower",
                                   cmap=cmaps[j], vmin=vmins[j], vmax=vmaxs[j])
                ax_img.set_title(titles[j], fontsize=11)
                ax_img.axis("off")

                s = img_stats(images[j])
                txt = (f"μ={s['mean']:.2f}  med={s['median']:.2f}  σ={s['std']:.2f}\n"
                       f"p5={s['p5']:.1f}  p95={s['p95']:.1f}  neg={100*s['neg_frac']:.2f}%")
                ax_img.text(0.02, -0.14, txt, transform=ax_img.transAxes,
                            fontsize=8, color="yellow",
                            bbox=dict(facecolor="black", alpha=0.5, pad=2))

                if j == 2:
                    color = "lime" if qc_label == "OK" else "red"
                    ax_img.text(0.02, 0.95, qc_label, transform=ax_img.transAxes,
                                fontsize=9, color=color, fontweight="bold",
                                bbox=dict(facecolor="black", alpha=0.5, pad=2))

                cb = fig.colorbar(im, ax=ax_img, fraction=0.03, pad=0.02)
                cb.ax.tick_params(labelsize=7)

                ax_hist = fig.add_subplot(gs[row_hist, j])
                data = images[j].ravel()
                p99 = np.percentile(data, 99)
                p1  = np.percentile(data, 1)
                ax_hist.hist(data, bins=120, range=(p1, p99))
                ax_hist.tick_params(labelsize=6)
                if j == 0:
                    ax_hist.set_ylabel("N", fontsize=7)
                ax_hist.set_xlabel("ADU", fontsize=7)

                if j == 2:
                    ax_hist.set_xlim(-dmax_rb, dmax_rb)
                if j == 4:
                    ax_hist.set_xlim(-dmax_bf, dmax_bf)

        fig.subplots_adjust(top=0.94)
        fig.suptitle(
            f"Noche: {night_dir.name} | Objeto: {objname} | Page {page+1}/{n_pages}",
            fontsize=16, y=0.98
        )

        page_name = f"reduction_{objname}_page_{page+1:02d}.png"
        page_path = night_dir / page_name
        plt.savefig(page_path, bbox_inches="tight", pad_inches=0.05, dpi=150)
        plt.close(fig)

        print(f"      ✓ Saved {page_path}")
        saved_paths.append(page_path)

    return saved_paths