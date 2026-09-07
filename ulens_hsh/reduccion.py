from astropy.io import fits
from astropy.io.fits import getval
from astropy import stats
import astropy.units as u
import ccdproc
from ccdproc import ImageFileCollection
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Backend no GUI, salva plots sin display
from pathlib import Path
from astropy.wcs import WCS
import warnings
from astropy.utils.exceptions import ErfaWarning
warnings.filterwarnings("ignore", category=ErfaWarning)
from fits_io import imstats, read_fits_data, img_stats, image_collection
from astropy import log
log.setLevel('ERROR')
from matplotlib.gridspec import GridSpec
from tqdm.auto import tqdm
from pathlib import Path
from ccdproc import ImageFileCollection
import ccdproc


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
'''
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
'''
def darkcombine(mydarks, darkcorrection=True, plot=True):
    if not darkcorrection:
        print("Darkcombine set to false")
        return None

    if len(mydarks) == 0:
        print("      NO DARK IMAGES TO COMBINE!")
        return {}

    night_dir = Path(mydarks[0]).parent
    exptimes = sorted(set(float(getval(d, "EXPTIME")) for d in mydarks))

    master_darks = {}

    for exptime in exptimes:
        dark_list = []

        for dark in mydarks:
            if float(getval(dark, "EXPTIME")) == exptime:
                ccd = ccdproc.CCDData.read(dark, unit="adu")
                dark_list.append(ccd)

        if len(dark_list) == 0:
            continue

        print(f"      Combining {len(dark_list)} darks with EXPTIME={exptime}s")

        combiner = ccdproc.Combiner(dark_list)
        combiner.sigma_clipping(low_thresh=2, high_thresh=5, func=np.ma.mean)

        master_dark = combiner.average_combine()
        master_dark.header = dark_list[0].header.copy()
        master_dark.header["EXPTIME"] = exptime
        outname = f"Dark{exptime:g}.fits"
        outpath = night_dir / outname
        master_dark.write(outpath, overwrite=True)

        fits.setval(outpath, "IMAGETYP", value="masterdark")
        fits.setval(outpath, "FILENAME", value=outname)
        fits.setval(outpath, "EXPTIME", value=exptime)

        master_darks[exptime] = master_dark
        
        if plot:
            d_min, d_max, d_mean, d_std = imstats(np.asarray(master_dark))

            plt.figure(dpi=250)
            plt.imshow(
                master_dark,
                vmin=d_mean - 4*d_std,
                vmax=d_mean + 4*d_std,
                origin="lower"
            )
            plt.colorbar()
            plt.title(f"Master dark {exptime:g}s")
            plt.savefig(
                night_dir / f"masterdark{exptime:g}.pdf",
                dpi="figure",
                format="pdf",
                bbox_inches="tight"
            )
            plt.close()

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




# =============================================================================
# ORQUESTADOR DE REDUCCIÓN
# =============================================================================

def run_reduction(images,
                  zero_correction=True,
                  flat_correction=True,
                  dark_correction=False,
                  scale_dark=True):
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
    df = images.summary.to_pandas()
    data_dir = images.location
    mybias   = [data_dir / f for f in df[df["imagetyp"]=="zero"].file]
    mydarks = [data_dir / f for f in df[(df["imagetyp"]=="dark") & (df["calibz"]=="no")].file]
    myflats  = [data_dir / f for f in df[(df["imagetyp"]=="flat")&(df["calibz"]=="no")].file]
    myimages = [data_dir / f for f in df[(df["imagetyp"]=="object") & (df["calibz"]=="no")].file]


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
                fits.setval(data_dir / f"B{Path(dark).name}", "FILENAME", value=f"B{Path(dark).name}")

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
                fits.setval(data_dir / f"B{flat.name}", "FILENAME", value=f"B{flat.name}")
                

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
                fits.setval(data_dir / f"B{image.name}", "FILENAME", value=f"B{image.name}")

    else:
        print("   → Zero correction disabled")
        master_bias = None

    # -------------------------------------------------------------------------
    # 3) Dark combine (opcional)
    # -------------------------------------------------------------------------
    #if dark_correction:
    #    print("   → Dark correction")
    #    master_darks = darkcombine(mydarks, dark_correction, data_dir)
    #    outputs["master_darks"] = master_darks
    # -------------------------------------------------------------------------
    # 3) Dark combine + dark correction
    # -------------------------------------------------------------------------
    if dark_correction:
        print("   → Dark correction")

        # 1) combinar darks bias-subtracted
        images = image_collection(data_dir)
        df = images.summary.to_pandas()

        mydarks_b = [
            data_dir / f for f in df[
                (df["imagetyp"] == "dark") &
                (df["calibz"] == "subtracted bias")
            ].file
        ]

        master_darks = darkcombine(mydarks_b, dark_correction)
        outputs["master_darks"] = master_darks

        # 2) aplicar dark a imágenes científicas ya bias-subtracted
        print("   → Applying dark correction to science images")

        images = image_collection(data_dir)
        df = images.summary.to_pandas()

        myimages_dark = [
            data_dir / f for f in df[
                (df["imagetyp"] == "object") &
                (df["calibz"]=="subtracted bias") &
                (df["calibd"]=="no") &
                (df["calibf"]=="no")
            ].file
        ]
        scaled_dark_usage = {}
        for image in tqdm(myimages_dark, desc="      → Dark subtraction from science images"):
            if scale_dark:
                exptime = float(getval(image, "EXPTIME"))

                if exptime in master_darks:
                    dark_key = exptime
                else:
                    dark_key = min(master_darks.keys(), key=lambda k: abs(k - exptime))
                    scaled_dark_usage[(dark_key, exptime)] = scaled_dark_usage.get((dark_key, exptime), 0) + 1
                    #print(f"      ⚠ Using scaled dark {dark_key}s for image {exptime}s: {image.name}")
                masterdark = master_darks[dark_key]
            else:
                exptime = float(getval(image, "EXPTIME"))

                if exptime not in master_darks:
                    print(f"      ⚠ No master dark for EXPTIME={exptime}s: {image.name}")
                    continue
                masterdark = master_darks[exptime]

            ccd = ccdproc.CCDData.read(image, unit="adu")

            ccd = ccdproc.subtract_dark(
                ccd,
                masterdark,
                exposure_time="EXPTIME",
                exposure_unit=u.second,
                scale=scale_dark,
                add_keyword={"calibd": "dark correction"}
            )

            ccd.write(data_dir / f"D{image.name}", overwrite=True)
            fits.setval(data_dir / f"D{image.name}", "FILENAME", value=f"D{image.name}")
        if scaled_dark_usage:
            print("      ⚠ Scaled darks used:")
            for (dark_exp, img_exp), n in scaled_dark_usage.items():
                print(f"         {n} images: dark {dark_exp:g}s → image {img_exp:g}s")

    # -------------------------------------------------------------------------
    # 4) Flat combine + flat correction
    # -------------------------------------------------------------------------
    if flat_correction:
        print("   → Flat correction")
        
        images = image_collection(data_dir)
        df = images.summary.to_pandas()
        #myimages = [data_dir / f for f in df[(df["imagetyp"]=="object") 
        #                                     & (df["calibz"]=="subtracted bias") 
        #                                     & (df["calibf"]=="no")].file]
        if dark_correction:
            myimages = [data_dir / f for f in df[
                (df["imagetyp"]=="object") &
                (df["calibz"]=="subtracted bias") &
                (df["calibd"]=="dark correction") &
                (df["calibf"]=="no")
            ].file]
        else:
            myimages = [data_dir / f for f in df[
                (df["imagetyp"]=="object") &
                (df["calibz"]=="subtracted bias") &
                (df["calibf"]=="no")
            ].file]
        myflats = [data_dir / f for f in df[(df["imagetyp"]=="flat") 
                                            & (df["calibz"]=="subtracted bias")
                                            & (df["calibf"]=="no")].file]
        
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
                fits.setval(data_dir / f"F{image.name}", "FILENAME", value=f"F{image.name}")

    print("✓ Reduction finished")
    return outputs
'''
# -----------------------------------------------------------------------------
# Plot de reducción
# -----------------------------------------------------------------------------
import pandas as pd
def plot_reduction(images, objname, output_name=None, show=False,
                   overwrite=False, per_page=30):
    """
    Genera un plot de control de calidad de la reducción:
    RAW, Bias, RAW-Bias, FlatBias, Bias-FlatBias + histogramas.

    Parameters
    ----------
    images : ImageFileCollection
        Collection of ccdproc.

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
    night_dir = images.location 
    output_path = night_dir / output_name
    if output_path.exists() and not overwrite:
        print(f"      ✓ Plot already exists: {output_path} (overwrite=False). Skipping.")
        return output_path

    # ------------------------------------------------------------------
    # Selección de archivos desde el dataset
    # ------------------------------------------------------------------
    if isinstance(images, ImageFileCollection):
        df = images.summary.to_pandas()
        #raw_files = df[(df["imagetyp"]=="object")&(df["calibz"]=="no")&(df["object"]==objname)].file.values
        #bias_files = df[(df["imagetyp"]=="object")&(df["calibz"]!="no")&(df["calibf"]=="no")&(df["object"]==objname)].file.values
        #flatbias_files = df[(df["imagetyp"]=="object")&(df["calibz"]!="no")&(df["calibf"]!="no")
        #                    &(df["astromet"]=="no")&(df["object"]==objname)].file.values
        raw_files = df[
            (df["imagetyp"]=="object") &
            (df["calibz"]=="no") &
            (df["object"]==objname)
        ].file.values

        bias_files = df[
            (df["imagetyp"]=="object") &
            (df["calibz"]=="subtracted bias") &
            (df["calibf"]=="no") &
            (df["file"].str.startswith("B", na=False)) &
            (~df["file"].str.startswith("DB", na=False)) &
            (df["object"]==objname)
        ].file.values

        dark_files = df[
            (df["imagetyp"]=="object") &
            (df["calibz"]=="subtracted bias") &
            (df["calibd"]=="dark correction") &
            (df["calibf"]=="no") &
            (df["object"]==objname)
        ].file.values

        flatbias_files = df[
            (df["imagetyp"]=="object") &
            (df["calibz"]=="subtracted bias") &
            (df["calibd"].isin(["dark correction", "no"])) &
            (df["calibf"]!="no") &
            (~df["file"].str.contains("_crclean", na=False)) &
            (df["object"]==objname)
        ].file.values
        
        raw = sorted([night_dir / f for f in raw_files])
        bias = sorted([night_dir / f for f in bias_files])
        flatbias = sorted([night_dir / f for f in flatbias_files])

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
'''
def plot_reduction(images, objname, output_name=None, show=False,
                   overwrite=False, per_page=20):
    """
    Genera un plot de control de calidad de la reducción:
    RAW, Bias, RAW-Bias, Dark, Bias-Dark, Flat, Dark-Flat, CR-cleaned + histogramas.

    Parameters
    ----------
    images : ImageFileCollection
        Collection of ccdproc.
    objname : str
        Nombre del objeto (para filtrar archivos).
    output_name : str or None
        Nombre del archivo de salida. Si None, se usa 'reduction_<obj>.png'.
    show : bool
        Si True, muestra el plot en pantalla.
    overwrite : bool
        Si False y la imagen existe, no hace nada.
    per_page : int
        Número de imágenes por página
    """

    output_name = f"reduction_{objname}.png"
    night_dir = images.location 
    output_path = night_dir / output_name
    
    if output_path.exists() and not overwrite:
        print(f"      ✓ Plot already exists: {output_path} (overwrite=False). Skipping.")
        return output_path

    # ------------------------------------------------------------------
    # Selección de archivos desde el dataset
    # ------------------------------------------------------------------
    if isinstance(images, ImageFileCollection):
        df = images.summary.to_pandas()
        for col in ["file", "filename", "imagetyp", "object", "calibz", "calibd", "calibf", "crclean"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        objname = objname.strip()
        raw_files = df[(df["imagetyp"]=="object")&(df["calibz"]=="no")&(df["object"]==objname)].file.values
        bias_files = df[
            (df["imagetyp"]=="object") & (df["calibz"]=="subtracted bias") &
            (df["calibf"]=="no") & (df["file"].str.startswith("B", na=False)) &
            (~df["file"].str.startswith("DB", na=False)) & (df["object"]==objname)
        ].file.values
        dark_files = df[
            (df["imagetyp"]=="object") & (df["calibz"]=="subtracted bias") &
            (df["calibd"]=="dark correction") & (df["calibf"]=="no") &
            (df["object"]==objname)
        ].file.values
        flatbias_files = df[
            (df["imagetyp"]=="object") & (df["calibz"]=="subtracted bias") &
            (df["calibd"].isin(["dark correction", "no"])) & (df["calibf"]!="no") &
            (~df["file"].str.contains("_crclean", na=False)) & (df["object"]==objname)
        ].file.values
        #flatbias_files = df[(df["imagetyp"]=="object")&(df["calibz"]!="no")&(df["calibf"]!="no")
        #                    &(~df["file"].str.contains("_crclean", na=False))
        #                    &(df["object"]==objname)].file.values
        crclean_files = df[(df["imagetyp"]=="object")
                          &(df["file"].str.contains("_crclean", na=False))
                          &(df["object"]==objname)].file.values

        raw = sorted([night_dir / f for f in raw_files])
        bias = sorted([night_dir / f for f in bias_files])
        dark = sorted([night_dir / f for f in dark_files])
        flatbias = sorted([night_dir / f for f in flatbias_files])
        crclean = sorted([night_dir / f for f in crclean_files])
    # Match sequences: cada raw debe tener su bias, flatbias, y (opcionalmente) crclean
    # Usamos el identificador de versión (ej: i0001, v0042)
    import re
    
    def extract_version(path):
        """Extrae el identificador de versión del nombre de archivo"""
        # Busca patrón como i0001, v0042, etc.
        match = re.search(r'([iv])(\d{4})', path.name.lower())
        if match:
            return match.group(1) + match.group(2)
        return None
    
    # Construir diccionario de versiones
    from collections import defaultdict
    sequences = defaultdict(dict)
    
    for r in raw:
        ver = extract_version(r)
        if ver:
            sequences[ver]['raw'] = r
    
    for b in bias:
        ver = extract_version(b)
        if ver:
            sequences[ver]['bias'] = b
            
    for d in dark:
        ver = extract_version(d)
        if ver:
            sequences[ver]['dark'] = d
    
    for fb in flatbias:
        ver = extract_version(fb)
        if ver:
            sequences[ver]['flatbias'] = fb
    
    for cr in crclean:
        ver = extract_version(cr)
        if ver:
            sequences[ver]['crclean'] = cr
    
    # Filtrar solo secuencias completas (al menos raw, bias, flatbias)
    complete_sequences = {
        ver: files for ver, files in sequences.items()
        if all(k in files for k in ['raw', 'bias', 'dark', 'flatbias'])
    }
    
    if len(complete_sequences) == 0:
        print(f"      No complete reduction sequences for object {objname}")
        return None
    
    # Ordenar por versión
    versions_sorted = sorted(complete_sequences.keys())
    n = len(versions_sorted)
    
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

        # Número de columnas: 6 si hay CR-cleaned, 5 si no
        has_crclean = any('crclean' in complete_sequences[versions_sorted[i]] 
                         for i in range(start, end))
        #n_cols = 6 if has_crclean else 5
        n_cols = 8 if has_crclean else 7

        fig = plt.figure(figsize=(4*n_cols, 5 * n_page))
        gs = GridSpec(
            nrows=2 * n_page, 
            ncols=n_cols,
            height_ratios=[4, 1] * n_page,
            hspace=0.35, 
            wspace=0.15
        )

        for local_i, i in enumerate(range(start, end)):
            row_img  = 2 * local_i
            row_hist = 2 * local_i + 1
            
            ver = versions_sorted[i]
            files = complete_sequences[ver]

            raw_img = read_fits_data(files['raw'])
            b_img   = read_fits_data(files['bias'])
            d_img   = read_fits_data(files['dark'])
            fb_img  = read_fits_data(files['flatbias'])
            cr_img  = read_fits_data(files['crclean']) if 'crclean' in files else None

            diff_rb = raw_img - b_img
            diff_bd = b_img - d_img
            diff_df = d_img - fb_img

            vmin_raw = np.percentile(raw_img, 5)
            vmax_raw = np.percentile(raw_img, 99)
            vmin_b   = np.percentile(b_img, 5)
            vmax_b   = np.percentile(b_img, 99)
            vmin_d   = np.percentile(d_img, 5)
            vmax_d   = np.percentile(d_img, 99)
            vmin_fb  = np.percentile(fb_img, 5)
            vmax_fb  = np.percentile(fb_img, 99)

            dmax_rb = np.percentile(np.abs(diff_rb), 99)
            dmax_bd = np.percentile(np.abs(diff_bd), 99)
            dmax_df = np.percentile(np.abs(diff_df), 99)
            
            images_list = [raw_img, b_img, diff_rb, d_img, diff_bd, fb_img, diff_df]
            titles = [
                "RAW: " + files['raw'].stem,
                "Bias: " + files['bias'].stem,
                "RAW − Bias",
                "Dark: " + files['dark'].stem,
                "Bias − Dark",
                "Flat: " + files['flatbias'].stem,
                "Dark − Flat"
            ]
            cmaps = ["gray", "gray", "seismic", "gray", "seismic", "gray", "seismic"]
            vmins = [vmin_raw, vmin_b, -dmax_rb, vmin_d, -dmax_bd, vmin_fb, -dmax_df]
            vmaxs = [vmax_raw, vmax_b,  dmax_rb, vmax_d,  dmax_bd, vmax_fb,  dmax_df]
            
            # Agregar CR-cleaned si existe
            if cr_img is not None:
                images_list.append(cr_img)
                titles.append("CR-cleaned: " + files['crclean'].stem)
                cmaps.append("gray")
                vmin_cr = np.percentile(cr_img, 5)
                vmax_cr = np.percentile(cr_img, 99)
                vmins.append(vmin_cr)
                vmaxs.append(vmax_cr)

            s_raw = img_stats(raw_img)
            s_rb  = img_stats(diff_rb)

            flags = []
            if s_rb["std"] > NOISE_RATIO * s_raw["std"]:
                flags.append("NOISE↑")
            if s_rb["neg_frac"] > NEG_FRAC_MAX:
                flags.append("NEG")

            qc_label = "OK" if len(flags) == 0 else "QC: " + ", ".join(flags)

            for j in range(len(images_list)):
                ax_img = fig.add_subplot(gs[row_img, j])
                im = ax_img.imshow(images_list[j], origin="lower",
                                   cmap=cmaps[j], vmin=vmins[j], vmax=vmaxs[j])
                ax_img.set_title(titles[j], fontsize=11)
                ax_img.axis("off")

                s = img_stats(images_list[j])
                txt = (f"μ={s['mean']:.2f}  med={s['median']:.2f}  σ={s['std']:.2f}\n"
                       f"p5={s['p5']:.1f}  p95={s['p95']:.1f}  neg={100*s['neg_frac']:.2f}%")
                
                # Agregar info de cosmic rays si es la columna CR-cleaned
                if j == len(images_list) - 1 and cr_img is not None:
                    try:
                        n_cosmics = fits.getval(files['crclean'], 'NCOSMIC', default=0)
                        txt += f"\nCRs removed: {n_cosmics}"
                    except:
                        pass
                
                ax_img.text(0.02, -0.14, txt, transform=ax_img.transAxes,
                            fontsize=8, color="yellow",
                            bbox=dict(facecolor="black", alpha=0.5, pad=2))

                if j == 2:  # columna RAW - Bias
                    color = "lime" if qc_label == "OK" else "red"
                    ax_img.text(0.02, 0.95, qc_label, transform=ax_img.transAxes,
                                fontsize=9, color=color, fontweight="bold",
                                bbox=dict(facecolor="black", alpha=0.5, pad=2))

                cb = fig.colorbar(im, ax=ax_img, fraction=0.03, pad=0.02)
                cb.ax.tick_params(labelsize=7)

                ax_hist = fig.add_subplot(gs[row_hist, j])
                data = images_list[j].ravel()
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
                    ax_hist.set_xlim(-dmax_bd, dmax_bd)
                if j == 6:
                    ax_hist.set_xlim(-dmax_df, dmax_df)

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


# =============================================================================
# COSMIC RAY CLEANING
# =============================================================================


def clean_cosmic_rays(fits_path, cr_cfg, output_suffix="_crclean", overwrite=True):
    """
    Remove cosmic rays using L.A.Cosmic (astroscrappy).
    Creates a new file with suffix instead of overwriting.
    
    Parameters
    ----------
    fits_path : Path
        Path to FITS file to clean
    cr_cfg : dict
        Configuration from config.yaml["cosmic_rays"]
    output_suffix : str
        Suffix to add before .fits extension (default: "_crclean")
    
    Returns
    -------
    tuple
        (output_path, n_cosmics, success)
    """
    from astroscrappy import detect_cosmics
    
    fits_path = Path(fits_path)
    
    # Generate output filename
    # Example: FBi0001_wcs.fits → FBi0001_wcs_crclean.fits
    output_path = fits_path.parent / f"{fits_path.stem}{output_suffix}.fits"
    
    # Check if already exists
    if output_path.exists() and not cr_cfg.get("overwrite", False):
        try:
            already_cleaned = fits.getval(output_path, 'CRCLEAN', default=False)
            if already_cleaned:
                n_cosmics = fits.getval(output_path, 'NCOSMIC', default=0)
                #print(f"      ⊙ Already exists: {output_path.name} ({n_cosmics} CRs)")
                return output_path, n_cosmics, True
        except:
            pass
    
    # Read image
    try:
        with fits.open(fits_path) as hdul:
            data = hdul[0].data.astype(float)
            header = hdul[0].header.copy()
    except Exception as e:
        print(f"      ✗ Cannot read {fits_path.name}: {e}")
        return None, 0, False
    
    # Prepare parameters
    params = {
        "sigclip": cr_cfg.get("sigclip", 4.5),
        "sigfrac": cr_cfg.get("sigfrac", 0.3),
        "objlim": cr_cfg.get("objlim", 5.0),
        "niter": cr_cfg.get("niter", 4),
        "readnoise": cr_cfg.get("readnoise", 15.0),
        "satlevel": cr_cfg.get("satlevel", 60000),
        "verbose": False,
    }
    
    # Detect and clean
    try:
        mask, cleaned = detect_cosmics(data, **params)
        n_cosmics = int(np.sum(mask))
    except Exception as e:
        print(f"      ✗ astroscrappy failed on {fits_path.name}: {e}")
        return None, 0, False
    
    # Write cleaned image as NEW file
    try:
        # Update header
        header["CRCLEAN"] = (True, "Cosmic rays cleaned?")
        header["NCOSMIC"] = (n_cosmics, "N cosmic ray pixels removed")
        header["CRSIGCLP"] = (params["sigclip"], "CR detection sigma")
        header["CRNITER"] = (params["niter"], "CR cleaning iterations")
        header["FILENAME"] = output_path.name
        header.add_history(f"Cosmic rays cleaned: {n_cosmics} pixels removed")
        header.add_history(f"Created from: {fits_path.name}")
        
        # Write new file
        fits.writeto(output_path, cleaned, header, overwrite=True)
        
        #if n_cosmics > 0:
        #    print(f"      ✓ Cleaned {n_cosmics:4d} CR pixels: {fits_path.name} → {output_path.name}")
        #else:
        #    print(f"      ⊙ No CRs detected: {output_path.name}")
            
        return output_path, n_cosmics, True
        
    except Exception as e:
        print(f"      ✗ Cannot write cleaned image {output_path.name}: {e}")
        return None, 0, False

def run_cosmic_ray_cleaning(images, cr_cfg, overwrite=True):
    """
    Clean cosmic rays from all calibrated science images.
    
    Parameters
    ----------
    images : ImageFileCollection
        Collection of images in the night directory
    cr_cfg : dict
        Configuration from config.yaml["cosmic_rays"]
    
    Returns
    -------
    dict
        Statistics: n_processed, n_cleaned, total_cosmics
    """
    df = images.summary.to_pandas()
    night_dir = images.location
    
    # Select calibrated science images that haven't been cleaned yet
    mask = (
        (df["imagetyp"] == "object") &
        (df["calibf"] != "no") &
        (df["astromet"] == "no") &
        ~df["file"].str.contains("_crclean", na=False)
    )
    
    to_clean = df[mask]["file"].values
    
    if len(to_clean) == 0:
        print("   → No images to clean (all already processed)")
        return {"n_processed": 0, "n_cleaned": 0, "total_cosmics": 0}
    
    print(f"   → Processing {len(to_clean)} images")
    
    stats = {
        "n_processed": 0,
        "n_cleaned": 0,
        "total_cosmics": 0,
        "output_files": [],
        "failed": []
    }
    
    for img_file in tqdm(to_clean, desc="      Cleaning cosmic rays"):
        img_path = night_dir / img_file
        
        output_path, n_cosmics, success = clean_cosmic_rays(
            img_path, 
            cr_cfg,
            overwrite=cr_cfg.get("overwrite", False)
        )
        
        if success:
            stats["n_processed"] += 1
            stats["output_files"].append(output_path)
            if n_cosmics > 0:
                stats["n_cleaned"] += 1
                stats["total_cosmics"] += n_cosmics
        else:
            stats["failed"].append(img_file)
    
    # Summary
    print(f"   ✓ Processed {stats['n_processed']}/{len(to_clean)} images")
    print(f"   ✓ Found cosmic rays in {stats['n_cleaned']} images")
    print(f"   ✓ Mean cosmic rays removed: {stats['total_cosmics']/stats['n_processed']}")
    
    if stats["failed"]:
        print(f"   ⚠ Failed on {len(stats['failed'])} images:")
        for f in stats["failed"]:
            print(f"      - {f}")
    
    return stats