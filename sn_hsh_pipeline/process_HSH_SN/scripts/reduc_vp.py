#%%
# =============================================================================
# ÚLTIMA ACTUALIZACIÓN: 29/03/2021
# =============================================================================

import astropy
from astropy.io import fits
from astropy.io.fits import getheader
from astropy.io.fits import getval
from astropy import stats
from astropy.time import Time
import ccdproc
from ccdproc import ImageFileCollection
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Backend no GUI, salva plots sin display
from pathlib import Path
import multiprocessing
import time
import os
import json
import requests
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
import warnings
from astropy.utils.exceptions import ErfaWarning
warnings.filterwarnings("ignore", category=ErfaWarning)



#%%
# =============================================================================
# ELECCIÓN DE CALIBRACIONES A REALIZAR
# =============================================================================
z = input('Press T if you want to perform zero correction, else press enter ')
f = input('Press T if you want to perform flat correction, else press enter ')
a = input('Press T if you want to run astrometry.net, else press enter ')
r = input('Press T if you want to remove images, else press enter ')

def t_f(var):
    if var=='T':
        return True
    else:
        return False

zerocorrection = t_f(z)
flatcorrection = t_f(f)
astrometry = t_f(a)
rm_images = t_f(r)

## Extra.
darkcorrection = False  # No cambiar. Función aún no terminada.
plots = False

# =============================================================================
# DEFINICIÓN DE FUNCIONES
# =============================================================================
imstats = lambda dat: (dat.min(), dat.max(), dat.mean(), dat.std())

def remove_file(filename):
    if Path(filename).is_file():
        os.remove(filename)
    else:
        pass
    return

#def solve_field(image, wcs_filename, ra, dec):
#    os.system("solve-field %s --new-fits %s \
#              --scale-units arcminwidth --scale-low 9 --scale-high 10 \
#              --cpulimit 60 \
#              --ra %s --dec %s --radius 1.0 \
#              --overwrite" % (image, wcs_filename, ra, dec))
#              #--overwrite --use-sextractor \
              #--sextractor-path /usr/local/bin/sextractor"%(image,wcs_filename,ra,dec))
#    return

def apply_astrometry(input_fits, output_path, api_key="wuupmjpswkcbncws"):
    """
    Sube una imagen FITS a Astrometry.net, obtiene el WCS y genera
    un archivo <original>_wcs.fit con la solución astrométrica aplicada
    en el directorio 'output_path'.

    Aplica los keywords WCS críticos uno por uno para asegurar que
    el WCS final sea válido.
    """
    input_fits = Path(input_fits)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # ------------------ LOGIN ------------------
    print(f"[Astrometry.net] Iniciando sesión...")
    r = requests.post(
        "http://nova.astrometry.net/api/login",
        data={"request-json": json.dumps({"apikey": api_key})}
    ).json()
    session = r.get("session")
    if not session:
        raise RuntimeError("No se pudo iniciar sesión en Astrometry.net")
    print(f"[Astrometry.net] Sesión iniciada OK")

    # ------------------ SUBIR IMAGEN ------------------
    print(f"[Astrometry.net] Subiendo imagen {input_fits.name}...")
    with open(input_fits, "rb") as f:
        files = {"file": f}
        payload = {"request-json": json.dumps({"session": session})}
        r = requests.post("http://nova.astrometry.net/api/upload", files=files, data=payload).json()

    if "subid" not in r:
        raise RuntimeError(f"Error al subir la imagen: {r}")
    subid = r["subid"]
    print(f"[Astrometry.net] Imagen subida (SUBID {subid})")

    # ------------------ ESPERAR JOB ------------------
    print(f"[Astrometry.net] Esperando asignación de JOB...")
    jobid = None
    while jobid is None:
        time.sleep(5)
        jobs = requests.get(f"http://nova.astrometry.net/api/submissions/{subid}").json().get("jobs", [])
        if jobs and jobs[0] is not None:
            jobid = jobs[0]
    print(f"[Astrometry.net] JOB asignado: {jobid}")

    # ------------------ ESPERAR RESOLUCIÓN ------------------
    print(f"[Astrometry.net] Resolviendo campo...")
    status = "processing"
    while status == "processing":
        time.sleep(4)
        status = requests.get(f"http://nova.astrometry.net/api/jobs/{jobid}").json().get("status")
    if status != "success":
        raise RuntimeError(f"Astrometry.net no pudo resolver la imagen (status={status})")
    print(f"[Astrometry.net] Solución WCS obtenida!")

    # ------------------ DESCARGAR HEADER WCS ------------------
    wcs_url = f"http://nova.astrometry.net/wcs_file/{jobid}"
    print(f"[Astrometry.net] Descargando solución WCS...")
    wcs_raw = requests.get(wcs_url).content

    tmp_wcs = input_fits.with_name(input_fits.stem + "_tmp_wcs.fit")
    with open(tmp_wcs, "wb") as f:
        f.write(wcs_raw)

    # ------------------ APLICAR HEADER WCS ------------------
    print(f"[Astrometry.net] Combinando WCS con la imagen original...")

    # Abrir la imagen original
    with fits.open(input_fits, mode="readonly") as orig_hdu:
        data = orig_hdu[0].data
        orig_hdr = orig_hdu[0].header.copy()

    # Abrir la solución WCS
    with fits.open(tmp_wcs) as wcs_hdu:
        wcs_hdr = wcs_hdu[0].header

    # Definir los keywords críticos
    critical_keys = [
        "CRPIX1","CRPIX2",
        "CRVAL1","CRVAL2",
        "CTYPE1","CTYPE2",
        "CUNIT1","CUNIT2",
        "CD1_1","CD1_2","CD2_1","CD2_2",
        "PC1_1","PC1_2","PC2_1","PC2_2",
        "EQUINOX"
    ]

    # Copiar solo los keywords críticos al header original
    for key in critical_keys:
        if key in wcs_hdr:
            orig_hdr[key] = wcs_hdr[key]
    orig_hdr['ASTROMET'] = ('yes', 'Astrometric solution applied')

    # Guardar la imagen final
    output_fits = output_path / f"{input_fits.stem}_wcs.fits"
    fits.writeto(output_fits, data, orig_hdr, overwrite=True)

    print(f"[Astrometry.net] Imagen final guardada en: {output_fits}")

    # Limpiar archivo temporal
    if tmp_wcs.exists():
        tmp_wcs.unlink()


    return output_fits
    
# =============================================================================
# ZEROCOMBINE
# =============================================================================
def zerocombine(mybias, zerocorrection):
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
            print("NO ZERO IMAGES TO COMBINE!")
            exit()

        ## Se rechazan imágenes que no cumplen criterios.
        bias_mean = np.array([])
        for bias in mybias:
            hdu = astropy.io.fits.open(bias)
            data = hdu[0].data
            m = stats.sigma_clipped_stats(data, sigma=2, maxiters=5)[0]
            bias_mean = np.append(bias_mean, m)
        mean_bias, std_bias = np.mean(bias_mean),np.std(bias_mean)
        print("Mean value and standard deviation of all zero images: %0.1f  %0.1f"%(mean_bias,std_bias))

        bias_list = []
        for bias in mybias:
            hdu = astropy.io.fits.open(bias)
            data = hdu[0].data
            m = stats.sigma_clipped_stats(data, sigma=2, maxiters=5)[0]
            if (m < mean_bias - 2*std_bias) or (m > mean_bias + 2*std_bias):
                print("REJECTED: %s"%bias, " MEAN: %0.3f"%m)
            else:
                bias_list.append(bias)

        print("\nZero images ready to combine: %s"%len(bias_list))
        if len(bias_list) == 0: 
            print("ALL ZERO IMAGES WERE REJECTED!!")
            print("THERE IS NOTHING MORE TO DO HERE")
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

        print("Combining zero images")
        master_bias = zero_combiner.average_combine()
        master_bias.header = meta
        master_bias_filename = 'Zero.fits'
        master_bias.write(master_bias_filename, overwrite=True)
        fits.setval(master_bias_filename, 'IMAGETYP', value='masterbias')
        fits.setval(master_bias_filename, 'FILENAME', value=master_bias_filename)

        ## Plot
        bias_min, bias_max, bias_mean, bias_std = imstats(np.asarray(master_bias))
        plt.figure(dpi=250)
        plt.imshow(master_bias, vmax=bias_mean + 4*bias_std,
                   vmin=bias_mean - 4*bias_std)
        plt.colorbar()
        plt.savefig("masterbias.pdf", dpi='figure', format='pdf')
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

def flatcombine(myflats, flatcorrection):
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
            print("NO FLAT FIELD IMAGES TO COMBINE!")
            exit()

        bands = []
        for flat in myflats:
            bands.append(getval(flat, 'FILTERS'))
        bands = np.unique(np.asarray(bands, dtype=str))
        
        ## Se rechazan imágenes que no cumplen criterios.
        flat_list = []
        for flat in myflats:
            hdu = fits.open(flat)
            data = hdu[0].data
            m = stats.sigma_clipped_stats(data, sigma=2, maxiters=5)[0]
            if m < 26000. or m > 50000.:
                print("REJECTED: %s"%flat, " MEAN: %0.3f"%m)
            else:
                flat_list.append(flat)
        
        master_flat = {}
        for band in bands:
            flat_filter = []
#            print("Combining flat images in %s-band"%band)
            for flat in flat_list:
                hdu = fits.open(flat)
                if hdu[0].header['filters'] == band:
                    meta = hdu[0].header
                    flat_filter.append(ccdproc.CCDData(data=hdu[0].data,
                                                         meta=meta, unit="adu"))
            if len(flat_filter) == 0:
                print("ALL %s-BAND FLAT FIELD IMAGES WERE REJECTED!"%band)
                exit()

            print("Combining %i flat images in %s-band"%(len(flat_filter),band))

            flat_combiner = ccdproc.Combiner(flat_filter)
            flat_combiner.sigma_clipping(low_thresh=2, high_thresh=5, func=np.ma.mean)
            scaling_func = lambda arr: 1/np.ma.average(arr)
            flat_combiner.scaling = scaling_func
            master_flat[band] = flat_combiner.average_combine()
            master_flat[band].header = meta
            master_flat_filename = "skyflat%s.fits"%band
            master_flat[band].write(master_flat_filename, overwrite=True)
            fits.setval(master_flat_filename, 'IMAGETYP', value='masterflat')
            fits.setval(master_flat_filename, 'FILENAME', value=master_flat_filename)

            del flat_filter

            ## Plot.
            f_min, f_max, f_mean, f_std = imstats(np.asarray(master_flat[band]))
            plt.figure(dpi=250)
            plt.imshow(master_flat[band], vmin=f_mean-5*f_std, vmax=f_mean+5*f_std)
            plt.colorbar()
            plt.savefig("masterflat%s.pdf"%band, dpi='figure', format='pdf')
            plt.close()
    else:
        print("Flatcombine set to false")
        return
    return master_flat

#%%
# =============================================================================
# MAIN
# =============================================================================
# SE HACEN LISTAS CON LOS DIFERENTES TIPOS DE IMÁGENES.
# =============================================================================
p = Path(".")
images = ImageFileCollection(p, keywords='*')

myfits = sorted(p.glob("*fit"))

mybias = images.files_filtered(imagetyp='zero')

mydarks = images.files_filtered(imagetyp='dark')
mydarks_cal = images.files_filtered(imagetyp='dark', calibz='subtracted bias')
mydarks = [dark for dark in mydarks if dark not in mydarks_cal]

myflats = images.files_filtered(imagetyp='flat')
myflats_cal = images.files_filtered(imagetyp='flat', calibz='subtracted bias')
myflats = [flat for flat in myflats if flat not in myflats_cal]

myimages = images.files_filtered(imagetyp='object')
myimages_cal = images.files_filtered(imagetyp='object', calibz='subtracted bias')
myimages = [image for image in myimages if image not in myimages_cal]

#%%
# =============================================================================
# EDICIÓN DEL HEADER:
# - SE AGREGA EL VALOR MEDIDO DE LA GANANCIA Y EL RUIDO DE LECTURA
# - SE AGREGA EL TIPO DE IMAGEN. (USUALMENTE YA ESTÁ HECHO)
# - SE AGREGA EL CAMPO 'FILTER' CON EL NOMBRE REAL DEL FILTRO.
# - SE AGREGA EL TIPO DE 'OBJETO' EN LAS IMÁGENES DE CALIBRACIÓN
# =============================================================================

print("Header edition")
gain = 2.0
rdnoise = 15.7
for file in myfits:
    with fits.open(file, 'update') as f:
        for hdu in f:
            hdu.header['GAIN'] = gain
            hdu.header['RDNOISE'] = rdnoise
            hdu.header['FILENAME'] = str(file)
            if 'FILTER01' in hdu.header:
                filter_str = hdu.header['FILTER01'].strip()  # Quita espacios, e.g., '(3) V'
                filter_letter = filter_str[-1] if filter_str[-1].isalpha() else 'NONE'  # Toma última letra si es alfabética (e.g., 'V')
                hdu.header['FILTERS'] = filter_letter
            #hdu.header['FILTERS'] = hdu.header['FILTER'][-1]
            hdu.header['DATE-OBS'] = Time(hdu.header['MJD-OBS'], format='mjd').iso.replace(" ","T")

            if hdu.header['IMAGETYP'] == 'zero': hdu.header['OBJECT'] = 'bias'
            elif hdu.header['IMAGETYP'] == 'dark': hdu.header['OBJECT'] = 'dark'
            elif hdu.header['IMAGETYP'] == 'flat': hdu.header['OBJECT'] = 'skyflat'
            
#%%
# =============================================================================
# SE GENERA TABLA DE DATOS CON INFORMACIÓN DEL HEADER.
# =============================================================================

keys = ['FILENAME','IMAGETYP','OBJECT','RA','DEC','EXPTIME','GAIN',
        'RDNOISE','FILTERS','DATE-OBS','TIME-OBS','MJD-OBS','AIRMASS']
values = []
filename = []
for file in myfits:
    header = getheader(file)
    values.append([header.get(key) for key in keys])

header = str(keys).replace("[","").replace("]","").replace("'","").replace(",","")
np.savetxt('images_data.txt', values, fmt="%s", header=header)
print("'images_data.txt' was created with information from the images\n")

#%%
print("Zero correction")
master_bias = zerocombine(mybias, zerocorrection)

if zerocorrection == True:
    print("Subtracting zero level from darks")
    if len(mydarks) == 0:
        print("No dark images for zero-level subtraction")
    for dark in mydarks:
        ccd = ccdproc.CCDData.read(dark, unit="adu")
        ccd = ccdproc.subtract_bias(ccd, master_bias, 
                                    add_keyword={'calibz': 'subtracted bias'})
        ccd.write("B%ss"%dark, overwrite=True)

    print("Subtracting zero level from flats")
    if len(myflats) == 0:
        print("No flat-field images for zero-level subtraction")
    for flat in myflats:
        ccd = ccdproc.CCDData.read(flat, unit="adu")
        ccd = ccdproc.subtract_bias(ccd, master_bias, 
                                    add_keyword={'calibz': 'subtracted bias'})
        ccd.write("B%ss"%flat, overwrite=True)

    print("Subtracting zero level from images")
    if len(myimages) == 0:
        print("No images for zero-level subtraction")
        exit()
    for image in myimages:
        ccd = ccdproc.CCDData.read(image, unit="adu")
        ccd = ccdproc.subtract_bias(ccd, master_bias, 
                                    add_keyword={'calibz': 'subtracted bias'})
        ccd.write("B%ss"%image, overwrite=True)

#%%
print("\nDark correction")
master_darks = darkcombine(mydarks, darkcorrection)

#%%
images = ImageFileCollection(p, keywords='*')
myflats = images.files_filtered(imagetyp='flat', calibz='subtracted bias')
myimages = images.files_filtered(imagetyp='object', calibz='subtracted bias')
myimages_cal = images.files_filtered(imagetyp='object', calibf='flat correction')
myimages = [image for image in myimages if image not in myimages_cal]

print("\nFlat-field correction")
master_flat = flatcombine(myflats, flatcorrection)

#%%
if flatcorrection == True:
    print("Applying flat-field correction to images")
    if len(myimages) == 0:
        print("No images for flat-field calibration")
        exit()
    for image in myimages:
        band = getval(image, 'filters')
        ccd = ccdproc.CCDData.read(image)
        ccd = ccdproc.flat_correct(ccd, master_flat[band], 
                                   add_keyword={'calibf': 'flat correction'})
        image_filename = "F%s"%image
        ccd.write(image_filename, overwrite=True)

#%%
# =============================================================================
# PLOTS
# =============================================================================
if plots == True:
    if not os.path.exists("plots"):
        os.system("mkdir plots")

    images = ImageFileCollection(p, keywords='*')
    myimages = images.files_filtered(imagetyp='object', 
                                     calibz='subtracted bias', 
                                     calibf='flat correction')
    myimages_wcs = images.files_filtered(imagetyp='object', 
                                         calibz='subtracted bias', 
                                         calibf='flat correction', 
                                         astromet='yes')
    myimages = [image for image in myimages if image not in myimages_wcs]

    print("\nMaking plots")
    for image in myimages:
        ccd = ccdproc.CCDData.read(image)
        #ccd = ccdproc.create_deviation(ccd, gain=gain*u.electron/u.adu , readnoise=rdnoise*u.electron)
        ma_ccd = np.ma.array(ccd.data, mask=ccd.mask)
        mean_ccd = np.ma.mean(ma_ccd)
        std_ccd = np.ma.std(ma_ccd)
            
        plt.figure(dpi=250)
        plt.imshow(ma_ccd, vmin=mean_ccd - 5*std_ccd, vmax=mean_ccd+10*std_ccd, cmap='gray', origin='lower')
        plot_name = image.replace("fits","pdf")
        plt.savefig("plots/%s"%plot_name, dpi='figure', format='pdf')
        plt.close()

#%%
# =============================================================================
# ASTROMETRY
# =============================================================================

if astrometry == True:
    print("\nAstrometry (Astrometry.net API)")

    images = ImageFileCollection(p, keywords='*')
    myimages = images.files_filtered(imagetyp='object',
                                     calibz='subtracted bias',
                                     calibf='flat correction')

    for image in myimages:
        sn = getval(image, 'OBJECT')          # nombre del objeto (carpeta)
        ra = getval(image, 'RA')
        dec = getval(image, 'DEC')

        output_dir = Path(sn)
        output_dir.mkdir(exist_ok=True)

        print(f"→ Resolviendo {image} para {sn}")

        try:
            wcs_file = apply_astrometry(
                input_fits=image,
                output_path=output_dir
            )

            fits.setval(wcs_file, 'ASTROMET', value='yes')
            print(f"✓ Guardado WCS en {wcs_file}")

        except Exception as e:
            print(f"✗ Error resolviendo {image}: {e}")

else:
    print("\nAstrometry set to false")

'''
if astrometry == True:
    print("\nAstrometry")
    print("\nPRECAUCIÓN: para realizar nuevamente la astrometría de una imagen es necesario")
    print("primero eliminar la imagen '_wcs.fits' del directorio correspondiente.")
    print("Si la imagen '_wcs.fits' ya existe, la astrometría no se hace.")
    print("")

    ## Create individual directories
    images = ImageFileCollection(p, keywords='*')
    myimages = images.files_filtered(imagetyp='object',
                                     calibf='flat correction')

    sn_names = []
    for image in myimages:
        sn_names.append(getval(image, 'OBJECT'))
    sn_names = np.unique(np.asarray(sn_names, dtype=str))

    for sn in sn_names:
        if not os.path.exists(sn): 
            os.system("mkdir %s"%sn)
            os.system("cp ../../scripts/comb.py %s"%sn)
            os.system("cp ../../scripts/aligRAF.py %s"%sn)
            #os.system("cp ../../scripts/phot_apv3.py %s"%sn)

    ## Astrometry
    images = ImageFileCollection(p, keywords='*')
    myimages = images.files_filtered(imagetyp='object', 
                                     calibz='subtracted bias', 
                                     calibf='flat correction')

    myimages_wcs = images.files_filtered(imagetyp='object', 
                                         calibz='subtracted bias', 
                                         calibf='flat correction', 
                                         astromet='yes')

    myimages = [image for image in myimages if image not in myimages_wcs]

    if len(myimages) == 0:
        print("NO FIELDS TO SOLVE")

    for image in myimages:
        wcs_filename = image.replace(".fits","_wcs.fits")
        sn = getval(image, 'OBJECT')
        if not os.path.exists("%s/%s"%(sn,wcs_filename)):
            print("Running solve-field for %s"%image)
            ra = getval(image, 'RA')
            dec = getval(image, 'DEC')

            if __name__ == '__main__':
                # Start solve_field as a process
                process = multiprocessing.Process(target=solve_field, name="solve_field", args=(image, wcs_filename, ra, dec))
                process.start()
                # Wait 10 seconds and check
                time.sleep(10)
                if os.path.exists(wcs_filename):
                    process.terminate()
                    process.join()
                else:
                    time.sleep(20)
                    process.terminate()
                    process.join()

            if os.path.exists(wcs_filename):
                fits.setval(wcs_filename, 'ASTROMET', value='yes')
                fits.setval(wcs_filename, 'FILENAME', value=wcs_filename)
            else:
                hdul = fits.open(image)
                hdul[0].writeto(wcs_filename)
                fits.setval(wcs_filename, 'ASTROMET', value='yes')
                fits.setval(wcs_filename, 'FILENAME', value=wcs_filename)
                fits.setval(wcs_filename, 'CRVAL1', value='INDEF')
                fits.setval(wcs_filename, 'CRVAL2', value='INDEF')
                fits.setval(wcs_filename, '_QUINOX', value=getval(wcs_filename, 'EQUINOX'))

            ## Remove and move output from sextractor.
            sn_name = getval(image, 'OBJECT')
            os.system("mv %s %s"%(wcs_filename,sn_name))

            prefix_filename = wcs_filename.replace("_wcs.fits","")
            remove_file("%s.axy"%prefix_filename)
            remove_file("%s.corr"%prefix_filename)
            remove_file("%s-indx.xyls"%prefix_filename)
            remove_file("%s.match"%prefix_filename)
            remove_file("%s.rdls"%prefix_filename)
            remove_file("%s.solved"%prefix_filename)
            remove_file("%s.wcs"%prefix_filename)
            remove_file("%s-indx.png"%prefix_filename)
            remove_file("%s-ngc.png"%prefix_filename)
            remove_file("%s-objs.png"%prefix_filename)

    ## Save file with unresolved fields.
    unsolved_images = []
    for sn in sn_names:
        images = ImageFileCollection(sn, keywords='CRVAL1')
        if len(os.listdir(sn)) != 0:
            unsolved_images_ind = images.files_filtered(imagetyp='object',
                                                        crval1='INDEF')
        for unsolved_image in unsolved_images_ind:
            unsolved_images.append(unsolved_image)
    np.savetxt("astrometry_unsolved_fields.txt", unsolved_images, fmt="%s")
    print("'astrometry_unsolved_fields.txt' contains all the unsolved fields")

else:
    print("\nAstrometry set to false")
'''

if rm_images == True:
    print("")
    print("Removing images")

    ## Borrar imágenes de paso para realizar corrección por flat.
    images = ImageFileCollection(p, keywords='*')
    images_to_delete = images.files_filtered(imagetyp='object', calibz='subtracted bias')
    myimages = images.files_filtered(imagetyp='object', calibf='flat correction')
    images_to_delete = [image for image in images_to_delete if image not in myimages]

    if len(images_to_delete) != 0:
        for image in images_to_delete: remove_file(image)

    ## Borrar imágenes de paso para realizar astrometría.
    images = ImageFileCollection(p, keywords='*')
    images_to_delete = images.files_filtered(imagetyp='object', calibf='flat correction')
    myimages = images.files_filtered(imagetyp='object', astromet='yes')
    images_to_delete = [image for image in images_to_delete if image not in myimages]

    if len(images_to_delete) != 0:
        for image in images_to_delete: remove_file(image)

    ## Borrar imágenes de paso para realizar corrección por flat.
    images = ImageFileCollection(p, keywords='*')
    images_to_delete = images.files_filtered(imagetyp='flat', calibz='subtracted bias')
    myimages = images.files_filtered(imagetyp='object', calibf='flat correction')
    images_to_delete = [image for image in images_to_delete if image not in myimages]

    if len(images_to_delete) != 0:
        for image in images_to_delete: remove_file(image)

