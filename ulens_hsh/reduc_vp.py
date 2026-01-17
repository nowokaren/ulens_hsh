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

