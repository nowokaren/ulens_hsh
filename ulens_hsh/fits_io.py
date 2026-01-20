"""
fits_io.py
Módulo de I/O e infraestructura FITS:
- Descubrimiento y clasificación de imágenes
- Normalización de headers
- Exportación de metadatos
- Limpieza de archivos intermedios
"""

from pathlib import Path
import os
import numpy as np
from astropy.io import fits
from astropy.io.fits import getheader, getval
from astropy.time import Time
from ccdproc import ImageFileCollection
import warnings
from astropy.utils.exceptions import ErfaWarning

warnings.simplefilter("ignore", ErfaWarning)


from astropy.io import fits
from collections import defaultdict

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from tqdm.auto import tqdm
import astropy.units as u


# =============================================================================
# Utilidades básicas
# =============================================================================

def imstats(dat):
    return dat.min(), dat.max(), dat.mean(), dat.std()


def remove_file(filename):
    filename = Path(filename)
    if filename.is_file():
        os.remove(filename)


# =============================================================================
# Escaneo y clasificación del dataset
# =============================================================================



def scan_dataset(path="."):
    """
    Escanea un directorio y clasifica archivos FITS por tipo y estado de calibración.

    Estados:
    - raw: sin calibraciones
    - calibz: bias-subtracted
    - calibf: flat-corrected
    - astro: con solución astrométrica aplicada

    Returns
    -------
    dict
        Diccionario con listas de archivos (todos Path absolutos):
        - dir
        - all
        - bias
        - darks_raw, darks_cal
        - flats_raw, flats_cal
        - images_raw, images_cal, images_flat, images_astro
    """
    p = Path(path).resolve()
    images = ImageFileCollection(p, keywords='*')

    # Todos los FITS
    all_fits = sorted(p.glob("*.fit")) + sorted(p.glob("*.fits"))
    all_fits = [f.resolve() for f in all_fits]

    # ------------------------------------------------------------------
    # Bias
    # ------------------------------------------------------------------
    bias = [ (p / f).resolve() for f in images.files_filtered(imagetyp='zero') ]

    # ------------------------------------------------------------------
    # Darks
    # ------------------------------------------------------------------
    darks_raw = images.files_filtered(imagetyp='dark')
    darks_cal = images.files_filtered(imagetyp='dark', calibz='subtracted bias')

    darks_raw = [ (p / f).resolve() for f in darks_raw if f not in darks_cal ]
    darks_cal = [ (p / f).resolve() for f in darks_cal ]

    # ------------------------------------------------------------------
    # Flats
    # ------------------------------------------------------------------
    flats_raw = images.files_filtered(imagetyp='flat')
    flats_cal = images.files_filtered(imagetyp='flat', calibz='subtracted bias')

    flats_raw = [ (p / f).resolve() for f in flats_raw if f not in flats_cal ]
    flats_cal = [ (p / f).resolve() for f in flats_cal ]

    # ------------------------------------------------------------------
    # Science images
    # ------------------------------------------------------------------
    images_raw   = images.files_filtered(imagetyp='object')
    images_cal   = images.files_filtered(imagetyp='object', calibz='subtracted bias')
    images_flat  = images.files_filtered(imagetyp='object', calibf='flat correction')

    # Imágenes con astrometría aplicada (ASTROMET = 'yes')
    images_astro = images.files_filtered(imagetyp='object', astromet='yes')
    
    # Imágenes combinadas (NCOMBINE presente)
    images_combined = images.files_filtered(imagetyp='object', ncombine='*')
    images_combined = [ (p / f).resolve() for f in images_combined ]
    # ------------------------------------------------------------------
    # Eliminar solapamientos (jerarquía: comb  astro > flat > calib > raw)
    # ------------------------------------------------------------------
    images_raw  = [f for f in images_raw if f not in images_cal and f not in images_flat and f not in images_astro]
    images_cal  = [f for f in images_cal if f not in images_flat and f not in images_astro]
    images_flat = [f for f in images_flat if f not in images_astro]

    # Convertir a Path absolutos
    images_raw   = [ (p / f).resolve() for f in images_raw ]
    images_cal   = [ (p / f).resolve() for f in images_cal ]
    images_flat  = [ (p / f).resolve() for f in images_flat ]
    images_astro = [ (p / f).resolve() for f in images_astro ]

    return {
        "dir": p,
        "all": all_fits,
        "bias": bias,
        "darks_raw": darks_raw,
        "darks_cal": darks_cal,
        "flats_raw": flats_raw,
        "flats_cal": flats_cal,
        "images_raw": images_raw,
        "images_cal": images_cal,
        "images_flat": images_flat,
        "images_astro": images_astro,
        "images_combined": images_combined
    }

# =============================================================================
# Edición y normalización de headers
# =============================================================================

def update_headers(fits_files, gain, rdnoise):
    """
    Normaliza headers de una lista de archivos FITS:
    - Agrega GAIN y RDNOISE
    - Normaliza FILTERS
    - Convierte MJD-OBS -> DATE-OBS (ISO)
    - Asigna OBJECT en imágenes de calibración
    """
    for file in fits_files:
        with fits.open(file, 'update') as f:
            for hdu in f:
                hdu.header['GAIN'] = gain
                hdu.header['RDNOISE'] = rdnoise
                hdu.header['FILENAME'] = str(file)

                # Filtro
                if 'FILTER01' in hdu.header:
                    filter_str = hdu.header['FILTER01'].strip()
                    filter_letter = (
                        filter_str[-1] if filter_str[-1].isalpha() else 'NONE'
                    )
                    hdu.header['FILTERS'] = filter_letter

                # Fecha
                if 'MJD-OBS' in hdu.header:
                    hdu.header['DATE-OBS'] = (
                        Time(float(hdu.header['MJD-OBS']), format='mjd')
                        .iso.replace(" ", "T")
                    )

                # Tipo de objeto para calibraciones
                if 'IMAGETYP' in hdu.header:
                    if hdu.header['IMAGETYP'] == 'zero':
                        hdu.header['OBJECT'] = 'bias'
                    elif hdu.header['IMAGETYP'] == 'dark':
                        hdu.header['OBJECT'] = 'dark'
                    elif hdu.header['IMAGETYP'] == 'flat':
                        hdu.header['OBJECT'] = 'skyflat'


# =============================================================================
# Exportación de metadatos
# =============================================================================

def dataset_metadata(dataset, night_dir, output_file="images_data.csv",
                     objects_csv=None, max_sep_deg=0.5, load_changes=True):
    """
    Genera una tabla con información del header de los archivos FITS.
    Si se proporciona un catálogo de objetos (objects_csv), intenta
    corregir el OBJECT según la posición (RA, DEC) y sobrescribe el FITS
    si hay cambios.
    """

    output_path = Path(night_dir, output_file)


    keys = [
        'IMAGETYP', 'CALIBZ', 'CALIBF', 'ASTROMET', 'OBJECT', 'RA', 'DEC', 'EXPTIME', 'GAIN',
        'RDNOISE', 'FILTERS', 'DATE-OBS', 'TIME-OBS', 'MJD-OBS', 'AIRMASS',
        'FILENAME', 'OBJ_MATCH_STATUS', 'CONTAINS_OBJECT'
    ]

    if load_changes and output_path.exists():
        df_existing = pd.read_csv(output_path)
        loaded_filenames = set(df_existing["FILENAME"].astype(str))
        values = df_existing.to_dict("records")
    else:
        loaded_filenames = set()
        values = []

    # Cargar catálogo si se proporciona
    if objects_csv is not None:
        objects_df = pd.read_csv(objects_csv)
        catalog_coords = SkyCoord(
            ra=objects_df["ra_deg"].values * u.deg,
            dec=objects_df["dec_deg"].values * u.deg
        )
    else:
        objects_df = None
        catalog_coords = None



    for file in tqdm(dataset["all"], desc="   Processing FITS files"):
        file = Path(file)
        if file.name in loaded_filenames:
            continue
        hdr = getheader(file)

        # --- Corrección del objeto si corresponde ---
        if objects_df is not None:
            obj_match_status = "NOT_CHECKED"
            # Opcional: limitar solo a imágenes crudas
            if dataset is None or file in dataset.get("images_raw", []):

                ra = hdr.get("RA")
                dec = hdr.get("DEC")

                try:
                    # RA puede venir en hh:mm:ss y DEC en grados
                    img_coord = SkyCoord(ra=ra, dec=dec, unit=(u.hourangle, u.deg))
                    sep = img_coord.separation(catalog_coords)
                    min_sep = sep.min()
                    best_idx = sep.argmin()

                    if min_sep < max_sep_deg * u.deg:
                        true_name = objects_df.iloc[best_idx]["objeto"]

                        if hdr.get("OBJECT") != true_name:
                            # Guardar objeto original
                            hdr["ORIG_OBJ"] = hdr.get("OBJECT")
                            hdr["OBJECT"] = true_name
                            obj_match_status = "CORRECTED"


                        else:
                            obj_match_status = "OK"
                    else:
                        obj_match_status = "NO_MATCH"

                except Exception as e:
                    obj_match_status = "ERROR"
            with fits.open(file, mode="update") as hdul:
                if obj_match_status == "CORRECTED":
                    hdul[0].header["ORIG_OBJ"] = hdr["ORIG_OBJ"]
                    hdul[0].header["OBJECT"] = true_name
                hdul[0].header["OBJ_MATCH_STATUS"] = obj_match_status
                hdul.flush()

        # --- Guardar metadata ---
        row = {}
        for key in keys:
            if objects_df is not None and key == "OBJ_MATCH_STATUS":
                row[key] = obj_match_status
            elif key == "FILENAME":
                row[key] = file.name
            else:
                row[key] = hdr.get(key)
        values.append(row)

    #header_str = ",".join(keys)
    #np.savetxt(output_path, values, fmt="%s", delimiter=",", header=header_str)

    df_final = pd.DataFrame(values, columns=keys)
    df_final.to_csv(output_path, index=False)

    return output_path


from astropy.io.fits import getheader
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
import pandas as pd
from pathlib import Path

'''
def dataset_metadata(dataset, night_dir, output_file="images_data.csv",
                     objects_csv=None, max_sep_deg=0.5):
    """
    Genera una tabla con información del header de los archivos FITS.
    Si se proporciona un catálogo de objetos (objects_csv), intenta
    corregir el OBJECT según la posición (RA, DEC) y sobrescribe el FITS
    si hay cambios.
    """

    output_path = Path(night_dir, output_file)

    keys = [
        'IMAGETYP', 'CALIBZ', 'CALIBF', 'ASTROMET', 'OBJECT', 'RA', 'DEC', 'EXPTIME', 'GAIN',
        'RDNOISE', 'FILTERS', 'DATE-OBS', 'TIME-OBS', 'MJD-OBS', 'AIRMASS',
        'FILENAME', 'OBJ_MATCH_STATUS', 'CONTAINS_OBJECT'
    ]

    # Cargar catálogo si se proporciona
    if objects_csv is not None:
        objects_df = pd.read_csv(objects_csv)
        catalog_coords = SkyCoord(
            ra=objects_df["ra_deg"].values * u.deg,
            dec=objects_df["dec_deg"].values * u.deg
        )
    else:
        objects_df = None
        catalog_coords = None

    values = []

    loaded_ds = pd.read_csv(Path(night_dir, "images_data.csv"))
    for file in dataset["all"]:
        file = Path(file)
        hdr = getheader(file)

        # --- Corrección del objeto si corresponde ---
        if objects_df is not None:
            obj_match_status = "NOT_CHECKED"
            # Opcional: limitar solo a imágenes crudas
            if dataset is None or file in dataset.get("images_raw", []):

                ra = hdr.get("RA")
                dec = hdr.get("DEC")

                try:
                    # RA puede venir en hh:mm:ss y DEC en grados
                    img_coord = SkyCoord(ra=ra, dec=dec, unit=(u.hourangle, u.deg))
                    sep = img_coord.separation(catalog_coords)
                    min_sep = sep.min()
                    best_idx = sep.argmin()

                    if min_sep < max_sep_deg * u.deg:
                        true_name = objects_df.iloc[best_idx]["objeto"]

                        if hdr.get("OBJECT") != true_name:
                            # Guardar objeto original
                            hdr["ORIG_OBJ"] = hdr.get("OBJECT")
                            hdr["OBJECT"] = true_name
                            obj_match_status = "CORRECTED"


                        else:
                            obj_match_status = "OK"
                    else:
                        obj_match_status = "NO_MATCH"

                except Exception as e:
                    obj_match_status = "ERROR"
            with fits.open(file, mode="update") as hdul:
                if obj_match_status == "CORRECTED":
                    hdul[0].header["ORIG_OBJ"] = hdr["ORIG_OBJ"]
                    hdul[0].header["OBJECT"] = true_name
                hdul[0].header["OBJ_MATCH_STATUS"] = obj_match_status
                hdul.flush()

        # --- Guardar metadata ---
        row = []
        for key in keys:
            if objects_df is not None and key == "OBJ_MATCH_STATUS":
                row.append(obj_match_status)
            elif key == "FILENAME":
                row.append(file.name)
            else:
                row.append(hdr.get(key))

        values.append(row)

    header_str = ",".join(keys)
    np.savetxt(output_path, values, fmt="%s", delimiter=",", header=header_str)

    return output_path
'''

def load_dataset_objects(night_dir, output_file):
    ds = pd.read_csv(Path(night_dir, output_file), usecols=["OBJECT"])
    return [obj for obj in ds["OBJECT"].unique() if obj not in ["bias", "skyflat"]]

# =============================================================================
# Limpieza de archivos intermedios
# =============================================================================

def cleanup_intermediate_files(path="."):
    """
    Elimina archivos intermedios de calibración:
    - Imágenes bias-subtracted sin flat
    - Imágenes flat-subtracted sin astrometría
    - Flats bias-subtracted
    """
    p = Path(path)
    images = ImageFileCollection(p, keywords='*')

    # 1) Borrar imágenes bias-subtracted sin flat
    images_to_delete = images.files_filtered(imagetyp='object', calibz='subtracted bias')
    myimages = images.files_filtered(imagetyp='object', calibf='flat correction')
    images_to_delete = [img for img in images_to_delete if img not in myimages]
    for image in images_to_delete:
        remove_file(image)

    # 2) Borrar imágenes flat-subtracted sin astrometría
    images = ImageFileCollection(p, keywords='*')
    images_to_delete = images.files_filtered(imagetyp='object', calibf='flat correction')
    myimages = images.files_filtered(imagetyp='object', astromet='yes')
    images_to_delete = [img for img in images_to_delete if img not in myimages]
    for image in images_to_delete:
        remove_file(image)

    # 3) Borrar flats bias-subtracted
    images = ImageFileCollection(p, keywords='*')
    images_to_delete = images.files_filtered(imagetyp='flat', calibz='subtracted bias')
    for image in images_to_delete:
        remove_file(image)


def get_image_fov_arcmin(wcs_header, nx, ny):
    """
    Calcula el tamaño del CCD en arcmin usando el WCS.
    Devuelve: width_arcmin, height_arcmin, diag_arcmin
    """

    w = WCS(wcs_header)

    # cuatro esquinas del CCD (píxeles)
    corners_px = np.array([
        [0, 0],
        [nx-1, 0],
        [0, ny-1],
        [nx-1, ny-1]
    ])

    # convertir a RA/DEC
    ras, decs = w.pixel_to_world_values(corners_px[:,0], corners_px[:,1])

    # convertir diferencias a arcmin
    width_deg  = np.abs(ras[1] - ras[0]) * np.cos(np.radians(np.mean(decs)))  # corrección por cos(dec)
    height_deg = np.abs(decs[2] - decs[0])
    diag_deg   = np.sqrt(width_deg**2 + height_deg**2)

    return width_deg * 60, height_deg * 60, diag_deg * 60

#Check if a given pair of XY coordinates are in the image.
def is_in_image(x, y, nx, ny, margin=0):
    return (margin <= x < nx - margin) and (margin <= y < ny - margin)


#Convert RADEC[deg] to XY
def radec2xy(img,stars):
    hdulist   = fits.open(img)
    data = hdulist[0].data
    ny, nx = data.shape
    w         = wcs.WCS(hdulist[0].header)
    wcs_coord = np.array(stars[[0,1]])
    pix_coord = w.wcs_world2pix(wcs_coord,0)
    xy = []
    outside_coords = []
    for i in range(len(pix_coord)):
        xpix, ypix = pix_coord[i][:2]  # aseguramos solo dos valores
        '''
        if is_in_image(xpix, ypix):
            xy.append((float(xpix), float(ypix)))
        else:
            print(f'WARNING: Star {i + 1} is not in image {img}')
        '''
        if is_in_image(xpix, ypix, nx, ny, margin=5):
            xy.append((float(xpix), float(ypix)))
        else:
            outside_coords.append((i+1, xpix, ypix))
            # print(f'WARNING: Star {i + 1} at x={xpix:.2f}, y={ypix:.2f} outside {nx}x{ny}')
    print(f"WARNING: {len(outside_coords)} stars outside {nx}x{ny}. {len(pix_coord)-len(outside_coords)} used.")
    return xy

#Calculate mean FWHM in X and Y axis for each image.
def FWHM_im(image, center, box=15, fwhm_default=4.0):
    """
    Mide la FWHM alrededor de un centro estimado.
    Devuelve: (amp, x_center, y_center, fwhm)
    Si la subimagen está vacía o solo contiene NaN, devuelve fwhm por defecto.
    """

    if center is None:
        return (0.0, 0.0, 0.0, fwhm_default)

    x0, y0 = center
    if np.isnan(x0) or np.isnan(y0):
        return (0.0, 0.0, 0.0, fwhm_default)

    try:
        data = fits.getdata(image)
    except Exception as e:
        print(f"❌ FWHM_im: no se pudo leer {image}: {e}")
        return (0.0, x0, y0, fwhm_default)

    ny, nx = data.shape
    x1 = int(max(0, x0 - box))
    x2 = int(min(nx, x0 + box))
    y1 = int(max(0, y0 - box))
    y2 = int(min(ny, y0 + box))

    sub = data[y1:y2, x1:x2]

    if sub.size == 0 or np.all(np.isnan(sub)):
        # Retornar FWHM por defecto
        return (0.0, x0, y0, fwhm_default)

    # Amplitud simple
    amp = np.nanmax(sub) - np.nanmin(sub)

    # Centro aproximado
    ycntr, xcntr = np.unravel_index(np.nanargmax(sub), sub.shape)
    xcntr += x1
    ycntr += y1

    # FWHM aproximado: promedio de ancho en X e Y a mitad de altura
    fwhm_x = np.sum(sub[int(ycntr-y1), :] > 0.5 * np.nanmax(sub))
    fwhm_y = np.sum(sub[:, int(xcntr-x1)] > 0.5 * np.nanmax(sub))
    fwhm = np.mean([fwhm_x, fwhm_y])

    # Asegurarse de que fwhm no sea cero
    if fwhm <= 0:
        fwhm = fwhm_default

    return (amp, xcntr, ycntr, fwhm)

#Calculate backgroud value and mean flux for each (bkg subtracted) image
def bg_flux_im(filename,xy,fwhm):
    data = fits.getdata(filename)
    #Image background
    sigma_clip = SigmaClip(sigma=3.)
    bkg = ModeEstimatorBackground(sigma_clip=sigma_clip)
    bkg_value  = bkg.calc_background(data)
    #Subtract backgrounf of each image
    data_b     = data-bkg_value
    #Aperture photometry
    aperture_r = fwhm
    aperture   = CircularAperture(xy,r=aperture_r)
    phot_table = aperture_photometry(data_b, aperture)
    annulus_r_in  = 10.
    annulus_r_out = 20.
    annulus_aperture = CircularAnnulus(xy, r_in=annulus_r_in,
                                       r_out=annulus_r_out)
    apers = [aperture, annulus_aperture]
    phot_table_bg = aperture_photometry(data_b, apers)
    bkg_mean = phot_table_bg['aperture_sum_1'] / annulus_aperture.area 
    bkg_sum = bkg_mean * aperture.area
    final_sum = phot_table_bg['aperture_sum_0'] - bkg_sum
    phot_table_bg['final_sum'] = final_sum
    #Image mean flux value 
    flux_value=np.nanmean(np.array(phot_table_bg['final_sum']))
    return(flux_value,bkg_value)

from pathlib import Path
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


# -----------------------------------------------------------------------------
# Utilidades
# -----------------------------------------------------------------------------

def read_fits_data(path):
    with fits.open(path) as hdul:
        return hdul[0].data.astype(float)

def img_stats(img):
    return {
        "mean": np.mean(img),
        "median": np.median(img),
        "std": np.std(img),
        "p5": np.percentile(img, 5),
        "p95": np.percentile(img, 95),
        "neg_frac": np.mean(img < 0)
    }




def identify_objects(image_paths, object_key="OBJECT"):
    """
    Identifica los objetos observados a partir del header FITS
    y agrupa las imágenes por objeto.

    Parameters
    ----------
    image_paths : list[Path]
        Lista de imágenes científicas.
    object_key : str
        Keyword FITS donde figura el nombre del objeto.

    Returns
    -------
    dict
        { objeto1: [img1, img2, ...],
          objeto2: [img3, img4, ...], ... }
    """
    objects = defaultdict(list)

    for img in image_paths:
        try:
            with fits.open(img) as hdul:
                hdr = hdul[0].header
                obj = hdr.get(object_key)

            if obj is None:
                # Fallback: usar nombre de archivo
                obj = img.stem

        except Exception:
            obj = img.stem

        objects[obj].append(img)

    return dict(objects)



from astropy.wcs import WCS
from astropy.io import fits
import astropy.units as u

def object_in_fov(fits_path, ra_obj, dec_obj):
    """
    ra_obj, dec_obj in degrees
    """
    with fits.open(fits_path) as hdul:
        header = hdul[0].header
        data = hdul[0].data
        wcs = WCS(header)

    ny, nx = data.shape

    obj_coord = SkyCoord(ra_obj*u.deg, dec_obj*u.deg)
    x, y = wcs.world_to_pixel(obj_coord)

    return (0 <= x < nx) and (0 <= y < ny), x, y


import pandas as pd
from pathlib import Path
from astropy.io import fits

def flag_object_in_fov(
    metadata_csv,
    objects_csv,
    night_dir,
    overwrite=False
):
    """
    Para cada imagen con astrometría, chequea si el objeto observado
    cae dentro del FOV y escribe CONTAINS_OBJECT = True/False en el header.
    """

    meta = pd.read_csv(metadata_csv)
    meta = meta[meta["ASTROMET"]=="yes"]
    objects = pd.read_csv(objects_csv)

    # Diccionario rápido nombre -> (ra, dec)
    obj_coords = {
        row["objeto"]: (row["ra_deg"], row["dec_deg"])
        for _, row in objects.iterrows()
    }

    for _, row in tqdm(meta.iterrows(), desc="Flagging objects in FOV"):

        objname = row["OBJECT"]
        if objname not in obj_coords:
            continue

        ra_obj, dec_obj = obj_coords[objname]
        fits_path = night_dir / row["FILENAME"]

        if not fits_path.exists():
            continue

        with fits.open(fits_path, mode="update") as hdul:
            hdr = hdul[0].header

            contains = object_in_fov(fits_path, ra_obj, dec_obj)

            hdr["CONTAINS_OBJECT"] = (
                bool(contains),
                "Target object falls inside image FOV"
            )
            hdul.flush()


            if not contains:
                print(f"   {fits_path.name}: CONTAINS_OBJECT = {contains}")
        

        #print(f"   {fits_path.name}: CONTAINS_OBJECT = {contains}")
