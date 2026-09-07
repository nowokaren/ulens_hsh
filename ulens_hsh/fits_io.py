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

warnings.filterwarnings("ignore", category=UserWarning, module="astropy")



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


# =============================================================================
# Edición y normalización de headers
# =============================================================================

def image_collection(night_dir, meta_csv=None ):
    images = ImageFileCollection(night_dir, keywords='*')
    if meta_csv is not None:
        df = images.summary.to_pandas()
        df.to_csv(meta_csv, index=False)
    return images

def update_headers(images: ImageFileCollection, gain: float, rdnoise: float, force_defaults: bool = False):
    """
    Normaliza y completa los headers de todos los archivos FITS en la colección.
    
    Agrega/normaliza:
    - GAIN, RDNOISE
    - FILENAME
    - FILTERS (a partir de FILTER o FILTER01)
    - DATE-OBS (a partir de MJD-OBS si existe)
    - OBJECT para calibraciones (bias, dark, skyflat)
    - Keywords de estado del pipeline con valores por defecto:
        CALIBZ, CALIBF, ASTROMET, OBJ_IN, NCOMBINE, INPUT_IMGS, QC_*
    
    Parámetros:
    -----------
    images : ImageFileCollection
        Colección de imágenes del directorio
    gain : float
        Valor de gain del instrumento
    rdnoise : float
        Valor de read noise del instrumento
    force_defaults : bool
        Si True, sobreescribe keywords de estado aunque ya existan
        (útil solo en la primera ejecución o para resetear)
    """
    # Convertimos a DataFrame para iterar fácilmente
    #images = ImageFileCollection(night_dir, keywords='*')
    
    summary = images.summary.to_pandas()
    
    state_keywords = {
        # Etapas de reducción
        'CALIBZ':     'no',           # 'no' → 'subtracted bias'
        'CALIBD':     'no',           # 'no' → 'subtracted dark'
        'CALIBF':     'no',           # 'no' → 'flat correction applied'
        'CRCLEAN':    'no',
        'ASTROMET':   'no',           # 'no' → 'yes' → 'failure'
        
        # Contenido científico,
        'ORIG_OBJ':   '',
        'OBJ_IN':     '',          # si el objeto objetivo está en el campo
        'OBJ_STAT':   'NOT_CHECKED',  # NOT_CHECKED, OK, CORRECTED, NO_MATCH, ERROR
        
        # Combinación
        'NCOMBINE':   '',
        'INPUT_IMGS': '',             # lista de nombres separados por coma
        
        # Quality Control (se irán llenando después)
        'QC_NSRCS':   '',
        'QC_FWHM':    '',
        'QC_ELLIP':   '',
        'QC_USE':     '',
        'QC_FLAGS':   '',
        'QC_DATE':    '',
    }

    updated_count = 0
    
    for idx, row in summary.iterrows():
        filepath = Path(images.location) / row['file']
        
        try:
            with fits.open(filepath, mode='update') as hdulist:
                hdr = hdulist[0].header

                
                # ───────────────────────────────────────────────
                # Siempre actualizamos estos (son instrumentales)
                # ───────────────────────────────────────────────
                hdr['GAIN'] = gain
                hdr['RDNOISE'] = rdnoise
                hdr['FILENAME'] = filepath.name
                
                # Filtro → estandarizamos a 'FILTERS'
                if 'FILTER01' in hdr:
                    fstr = str(hdr['FILTER01']).strip()
                    hdr['FILTERS'] = fstr[-1] if fstr[-1].isalpha() else 'NONE'
                elif 'FILTER' in hdr:
                    fstr = str(hdr['FILTER']).strip()
                    hdr['FILTERS'] = fstr[-1] if fstr[-1].isalpha() else 'NONE'
                
                # Fecha
                if 'MJD-OBS' in hdr and 'DATE-OBS' not in hdr:
                    try:
                        t = Time(float(hdr['MJD-OBS']), format='mjd')
                        hdr['DATE-OBS'] = t.iso.replace(" ", "T")
                    except:
                        pass
                
                # Calibraciones → asignamos OBJECT estándar
                if 'IMAGETYP' in hdr:
                    typ = str(hdr['IMAGETYP']).lower()
                    if 'zero' in typ or 'bias' in typ:
                        hdr['OBJECT'] = 'bias'
                    elif 'dark' in typ:
                        hdr['OBJECT'] = 'dark'
                    elif 'flat' in typ or 'skyflat' in typ:
                        hdr['OBJECT'] = 'skyflat'

                # ───────────────────────────────────────────────
                # Keywords de estado del pipeline (solo si no existen o force=True)
                # ───────────────────────────────────────────────
                for key, default in state_keywords.items():

                    if key not in hdr.keys() or force_defaults:
                        hdr[key] = default
                
                hdulist.flush()
                updated_count += 1
                
        except Exception as e:
            print(f"Error actualizando {filepath.name}: {e}")
            continue
    

    print(f"Headers actualizados en {updated_count} archivos.")
    return updated_count


# =============================================================================
# Exportación de metadatos
# =============================================================================

def match_objects_to_catalog(
    images: ImageFileCollection,
    objects_csv = None,
    max_sep_deg: float = 0.5,
    only_raw: bool = True
) -> int:
    """
    Corrige el keyword OBJECT en los headers usando un catálogo de coordenadas.
    Escribe OBJ_STAT y (si corrige) ORIG_OBJ.
    
    Retorna cuántos archivos fueron modificados.
    """
    if not objects_csv:
        return 0


    objects_df = pd.read_csv(objects_csv)
    catalog_coords = SkyCoord(
        ra=objects_df["ra_deg"].values * u.deg,
        dec=objects_df["dec_deg"].values * u.deg
    )

    summary = images.summary.to_pandas()
    updated = 0

    for _, row in tqdm(summary.iterrows(), total=len(summary), desc="Matching objects"):
        filepath = images.location / row["file"]
        hdr = fits.getheader(filepath)

        # Opcional: solo procesar raw si se pide
        if only_raw and str(hdr.get("CALIBZ", "")).lower() != "no":
            continue

        try:
            ra = hdr.get("RA")
            dec = hdr.get("DEC")
            if not (ra and dec):
                continue

            img_coord = SkyCoord(ra=ra, dec=dec, unit=(u.hourangle, u.deg))
            sep = img_coord.separation(catalog_coords)
            min_sep = sep.min().deg

            if min_sep > max_sep_deg:
                status = "NO_MATCH"
            else:
                true_name = objects_df.iloc[sep.argmin()]["objeto"]
                current = hdr.get("OBJECT", "").strip()

                with fits.open(filepath, mode="update") as hdul:
                    if current and current != true_name:
                        hdul[0].header["ORIG_OBJ"] = current
                        hdul[0].header["OBJECT"] = true_name
                        status = "CORRECTED"
                        updated += 1
                    else:
                        status = "OK" if current == true_name else "MATCHED"

                    hdul[0].header["OBJ_STAT"] = status
                    hdul.flush()

        except Exception:
            with fits.open(filepath, mode="update") as hdul:
                hdul[0].header["OBJ_STAT"] = "ERROR"
                hdul.flush()

    print(f"  Actualizados {updated} archivos con corrección de OBJECT")

    return updated




def load_dataset_objects(night_dir, images):
    night_objects = images.summary.to_pandas()["object"].unique()
    return [obj for obj in night_objects if obj not in ["bias", "skyflat", "dark"]]

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
    images,
    objects_csv,
    night_dir,
    overwrite=False
):
    """
    Para cada imagen con astrometría, chequea si el objeto observado
    cae dentro del FOV y escribe OBJ_IN = True/False en el header.
    """

    df = images.summary.to_pandas()
    df = df[df["astromet"]=="yes"]
    objects = pd.read_csv(objects_csv)

    # Diccionario rápido nombre -> (ra, dec)
    obj_coords = {
        row["objeto"]: (row["ra_deg"], row["dec_deg"])
        for _, row in objects.iterrows()
    }

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Flagging objects in FOV"):

        objname = row["object"]
        filename = row["filename"]
        if objname not in obj_coords:
            print(f"Object = '{objname}' not found in catalog. (image={filename}")
            continue

        ra_obj, dec_obj = obj_coords[objname]
        fits_path = night_dir / filename

        if not fits_path.exists():
            continue

        with fits.open(fits_path, mode="update") as hdul:
            hdr = hdul[0].header

            contains = object_in_fov(fits_path, ra_obj, dec_obj)

            hdr["obj_in"] = (
                bool(contains),
                "Target object falls inside image FOV"
            )
            hdul.flush()


            if not contains:
                print(f"   {fits_path.name}: OBJ_IN = {contains}")
        

        #print(f"   {fits_path.name}: OBJ_IN = {contains}")


def cleanup_intermediate_files(path="."):
    """
    Elimina archivos intermedios de calibración:
    - Imágenes bias-subtracted sin flat (B*.fits)
    - Imágenes flat-subtracted sin CR cleaning (FB*.fits sin _crclean)
    - Flats bias-subtracted
    
    Mantiene:
    - Raw originales
    - CR-cleaned finales (*_crclean.fits)
    """
    p = Path(path)
    images = ImageFileCollection(p, keywords='*')
    df = images.summary.to_pandas()

    # 1) Borrar imágenes bias-subtracted sin flat (B*.fits)
    print("   → Removing bias-subtracted images without flat...")
    bias_only = df[
        (df["imagetyp"] == "object") & 
        (df["calibz"] == "subtracted bias") &
        (df["calibf"] == "no") &
        (df["astromet"] == "no") &
        (df["crclean"] == "no")
    ]["file"].values
    
    for img in bias_only:
        remove_file(p / img)
        print(f"      - {img}")
    
    # 2) Borrar imágenes flat-subtracted sin CR cleaning (FB*.fits)
    print("   → Removing flat-corrected images without CR cleaning...")
    flat_no_cr = df[
        (df["imagetyp"] == "object") &
        (df["calibf"] == "flat correction") &
        (df["astromet"] == "no") &
        (df["crclean"] == "no")
    ]["file"].values
    
    for img in flat_no_cr:
        remove_file(p / img)
        print(f"      - {img}")

    # 3) Borrar flats bias-subtracted
    print("   → Removing bias-subtracted flats...")
    flats_bias = df[
        (df["imagetyp"] == "flat") & 
        (df["calibz"] == "subtracted bias")
    ]["file"].values
    
    for img in flats_bias:
        remove_file(p / img)
        print(f"      - {img}")
    
    print(f"   ✓ Cleanup complete. Removed {len(bias_only) + len(flat_no_cr) + len(flats_bias)} files")