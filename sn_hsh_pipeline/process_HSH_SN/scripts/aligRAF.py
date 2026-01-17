#!/usr/bin/env python
# coding: utf-8

import numpy as np
from astropy.io import fits
from ccdproc import Combiner
from skimage.transform import estimate_transform, warp
from ccdproc import CCDData
import astropy.units as u
from astropy.wcs import WCS
from astropy.io import fits
from astropy.io.fits import PrimaryHDU, Header
from astropy.wcs import WCS

###############################################################
# 1) CENTROID: reemplazo moderno de IRAF center
###############################################################

from photutils.centroids import centroid_com

def center_e(path, im, coords_file):
    """
    Detecta centroids de estrellas dadas en coords_file usando photutils.
    Guarda un archivo .ctr como IRAF, con formato compatible con comb.py.
    """

    data = fits.getdata(im)
    stars = np.loadtxt(coords_file)

    centers = []

    for (x, y, mag) in stars:
        cx, cy = centroid_com(data[int(y)-5:int(y)+6, int(x)-5:int(x)+6])
        cx += int(x) - 5
        cy += int(y) - 5
        centers.append([cx, cy])

    out_file = im + ".ctr"
    np.savetxt(out_file, np.array(centers), fmt="%.6f")

    print("center")
    print(out_file)
    return out_file


###############################################################
# 2) GEOMAP: estimar transformación geométrica (similar a IRAF)
###############################################################

def geomap_e(path, xycoords, out_geo):
    """
    Reproduce IRAF geomap:
    Lee el archivo geomap.xy (x1,y1,x2,y2) y estima una transformación polinómica.
    Guarda la matriz en un archivo .npy.
    """

    xy = np.loadtxt(xycoords)
    x1, y1, x2, y2 = xy[:,0], xy[:,1], xy[:,2], xy[:,3]

    src = np.vstack([x1, y1]).T
    dst = np.vstack([x2, y2]).T

    # Transformación polinómica orden 2 como IRAF
    tform = estimate_transform("affine", src, dst)

    np.save(out_geo, tform.params)

    print("Geomap")
    print(out_geo)
    return out_geo


###############################################################
# 3) GEOTRAN: aplicar la transformación geométrica
###############################################################

def geotran_e(path, in_img, out_img, database, transforms):
    """
    Reproduce IRAF geotran usando skimage.warp.
    Carga la matriz guardada y reproyecta la imagen.
    """

    params = np.load(database + ".npy")

    data = fits.getdata(in_img)
    header = fits.getheader(in_img)

    # warp espera una transformada inversa → invertimos la matriz
    inv_tform = np.linalg.inv(params)

    def inv_map(coords):
        # coords = (rr, cc)
        r, c = coords[:,0], coords[:,1]
        homog = np.vstack([c, r, np.ones_like(c)])
        x_new = inv_tform @ homog
        return np.vstack([x_new[1], x_new[0]]).T

    new_data = warp(data, inv_map, preserve_range=True)

    fits.writeto(out_img, new_data.astype(data.dtype), header, overwrite=True)

    print("Geotran")
    print("output=" + out_img)
    return out_img


###############################################################
# 4) IMCOMBINE: combinación moderna con CCDPROC (sigma-clipping)
###############################################################
'''
def imcombine_e(path, input_list, out_img_ic):
    """
    Reproduce IRAF imcombine usando CCDPROC.
    Usa mediana + sigma clipping = equivalente a IRAF (y mejor).
    """

    with open(input_list) as f:
        files = [l.strip() for l in f.readlines()]

    data_list = [CCDData.read(fn, unit=u.adu) for fn in files]
    combiner = Combiner(data_list)


    # mismo rechazo sigma-clip que IRAF
    combiner.sigma_clipping(low_thresh=3, high_thresh=3)

    combined = combiner.median_combine()

# 1. Convierte header corrupto a Header válido
    if not isinstance(combined.header, Header):
        clean_header = Header(combined.header)
    else:
        clean_header = combined.header.copy()
    
    # 2. Keywords FITS obligatorios
    clean_header['SIMPLE'] = True
    clean_header['BITPIX'] = -64  # float64
    clean_header['NAXIS'] = 2
    clean_header['NAXIS1'] = combined.data.shape[1]
    clean_header['NAXIS2'] = combined.data.shape[0]
    
    # 3. WCS ESTÁNDAR (crítico para catalogo.py y Aladin)
    clean_header['CTYPE1'] = 'RA---TAN'     # Tangencial estándar
    clean_header['CTYPE2'] = 'DEC--TAN'
    clean_header['CUNIT1'] = 'deg'
    clean_header['CUNIT2'] = 'deg'
    #clean_header['CRPIX1'] = combined.data.shape[1]/2 + 1  # Centro píxel (1-based)
    #clean_header['CRPIX2'] = combined.data.shape[0]/2 + 1
    
    # 4. Copia CRVAL/CD del header ORIGINAL (de la primera imagen, que SÍ tiene WCS bueno)
    first_img = files[0]  # Primera imagen del input_list (calibrada con WCS válido)
    h0 = fits.getheader(first_img)
    for key in ['CRVAL1', 'CRVAL2', 'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2']:
        if key in h0:
            clean_header[key] = h0[key]
    
    # 5. SIP distorsión (si existe, cópiala)
    for key in h0:
        if key.startswith(('A_', 'B_', 'AP_', 'BP_')):
            clean_header[key] = h0[key]

    # 6. Keywords HSH/CASLEO (preserva trazabilidad)
    for key in ['TELESCOP', 'INSTRUME', 'FILTERS', 'EXPTIME', 'GAIN', 'OBJECT', 'DATE-OBS']:
        if key in h0:
            clean_header[key] = h0[key]

    # 7. Verifica WCS válido	
    w_test = WCS(clean_header)
    if hasattr(w_test.wcs, 'crval'):
        print(f"WCS combinada OK: centro RA={w_test.wcs.crval[0]:.5f}°, Dec={w_test.wcs.crval[1]:.5f}°")
    else:
        print("WCS creado pero sin CRVAL – revisa header original")
    
    # 8. Guarda
    primary_hdu = PrimaryHDU(data=combined.data, header=clean_header)
    primary_hdu.writeto(out_img_ic, overwrite=True) 


    print("Imcombine")
    print(out_img_ic)
    return out_img_ic
    '''

def imcombine_e(path, input_list, out_img_ic):
    """
    Combina imágenes con mediana + sigma clipping.
    Hereda el WCS COMPLETO de la imagen de referencia.
    """

    # Leer lista de archivos
    with open(input_list) as f:
        files = [l.strip() for l in f if l.strip()]

    # Leer datos
    data_list = [CCDData.read(fn, unit=u.adu) for fn in files]
    combiner = Combiner(data_list)

    # Sigma clipping tipo IRAF
    combiner.sigma_clipping(low_thresh=3, high_thresh=3)
    combined = combiner.median_combine()

    # === HEADER ===
    # Usar el header COMPLETO de la imagen de referencia
    ref_header = fits.getheader(files[0]).copy()

    # Actualizar dimensiones
    ref_header['NAXIS']  = 2
    ref_header['NAXIS1'] = combined.data.shape[1]
    ref_header['NAXIS2'] = combined.data.shape[0]

    # Metadata de combinación
    ref_header['NCOMBINE'] = len(files)

    # Limpieza mínima (opcional)
    for key in ['BZERO', 'BSCALE']:
        ref_header.pop(key, None)

    # Verificación WCS
    w = WCS(ref_header)
    assert w.is_celestial, "WCS inválido en imagen combinada"

    # Guardar
    hdu = fits.PrimaryHDU(data=combined.data, header=ref_header)
    hdu.writeto(out_img_ic, overwrite=True)

    print("Imagen combinada creada correctamente:")
    print(out_img_ic)

    return out_img_ic

    

