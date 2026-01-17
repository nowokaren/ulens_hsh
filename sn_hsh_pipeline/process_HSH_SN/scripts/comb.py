from __future__ import print_function
import os
import io
import time
import imexam
import argparse
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy import wcs
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from scipy.spatial import distance
from datetime import datetime as dt
#from photutils import IRAFStarFinder
from imexam.math_helper import gfwhm
from reproject import reproject_interp
from imexam.imexamine import Imexamine
from ccdproc import ImageFileCollection
from astropy.visualization import simple_norm
from photutils import ModeEstimatorBackground
from reproject.mosaicking import reproject_and_coadd
from astropy.stats import SigmaClip, sigma_clipped_stats
from photutils import aperture_photometry,CircularAperture,CircularAnnulus
import requests
from astropy.io.votable import parse
import io
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.coordinates import SkyCoord, search_around_sky
from astroquery.vizier import Vizier
from aligRAF import center_e,geomap_e,geotran_e,imcombine_e

use_ds9=False
###############################################################################
# Auxiliar functions
###############################################################################

# AUTOMATIC .COO GENERATION USING GAIA + WCS
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
from astropy import units as u

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


def generate_ref_psf_coo(obj_name, ra_center, dec_center, img_dir, image_files, 
                         fov_frac=0.3, min_mag=9, max_mag=12, plot=False):
    with fits.open(os.path.join(img_dir, image_files[0])) as hdul:
        w_ref = WCS(hdul[0].header)
        ny, nx = hdul[0].data.shape
    
    corners_pix = np.array([[0,0],[nx,0],[0,ny],[nx,ny]])
    ra_c, dec_c = w_ref.pixel_to_world_values(corners_pix[:,0], corners_pix[:,1])
    coord_corners = SkyCoord(ra_c*u.deg, dec_c*u.deg)
    
    _, _, sep2d, _ = search_around_sky(coord_corners, coord_corners, 180*u.deg)
    fov_diag_arcmin = sep2d.max().to(u.arcmin).value
    search_radius_arcmin = fov_diag_arcmin * fov_frac   # radio ≈ diagonal/2
    
    print(f"FOV diagonal: {fov_diag_arcmin:.1f}' → Radio búsqueda: {search_radius_arcmin:.1f}'")
    
    
    coord = SkyCoord(ra_center*u.deg, dec_center*u.deg)
    radius = search_radius_arcmin * u.arcmin
    
    catalogs = {
        "Gaia": {
            "cat": "I/355/gaiadr3",
            "ra": "RA_ICRS",
            "dec": "DE_ICRS",
            "mag": "Gmag"
        },
        "2MASS": {
            "cat": "II/246/out",
            "ra": "_RAJ2000",
            "dec": "_DEJ2000",
            "mag": "Jmag"
        },
        "APASS": {
            "cat": "II/336/apass9",
            "ra": "_RAJ2000",
            "dec": "_DEJ2000",
            "mag": "Vmag"
        }
    }
    
    refs = []
    
    for name, c in catalogs.items():
        print(f"\nBuscando en {name}...")
        mag_limit = f"{min_mag}..{max_mag}"
        try:
            v = Vizier(columns=[c["ra"], c["dec"], c["mag"]],
                       column_filters={c["mag"]: mag_limit}, row_limit=10000)
            r = v.query_region(coord, radius=radius, catalog=c["cat"])
    
            if not r or len(r[0]) == 0:
                print("  → vacío")
                continue
    
            t = r[0]
            ra  = t[c["ra"]].filled(np.nan)
            dec = t[c["dec"]].filled(np.nan)
            mag = t[c["mag"]].filled(np.nan)
    
            ok = ~np.isnan(ra) & ~np.isnan(dec) & ~np.isnan(mag)
            ra, dec, mag = ra[ok], dec[ok], mag[ok]
    
            # excluir objeto
            sep = coord.separation(SkyCoord(ra, dec)).arcmin
            mask = sep > 0.5
            ra, dec, mag = ra[mask], dec[mask], mag[mask]
    
            idx = np.argsort(mag)
    
            for i in idx:
                refs.append((ra[i], dec[i], mag[i], name))
    
            print(f"  → {len(idx)} refs")
    
        except Exception as e:
            print(f"  Error: {e}")
    
    refs = np.array(refs, dtype=object)
    
    if len(refs) == 0:
        raise RuntimeError("No se encontraron estrellas de referencia")
    
    if plot:
        colors = {"Gaia": "yellow", "2MASS": "orange", "APASS": "lime"}
        
        fig, axes = plt.subplots(2, 3, figsize=(18,12))
        axes = axes.flatten()
        
        for ax, img in zip(axes, image_files):
        
            with fits.open(os.path.join(img_dir, img)) as hdul:
                data = hdul[0].data
                w = WCS(hdul[0].header)
        
            ax.imshow(data, cmap="gray",
                      norm=simple_norm(data, "sqrt", percent=99.5),
                      origin="lower")
        
            for cat in np.unique(refs[:,3]):
                m = refs[:,3] == cat
                x, y = w.world_to_pixel_values(refs[m,0].astype(float),
                                               refs[m,1].astype(float))
                ax.scatter(x, y, s=140, facecolors='none',
                           edgecolors=colors[cat], lw=2, label=cat)
        
            xo, yo = w.world_to_pixel_values(ra_center, dec_center)
            ax.plot(xo, yo, 'o', ms=30, mew=2, color='cyan', fillstyle='none')
        
            ax.set_title(img)
            ax.legend(fontsize=8)
        
        plt.suptitle(f"{obj_name} — estrellas de referencia", fontsize=16)
        plt.tight_layout()
        plt.savefig(objname+filt+'_refpsf.png')
    
    outdir = Path(mpath.parents[1]) / "objetos" / objname
    outdir.mkdir(parents=True, exist_ok=True)

    outfile = outdir / f"{objname}_refpsf.coo"

    np.savetxt(
        outfile,
        refs[:,:3].astype(float),
        fmt="%.6f %.6f %.3f"
    )

    #np.savetxt(str(mpath.parents[1])+'/objetos/'+objname+'/'+objname+'_refpsf.coo',
    #           refs[:,:3].astype(float),
    #           fmt="%.6f %.6f %.3f")
    
    print("\nArchivo .coo generado correctamente.")
    return True



  

#Delete files from possible previous run from the list
def remove(filt):
    os.system('rm -f *'+filt.lower()+'*_trans0.fits')
    os.system('rm -f *'+filt.lower()+'*_trans.fits')
    os.system('rm -f *'+filt.lower()+'*_scaled.fits')
    os.system('rm -f *'+filt.lower()+'_comb.fits')
    os.system('rm -f *'+filt.lower()+'_comb2.fits')
    print()
    return()

#Show images in ds9
def showinds9(img, mark_region='no', positions=[], use_ds9=False):
    """
    Muestra imagen en DS9 solo si use_ds9=True
    """
    if use_ds9:
        viewer = imexam.connect("alds9")
        viewer.frame(1)
        viewer.load_fits(img)
        viewer.zoomtofit()
        viewer.scale()
        if mark_region == 'yes' and len(positions) > 0:
            viewer.mark_region_from_array(positions, size=15)
        viewer.close()
    return()

#Check if a given pair of XY coordinates are in the image.
def is_in_image(x, y):
    out = bool
    if x > 10 and x < 1014 and y > 10 and y < 1014:
        out = True
    else:
        out = False
    return out
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

    '''
    xy=[]
    for i in range(len(pix_coord)):
        if is_in_image(pix_coord[i][0],pix_coord[i][1]):
            xy.append((pix_coord[i][0],pix_coord[i][1]))
        else:
            print(f'WARNING: Star {i + 1} is not in image {img}')
    return(xy)
    '''



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



#Interact with images to select stars in reference image in case of no WCS
def select_xy(img):
    out_file = img + '.coo'

    import subprocess
    import time
    import imexam

    # 1. Verificar sesiones activas
    chkds9 = imexam.list_active_ds9()
    nameds9 = [c[0] for c in chkds9.values()]

    # 2. Si no existe alds9, lanzarlo
    if 'alds9' not in nameds9:
        subprocess.Popen(['ds9', '-title', 'alds9'])
        time.sleep(2.5)   # IMPORTANTE: ds9 tarda en registrar el socket

    # 3. Re-chequear (clave)
    chkds9 = imexam.list_active_ds9()
    nameds9 = [c[0] for c in chkds9.values()]

    if 'alds9' not in nameds9:
        raise RuntimeError("DS9 no pudo iniciarse o no registró la sesión 'alds9'")

    print('#'*45)
    print('Interaction with ds9 started.')
    print('Use "a" to select reference stars: ')
    print('#'*45)
    print()

    viewer = imexam.connect("alds9")
    viewer.frame(1)
    viewer.load_fits(img)
    viewer.zoomtofit()
    viewer.scale()
    
    #Remove auxiliary files
        # Verificar si DS9 está activo
    chkds9 = imexam.list_active_ds9()
    nameds9 = [c[0] for c in chkds9.values()]

    if 'alds9' not in nameds9:
        subprocess.Popen(['ds9', '-title', 'alds9'])
        time.sleep(1.5)

    print('#'*45)
    print('Interaction with ds9 started.')
    print('Use "a" to select reference stars: ')
    print('#'*45)
    print()

    viewer = imexam.connect("alds9")
    viewer.frame(1)
    viewer.load_fits(img)
    viewer.zoomtofit()
    viewer.scale()
    os.system('rm -f fwhm1.imx')
    os.system('rm -f fwhm11.imx')
    print('#'*45)
    print('Interaction with ds9 started.')
    print('Use "a" to select reference stars: ')
    print('#'*45)
    print()
    #time.sleep(1.5)
    #Connect to ds9 in order to select stars
    viewer=imexam.connect("alds9") 
    viewer.frame(1) 
    viewer.load_fits(img)
    viewer.zoomtofit() 
    viewer.scale()
    #Save the same data that appears on screen
    viewer.setlog(filename='fwhm1.imx')
    #Define radius where to calculate the position
    viewer.set_plot_pars('a','radius',10)
    viewer.imexam()
    #Close log
    viewer.setlog(on=False) 
    viewer.close()
    #Copy log file to erase empty lines    
    with open(str(mpath)+'/fwhm1.imx') as infile,\
         open('fwhm11.imx', 'w') as outfile:
        for line in infile:
            if not line.strip(): continue  # skip the empty line
            outfile.write(line)  # non-empty line. Write it to output
    #Read x,y, fwhmx & fwhmy of the selected stars
    log_f=np.loadtxt('fwhm11.imx',dtype='str',delimiter='\t',usecols=(0,))
    x=[]
    y=[]
    fwhmx=[]
    fwhmy=[]
    for j in range(2,len(log_f),3):
        x.append(float(log_f[j][:6]))
        y.append(float(log_f[j][6:30]))
        fwhmx.append(float(log_f[j][-9:-5]))
        fwhmy.append(float(log_f[j][-4:]))   
    xy=[]
    for i in range(len(x)):
        xy.append((x[i],y[i]))
    #Show the position of the selected stars on ds9
    print(img+' stars marked in ds9.\nCenters of given stars in image:')
    showinds9(img, mark_region = 'yes', positions = xy, use_ds9=use_ds9)
    #Write positions in file
    xy_w= str(np.array(xy))
    xy_w=xy_w.replace("["," ")
    xy_w=xy_w.replace("]"," ")
    aux=open(out_file,'w+')
    aux.write(xy_w)
    aux.close()
    print(out_file+' file created with the xy position of the given stars')
    return(out_file)

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

###############################################################################
# Alineation and combination functions
###############################################################################
#Alingment without wcs
def shift(imfiles):
    #Read image headers
    hdu_list = [fits.open(imfiles[0][i])[0] for i in range(len(imfiles))]
    #Save final files
    final_files=[]
    #Selecte reference image
    if len(hdu_list)>1:
        ref_index=int(len(hdu_list)/2)
        ref_img=imfiles[0][ref_index]
    else:
        ref_index=0
        ref_img=imfiles[0][ref_index]
    print()
    print('#'*45)
    counter=len(imfiles)
    print('Number of images to interact with: ')
    print(counter)
    print()   
    '''     
    #Select coordinates in reference image 
    coordfile1=select_xy(ref_img)
    #Center reference image
    out_cen1=center_e(str(mpath.parents[2])+'/scripts/',ref_img,coordfile1)
    '''
    ref_coo_file = str(mpath.parents[1])+'/objetos/'+objname+'/'+objname+'_refpsf.coo'
    if os.path.exists(ref_coo_file):
        print(f"Using existing reference star file: {ref_coo_file}")
        coordfile1 = ref_coo_file
    else:
        print("Reference file not found. Please select stars manually in DS9.")
        coordfile1 = select_xy(ref_img)

    out_cen1 = center_e(str(mpath.parents[2])+'/scripts/', ref_img, coordfile1)

    
    #Create input for geomap
    skip=np.arange(0,500,1)
    xyc1=pd.read_csv(out_cen1,delim_whitespace=True,comment='#',
                     header=None,skiprows=skip[0::2])
    time.sleep(0.5)
    for i in range(len(imfiles[0])):
        if i!=ref_index:
            print('#'*45)
            counter-=1
            print('Remaining images to interact with ',counter)
            #Select coordinates
            coordfile2 = coordfile1 # coordfile2=select_xy(imfiles[0][i])
            print('Coordinate files: ')
            print(coordfile1)
            print(coordfile2)
            #Center    
            out_cen2=center_e(str(mpath.parents[2])+'/scripts/',
                              imfiles[0][i],coordfile2)
            #Create input file for geomap  
            xyc2=pd.read_csv(out_cen2,delim_whitespace=True,comment='#',
                             header=None,skiprows=skip[0::2])
            xyc=pd.DataFrame([xyc1[0],xyc1[1],xyc2[0],xyc2[1]]).transpose()
            xyc_w= str(np.array(xyc))
            xyc_w=xyc_w.replace("["," ")
            xyc_w=xyc_w.replace("]"," ")
            aux=open('geomap.xy','w+')
            aux.write(xyc_w)
            aux.close()
            #Geomap      
            geomap_e(str(mpath.parents[2])+'/scripts/','geomap.xy','out.geo')
            #Geotran
            geotran_e(str(mpath.parents[2])+'/scripts/',imfiles[0][i],
                      imfiles[0][i][:-5]+'_trans.fits','out.geo','geomap.xy')
            final_files.append(imfiles[0][i][:-5]+'_trans.fits')
            print('Transformed image = '+imfiles[0][i][:-5]+'_trans0.fits')
            print()
        elif i==ref_index:
            final_files.append(imfiles[0][i])
    return(final_files)

def load_target_coordinates(imdir):
    """
    Lee automáticamente la RA/Dec del objeto de interés desde:
    ../../objetos/<OBJETO>/<OBJETO>.coo

    Acepta dos formatos:
    - decimal:     RA DEC  (en grados)
    - sexagesimal: HH:MM:SS  DD:MM:SS

    Devuelve RA, Dec en grados (floats).
    """

    import numpy as np
    from pathlib import Path
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    # nombre del objeto tomado del path (ej: "OGLE-2025-BLG-0397")
    objname = Path(imdir).resolve().name
    
    '''
    # ruta al archivo .coo del objeto
    coo_path = Path(imdir).resolve().parents[1] / "objetos" / objname / f"{objname}.coo"

    if not coo_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo del objeto: {coo_path}")

    # leemos TODAS las líneas de texto del archivo
    with open(coo_path, "r") as f:
        line = f.readline().strip()

    # separo por espacios
    parts = line.split()

    if len(parts) != 2:
        raise ValueError(f"Formato inválido en {coo_path}: debe contener 'RA DEC'")
        
    ra_str, dec_str = parts
    '''
    posiciones = pd.read_csv("../../objetos.csv").query(f"objeto == '{objname}'")
    ra_str, dec_str = posiciones[["ra_deg", "dec_deg"]].values[0]
    

    # CASO 1 — decimal (float)
    try:
        ra = float(ra_str)
        dec = float(dec_str)
        return ra, dec
    except ValueError:
        pass  # no eran floats → probamos sexagesimal

    # CASO 2 — sexagesimal → usar astropy
    try:
        coord = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))
        return float(coord.ra.deg), float(coord.dec.deg)
    except Exception as e:
        raise ValueError(f"No se pudo interpretar RA/DEC en {coo_path}: {e}")



#Alingment with some wcs
def algn_with_some_wcs(imfiles,crv, objname):
    final_files=[]
    #Make a list of the images with WCS
    wcs_img=[imfiles[0][j] for j in range(len(crv)) if crv[j] != 'INDEF']
    print()
    print('#'*45)
    counter=len(imfiles)-len(wcs_img)
    print('Number of images to interact with: ')
    print(counter)
    print()
    #Select reference image
    ref_img=wcs_img[0]
    ref_hdu=[fits.open(wcs_img[0])[0]]
    '''
    #Select coordinates in reference image 
    coordfile1=select_xy(ref_img)
    #Center reference image
    out_cen1=center_e(str(mpath.parents[2])+'/scripts/',ref_img,coordfile1)
    '''

    # ================================
    # AUTO-GENERATE .COO IF REQUESTED
    # ================================
    auto = "A" #input("\nPress A for automatic reference-star .coo using Gaia, else ENTER: ")
    
    if auto.upper() == "A":
        print("\nUsando generación automática con Gaia...")
    
    # Calcula radio ~mitad FOV
    hdul = fits.open(imfiles[0][0])
    ny, nx = hdul[0].data.shape
    try:
        cdelt = np.mean(np.abs([hdul[0].header['CDELT1'], hdul[0].header['CDELT2']])) * 60  # arcmin/pix
    except:
        cdelt = 0.53  # fallback HSH arcsec/pix
    hdul.close()
    search_radius_arcmin = (min(nx, ny) * cdelt / 60) / 2
    
    imdir = os.getcwd()
    ra_obj, dec_obj = load_target_coordinates(imdir)
    print(f"Target: RA={ra_obj:.6f} Dec={dec_obj:.6f}")
    
    ref_coo_file = str(mpath.parents[1])+'/objetos/'+objname+'/'+objname+'_refpsf.coo'	
    if not os.path.exists(ref_coo_file):
        print(f"Usando refs automático: {ref_coo_file}")
        try:
            stars_auto = generate_ref_psf_coo(objname, ra_obj, dec_obj, imdir, lista, 
                         fov_frac=0.3, min_mag=9, max_mag=12, plot=True)
            coordfile1 = ref_coo_file
        except:
            print("Generación automática falló. Selecciona manual en DS9.")
            ref_img = wcs_img[0] if 'wcs_img' in locals() else imfiles[0][0]
            coordfile1 = select_xy(ref_img)
    else:
        print("Archivo de objetos de referncia para alineación existente. Eliminar si se quiere recargar")
        coordfile1 = ref_coo_file
    
    out_cen1 = center_e(str(mpath.parents[2])+'/scripts/', ref_img if 'ref_img' in locals() else wcs_img[0], coordfile1)

    #Create input for geomap
    skip=np.arange(0,500,1)
    xyc1=pd.read_csv(out_cen1,delim_whitespace=True,comment='#',
                     header=None,skiprows=skip[0::2])
    #Save WCS info to edit shifted images header
    #cv1_ref=hdu[0].header['CRVAL1'])
    time.sleep(0.5)
    for i in range(len(imfiles[0])):
        if imfiles[0][i] not in wcs_img:
            print('#'*45)
            counter-=1
            print('Remaining images to interact with ',counter)
            #Select coordinates
            coordfile2 = coordfile1 #coordfile2=select_xy(imfiles[0][i])
            print('Coordinate files: ')
            print(coordfile1)
            print(coordfile2)
            #Center    
            out_cen2=center_e(str(mpath.parents[2])+'/scripts/',imfiles[0][i],
                              coordfile2)
            #Create input file for geomap  
            xyc2=pd.read_csv(out_cen2,delim_whitespace=True,comment='#',
                             header=None,skiprows=skip[0::2])
            xyc=pd.DataFrame([xyc1[0],xyc1[1],xyc2[0],xyc2[1]]).transpose()
            xyc_w= str(np.array(xyc))
            xyc_w=xyc_w.replace("["," ")
            xyc_w=xyc_w.replace("]"," ")
            aux=open('geomap.xy','w+')
            aux.write(xyc_w)
            aux.close()
            #Geomap      
            geomap_e(str(mpath.parents[2])+'/scripts/','geomap.xy','out.geo')
            #Geotran
            geotran_e(str(mpath.parents[2])+'/scripts/',imfiles[0][i],
                      imfiles[0][i][:-5]+'_trans0.fits','out.geo','geomap.xy')
            #Edit header of shifted images
            orgnl_img=[fits.open(imfiles[0][i])[0]]
            shift_img=[fits.open(imfiles[0][i][:-5]+'_trans0.fits')[0]]
            new_data = shift_img[0].data
            new_head=orgnl_img[0].header[:-3]+ref_hdu[0].header[45:]
            fits.writeto(imfiles[0][i][:-5]+'_trans0.fits', new_data,
                         new_head,overwrite=True)
            final_files.append(imfiles[0][i][:-5]+'_trans0.fits')
            print('Transformed image = '+imfiles[0][i][:-5]+'_trans0.fits')
            print()
    return(final_files+wcs_img)

#Alingment with wcs
def algn_with_wcs(imfiles,band):
    #Read image headers
    hdu_list = [fits.open(imfiles[i])[0] for i in range(len(imfiles))]
    #Use middle image to align
    if len(hdu_list)>1:
        ref_index=int(len(hdu_list)/2)
        hdu1 = hdu_list[ref_index]
    else:
        ref_index=0
        hdu1=hdu_list[ref_index]
    #Check image EXPTIME remove if B exptime is != 90 or other is != 60 
    #ETIME = [i.header['EXPTIME'] for i in hdu_list]
    #etrem_indx = []
    #for i in range(len(ETIME)):
    #    if band == 'B':
    #        if ETIME[i] != 90.:
    #            etrem_indx.append(i)
    #            print('Image ',imfiles[i])
    #            print('will be removed due to bad EXPTIME')
    #    else:
    #        if ETIME[i] != 60.:
    #            etrem_indx.append(i)
    #            print('Image ',imfiles[i])
    #            print('will be removed due to bad EXPTIME')
    #Reproject ~ align based on WCS solution
    for i in range(len(imfiles)):
        if i != ref_index: # and i not in etrem_indx:
            hdu2 = hdu_list[i]
            array,footprint = reproject_interp(input_data=hdu2,
                                              output_projection=hdu1.header)
            #Save aligned images into fits files
            fits.writeto(imfiles[i][:-5]+'_trans.fits', array, hdu1.header,
                         overwrite=True)
    #Create and save the combination list with the aligned images
    aligned_imgs = [imfiles[i][:-5]+'_trans.fits'
                    for i in range(len(imfiles)) 
                    if imfiles[i]!=imfiles[ref_index]] # i not in etrem_indx
                   # and imfiles[i]!=imfiles[ref_index]]
    aligned_imgs.insert(ref_index,imfiles[ref_index])
    return(aligned_imgs)

#Combination
def comb(comb_list,scaled_list):
    #Read the headers of the original images
    hdu_in = [fits.open(comb_list[i])[0] for i in range(len(comb_list))]
    #Name of output file = obj + filt + date
    #obj and filt from first image header.
    objn = comb_list[0].split('_')
    obj  = objn[0][2:] 
    filt = hdu_in[0].header['FILTERS']
    salida = obj+'_'+filt.lower()+'_comb.fits'
    #Combine with IRAF
    imcombine_e(str(mpath.parents[2])+'/scripts/',scaled_list,salida)
    #Calculate mean DATE-OBS
    DOB = pd.Series([dt.strptime(i.header['DATE-OBS'],"%Y-%m-%dT%H:%M:%S.%f")
                     for i in hdu_in])
    mean_DOB = DOB.mean()
    #Calculate mean TIME-OBS = UT and ST
    TOB = pd.Series([dt.strptime(i.header['TIME-OBS'],"%H:%M:%S.%f")
                    for i in hdu_in])
    SidT = pd.Series([dt.strptime(i.header['ST'],"%H:%M:%S.%f")
                    for i in hdu_in])
    mean_TOB  = TOB.mean()
    mean_SidT = SidT.mean()
    #Calculate mean MJD-OBS
    MJD = [i.header['MJD-OBS'] for i in hdu_in]
    mean_MJD = np.mean(MJD)
    #Calculate mean AIRMASS
    AIRM = [i.header['AIRMASS'] for i in hdu_in]
    mean_AIRM = np.mean(AIRM)
    #Read SN coordinates
    #sncatcoord=pd.read_csv(str(mpath.parents[1])+'/objetos/'+obj+'/'+obj\
    #                       +'.coo',header=None,delim_whitespace=True,
    #                       comment='#')
    #snradeg  = str(float(sncatcoord[0]))
    #sndecdeg = str(float(sncatcoord[1]))
    df = pd.read_csv(str(mpath.parents[1])+'/objetos.csv')
    snradeg,sndecdeg = df[df["objeto"]=="OGLE-2025-BLG-0451"][["ra_deg", "dec_deg"]].values[0]
    #Update combined image header
    with fits.open(salida,'update') as f:
        for hdu in f:
            hdu.header['DATE-OBS'] = (mean_DOB.strftime("%Y-%m-%dT%H:%M:%S.%f"),
                                      'mean date of observation (yyy-mm-dd)')
            hdu.header['TIME-OBS'] = (mean_TOB.strftime("%H:%M:%S.%f"),
                                      'mean time at start of observation')
            hdu.header['UT']       = (mean_TOB.strftime("%H:%M:%S.%f"),
                                      'mean universal time')
            hdu.header['ST']       = (mean_SidT.strftime("%H:%M:%S.%f"),
                                      'mean sidereal time')
            hdu.header['MJD-OBS']  = (mean_MJD, 'mean MJD of observation')
            hdu.header['AIRMASS']  = (mean_AIRM, 'mean airmass')
    hdusal=fits.open(salida,'update')
    hdusal[0].header.insert(26,('SNRA',snradeg,'SN RA in Deg'))
    hdusal[0].header.insert(27,('SNDEC',sndecdeg,'SN DEC in Deg'))
    hdusal.close()
    return(salida)

import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.visualization import simple_norm
from astropy.coordinates import SkyCoord
import astropy.units as u
import warnings
from matplotlib import MatplotlibDeprecationWarning

warnings.filterwarnings(
    "ignore",
    category=MatplotlibDeprecationWarning
)
from astropy import log
log.setLevel("WARNING")


def plot_catalog_on_image(
    fits_file,
    catalog,
    ra_col,
    dec_col,
    obj_ra,
    obj_dec,
    out_png,
    title=""
):
    with fits.open(fits_file) as hdul:
        data = hdul[0].data
        wcs = WCS(hdul[0].header)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection=wcs)
    ax.imshow(
        data,
        cmap="gray",
        norm=simple_norm(data, "sqrt", percent=99.5),
        origin="lower"
    )
    x, y = wcs.world_to_pixel_values(
        catalog[ra_col],
        catalog[dec_col]
    )
    ax.plot(
        x, y,
        marker="o",
        ls="",
        ms=7,
        mec="yellow",
        mfc="none",
        mew=1.3,
        label="Estrellas referencia"
    )
    xo, yo = wcs.world_to_pixel_values(obj_ra, obj_dec)
    ax.plot(
        xo, yo,
        marker="o",
        ls="",
        ms=11,
        mec="cyan",
        mfc="none",
        mew=2.0,
        label="Objeto"
    )
    ax.set_title(title, fontsize=12, pad=10)
    ax.coords[0].set_axislabel("RA (J2000)")
    ax.coords[1].set_axislabel("Dec (J2000)")
    ax.legend(loc="upper right", fontsize=9)
    fig.subplots_adjust(
        left=0.12,
        right=0.97,
        bottom=0.10,
        top=0.95
    )
    fig.savefig(
        out_png,
        dpi=150,
        bbox_inches="tight",
        pad_inches=0.02
    )
    plt.close(fig)
    print(f"Plot guardado: {out_png}")


###############################################################################
# Main
###############################################################################
#Select all images taken with same filter
print()
parser=argparse.ArgumentParser()
parser.add_argument('-F',help='filter')
arguments = parser.parse_args()
filt=arguments.F
print('Combining filter ',filt)
print()
#Check active ds9 windows
'''
print('ds9 active sessions: ')
chkds9  = imexam.list_active_ds9() 
nameds9 = ([c[0] for c in chkds9.values()])
#If our ds9 window (alds9) is already open do nothing, else open it
if 'alds9' not in nameds9:
    subprocess.Popen(['ds9', '-title', 'alds9'])
    time.sleep(1.5)
'''
remove(filt)
time.sleep(0.5)
while True:
    print('Selecting files')
    #Select all images taken with same filter
    mpath  = Path().absolute()
    images = ImageFileCollection(str(mpath)+'/',keywords='*')
    lista  = images.files_filtered(calibz='subtracted bias',
                                   calibf='flat correction',#astromet='yes',
                                   filters=filt)
    imfiles = pd.DataFrame(lista)
    imfiles.columns = [0]
    print(imfiles[0][0])
    objname=imfiles[0][0].split('_')[0][2:]
    # Generación automática del .coo siempre al inicio
    imdir = os.getcwd()
    ra_obj, dec_obj = load_target_coordinates(imdir)

# Calcula radio ~mitad FOV
    hdul = fits.open(imfiles[0][0])
    ny, nx = hdul[0].data.shape
    try:
        cdelt = np.mean(np.abs([hdul[0].header['CDELT1'], hdul[0].header['CDELT2']])) * 3600  # arcsec/pix
    except:
        cdelt = 0.53  # HSH fallback
    hdul.close()
    search_radius_arcmin = (min(nx, ny) * cdelt / 60) / 2
    

    global_coo = str(mpath.parents[1]) + f'/objetos/{objname}/{objname}_refpsf.coo'
    if not os.path.exists(global_coo):
        try:
            stars_auto = generate_ref_psf_coo(objname, ra_obj, dec_obj, imdir, lista, 
                         fov_frac=0.3, min_mag=9, max_mag=12, plot=True)
        except:
            print("Generación automática falló. Selecciona refs manual en DS9.")
            # Fuerza manual en primera imagen
            coordfile_global = select_xy(imfiles[0][0])
            # Copia a carpeta global
            os.system(f"mkdir -p $(dirname {global_coo}) && cp {coordfile_global} {global_coo}")
    else:
        print(f"Refs automático generado y copiado a {global_coo}")
    if not any('trans' in imf for imf in imfiles[0]):
        break
    #Read field stars file
	

print("Collected images:")
print(*('\t'+im+'\n' for im in imfiles[0]))
print()

#Check if every image has solved wcs a succesfull astrometry
crv=[]
for imf in imfiles[0]:
    #Read image headers
    hdu = [fits.open(imf)[0]]
    #Save crval value
    crv.append(hdu[0].header['CRVAL1'])

if all(cv=='INDEF' for cv in crv):
#Align and combine manually without WCS
    #Shift images
    shift_imgs=shift(imfiles)
    #Append 0 to beggining of image list to know it does not have WCS
    shift_imgs.insert(0,'0')
elif any(cv=='INDEF' for cv in crv) and not all(cv=='INDEF' for cv in crv):
#Align the ones withuot WCS w.r.t. the first image with WCS
#Copy the WCS solution to the mentioned images
#Then combined all images as if the astrometric solution was good for all
    #Shift images
    shift_imgs=algn_with_some_wcs(imfiles,crv, objname)
    #Append 1 to beggining of image list to know it does have WCS
    shift_imgs.insert(0,'1')
else:
    shift_imgs=imfiles[0].tolist()
    #Append 1 to beggining of image list to know it does have WCS
    shift_imgs.insert(0,'1')
    


#If images have WCS
if shift_imgs[0]=='1':
    #Read reference stars
    stars = pd.read_csv(str(mpath.parents[1])+'/objetos/'+objname+'/'+\
                    objname+'_refpsf.coo', delim_whitespace=True,comment='#',
                    header=None)
    #Get the position xy and measure FWHM for field stars
    #Get the mean FWHM for each image
    # -------------------- BLOQUE FWHM START --------------------
    xy = []
    im_mean_fwhmx = []
    im_std_fwhmx  = []
    im_mean_fwhmy = []
    im_std_fwhmy  = []

    # Convertimos las coordenadas RA/Dec de las estrellas a XY para cada imagen
    for im in range(1, len(shift_imgs)):
        star_positions = radec2xy(shift_imgs[im], stars)
        xy.append(star_positions)

    imfiles1 = []

    # Primera imagen
    imfiles1.append(shift_imgs[1])

    # Primera estrella del archivo .coo ya traducida a píxeles
    if len(xy[0]) == 0:
        print(f"❌ No hay estrellas válidas en la imagen {shift_imgs[1]}")
        fwhm_info = (0.0, 0.0, 0.0, 4.0)
    else:
        center_star = xy[0][0]
        center_star = (float(center_star[0]), float(center_star[1]))
        fwhm_info = FWHM_im(shift_imgs[1], center_star)

    # Guardar FWHM de forma segura
    im_mean_fwhmx.append(fwhm_info[3])
    im_std_fwhmx.append(0)
    im_mean_fwhmy.append(fwhm_info[3])
    im_std_fwhmy.append(0)

    # Loop sobre el resto de las imágenes
    for im in range(2, len(shift_imgs)):
        # Verificar que haya al menos una estrella
        if len(xy[im-1]) == 0:
            print(f"❌ No hay estrellas válidas en la imagen {shift_imgs[im]}")
            fwhm_info = (0.0, 0.0, 0.0, 4.0)
        else:
            center_star = xy[im-1][0]
            center_star = (float(center_star[0]), float(center_star[1]))
            fwhm_info = FWHM_im(shift_imgs[im], center_star)
    
        # Comprobar si la estrella está dentro de los límites y FWHM > 0
        if fwhm_info[3] > 0:
            imfiles1.append(shift_imgs[im])
            im_mean_fwhmx.append(fwhm_info[3])
            im_std_fwhmx.append(0)
            im_mean_fwhmy.append(fwhm_info[3])
            im_std_fwhmy.append(0)
        else:
            print('*'*75)
            print(f"Imagen {shift_imgs[im]} ignorada por FWHM inválida")
            print()

    #Get the mean FWHM of the set of images (global mean FWHM)
    glob_mean_fwhmx = np.mean(im_mean_fwhmx) 
    glob_std_fwhmx  = np.std(im_mean_fwhmx)  
    glob_mean_fwhmy = np.mean(im_mean_fwhmy) 
    glob_std_fwhmy  = np.std(im_mean_fwhmy)
    #Remove images with im_FWHM > global_FWHM + 3sigma,
    #save good images in imfiles2
    #Save FWHM to calculate aperture to measure mean flux of each image
    imfiles2=[]
    fwhm_4ap=[]
    for i in range(len(imfiles1)):
        if np.abs(im_mean_fwhmx[i] - glob_mean_fwhmx) <= 3*glob_std_fwhmx \
        and np.abs(im_mean_fwhmy[i] - glob_mean_fwhmy) <= 3*glob_std_fwhmy:
            imfiles2.append(imfiles1[i])
            fwhm_4ap.append(max(im_mean_fwhmx[i],im_mean_fwhmy[i]))

            imfiles2.append(imfiles1[i])
            fwhm_4ap.append(max(im_mean_fwhmx[i-1],im_mean_fwhmy[i-1]))
        else:
            print('*'*75)
            print('Image ', imfiles1[i],' removed due to bad seeing')
            print()
    #Measure background and calculate aperture photometry for
    #selected stars
    #im_meax* values is the mean flux and bkg value per image
    #glob* values are the mead flux and bkg value of all the images
    '''
    im_mean_flux = []
    im_mean_bkg  = []
    for i in range(len(imfiles2)):
        fx,bg = bg_flux_im(imfiles2[i],xy[i],fwhm_4ap[i])
        im_mean_flux.append(fx)
        im_mean_bkg.append(bg)
    glob_mean_flux = np.mean(im_mean_flux)
    '''
    # Crear lista de coordenadas correspondientes a las imágenes filtradas
    xy2 = [xy[shift_imgs.index(f)-1] for f in imfiles2]
    
    # Calcular flujo y background para cada imagen filtrada
    im_mean_flux = []
    im_mean_bkg  = []
    
    for i in range(len(imfiles2)):
        fx, bg = bg_flux_im(imfiles2[i], xy2[i], fwhm_4ap[i])
        im_mean_flux.append(fx)
        im_mean_bkg.append(bg)

    glob_mean_flux = np.mean(im_mean_flux)

    #Calculate reference flux and background to subtract to every image
    ref_img = imfiles2[int((len(imfiles2)/2)-1)]
    ref_flux,ref_bkg_value = bg_flux_im(ref_img,xy[int((len(imfiles2)/2)-1)],
                                        fwhm_4ap[int((len(imfiles2)/2)-1)])
    #Align images
    trans_files = algn_with_wcs(imfiles2,filt)
    #Scale images by bkg and flux and save them
    scal_files = []
    for i in range(len(trans_files)):
        hdu  = fits.open(trans_files[i])[0]
        data = fits.getdata(trans_files[i])
        #Scaled image: (img-bkg)*(ref_flux/img_flux)
        '''
        data_norm = (data-im_mean_bkg[i])*(ref_flux/im_mean_flux[i])
        simgname  = trans_files[i][:-5]+'_scaled.fits'
        scal_files.append(simgname)
        fits.writeto(simgname,data_norm,hdu.header,overwrite=True)
        '''

        if np.ma.isMaskedArray(data):
            data = data.filled(0)  # reemplaza los píxeles enmascarados por 0
        data = np.array(data, dtype=np.float64)

        # Escalado por fondo y flujo de referencia
        data_norm = (data - im_mean_bkg[i]) * (ref_flux / im_mean_flux[i])
        data_norm = np.nan_to_num(data_norm)

        # Nombre de salida
        simgname  = trans_files[i][:-5]+'_scaled.fits'
        scal_files.append(simgname)

        # Guardar FITS
        if np.ma.isMaskedArray(data_norm):
            data_norm = data_norm.filled(np.nan)  # o 0
        fits.writeto(simgname, data_norm, hdu.header, overwrite=True)

    #Save scaled aligned images to list to then combine them
    with open('input_combine.lst','w+') as outfile:
        outfile.write('\n'.join(scal_files))
        outfile.write('\n')
    #Combine images 
    comb_img = comb(imfiles2,'input_combine.lst')
    #Add background to combined image
    hdu_comb    = fits.open(comb_img)[0]
    data_comb   = fits.getdata(comb_img)
    data_comb_f = data_comb+np.mean(im_mean_bkg)
    #Save combined image with added mean bkg
    fits.writeto(comb_img,data_comb_f,hdu_comb.header,overwrite=True)
    #Show combined image in ds9
    showinds9(comb_img, use_ds9=use_ds9)
    
#If images do not have WCS
elif shift_imgs[0]=='0':
    imfiles2=[]
    for i in range(1,len(shift_imgs)):
        imfiles2.append(shift_imgs[i])
    #Save scaled aligned images to list to then combine them
    with open('input_combine.lst','w+') as outfile:
        outfile.write('\n'.join(imfiles2))
        outfile.write('\n')
    #Combine images 
    comb_img = comb(imfiles2,'input_combine.lst')
    #Show combined image in ds9
    showinds9(comb_img, use_ds9=use_ds9)
'''
if not os.path.exists(f"{objname}_cat.csv"):
    generate_gaia_catalog(objname, comb_img, radius_arcmin=4.2)  # ~12' cubre todo HSH + margen
else:
    print("Catálogo de fuentes de referencia para fotometría existente. Si quiere crearlo de nuevo, elimine el existente")
'''    
#Move pixel masks to pix_mask folder
#print('Pixel masks moved to  pix_mask folder')
#if not os.path.exists('pix_mask'):
#    os.system("mkdir %s"%'pix_mask')
#os.system('mv *'+filt.lower()+'*.pl pix_mask')
#os.system('mv *'+filt+'*.pl pix_mask 2>/dev/null || true')  # Ignora error si no hay archivos

# ==========================
# PLOTS FINALES DE CONTROL
# ==========================

# Catálogo Gaia (fotometría)
cat_phot = pd.read_csv(f"../../objetos/{objname}/{objname}_cat.csv")

plot_catalog_on_image(
    fits_file=comb_img,
    catalog=cat_phot,
    ra_col="RA",
    dec_col="Dec",
    obj_ra=ra_obj,
    obj_dec=dec_obj,
    out_png=f"{objname}{filt}_cat.png",
    title=f"{objname} – Catálogo Gaia (fotometría)"
)

# Catálogo de alineación
cat_align = pd.read_csv(
    str(mpath.parents[1]) + f"/objetos/{objname}/{objname}_refpsf.coo",
    delim_whitespace=True,
    header=None,
    names=["RA", "Dec", "mag"]
)

plot_catalog_on_image(
    fits_file=comb_img,
    catalog=cat_align,
    ra_col="RA",
    dec_col="Dec",
    obj_ra=ra_obj,
    obj_dec=dec_obj,
    out_png=f"{objname}{filt}_cat_alig.png",
    title=f"{objname} – Catálogo alineación"
)


#Remove auxiliary files
os.system('rm -f *.coo')
os.system('rm -f *.ctr')
os.system('rm -f *.obj')
os.system('rm -f *_scaled.fits')
os.system('rm -f *_trans0.fits')
os.system('rm -f *_trans.fits')
os.system('rm -f *_trans_scaled.fits')
os.system('rm -f *_trans.fits.ctr')
os.system('rm -f *.fits.ctr')
os.system('rm -f input_combine.lst')
os.system('rm -f fwhm*')
os.system('rm -f geomap.xy')
os.system('rm -f out.geo')
time.sleep(1.)
