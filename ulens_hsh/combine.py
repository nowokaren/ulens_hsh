from alineacion import center_e,geomap_e,geotran_e,imcombine_e
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy import wcs
from astropy.stats import SigmaClip
from photutils import ModeEstimatorBackground
from photutils import aperture_photometry,CircularAperture,CircularAnnulus
from reproject import reproject_interp
from datetime import datetime, timezone
from tqdm.auto import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from fits_io import read_fits_data, img_stats, image_collection
from matplotlib.gridspec import GridSpec
import re
from collections import defaultdict
from ccdproc import ImageFileCollection
from datetime import datetime





#Delete files from possible previous run from the list


def remove_aligment_tempfiles(filt, workdir="."):
    """
    Remove intermediate FITS files for a given filter.
    """

    patterns = [
        f"*{filt.lower()}*_trans0.fits",
        f"*{filt.lower()}*_trans.fits",
        f"*{filt.lower()}*_scaled.fits",
        f"*{filt.lower()}*_comb2.fits",
    ]

    workdir = Path(workdir)

    removed = []

    for pattern in patterns:
        for file in workdir.glob(pattern):
            file.unlink()
            removed.append(file.name)

    return removed


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


#Alingment with wcs
def algn_with_wcs(imfiles):
    '''imfiles: list of path of images to be combined (all images must have the same filter band)'''
    #Read image headers
    hdu_list = [fits.open(imfiles[i])[0] for i in range(len(imfiles))]
    #Use middle image to align
    if len(hdu_list) > 1:
        ref_index = int(len(hdu_list) / 2)
        hdu1 = hdu_list[ref_index]
    else:
        ref_index = 0
        hdu1 = hdu_list[ref_index]

    aligned_imgs = []
    for i in range(len(imfiles)):
        impath = str(imfiles[i])
        if i != ref_index:
            hdu2 = hdu_list[i]
            array, footprint = reproject_interp(input_data=hdu2,
                                                output_projection=hdu1.header)
            #Save aligned images into fits files
            trans_path = impath[:-5] + '_trans.fits'
            fits.writeto(trans_path, array, hdu1.header, overwrite=True)
            aligned_imgs.append(trans_path)
        else:
            aligned_imgs.append(impath)
    
    return aligned_imgs

#Combination
def comb(objname, filter_band, comb_list,scaled_list, objects_csv, output_path="."):
    #Read the headers of the original images
    hdu_in = [fits.open(comb_list[i])[0] for i in range(len(comb_list))]
    #Name of output file = obj + filt + date
    combine_file = Path(output_path, f"{objname}_{filter_band.lower()}_comb.fits")
    #Combine
    imcombine_e(scaled_list,combine_file)
    #Calculate mean DATE-OBS
    DOB = pd.Series([datetime.strptime(i.header['DATE-OBS'],"%Y-%m-%dT%H:%M:%S.%f")
                     for i in hdu_in])
    mean_DOB = DOB.mean()
    #Calculate mean TIME-OBS = UT and ST
    TOB = pd.Series([datetime.strptime(i.header['TIME-OBS'],"%H:%M:%S.%f")
                    for i in hdu_in])
    SidT = pd.Series([datetime.strptime(i.header['ST'],"%H:%M:%S.%f")
                    for i in hdu_in])
    mean_TOB  = TOB.mean()
    mean_SidT = SidT.mean()
    #Calculate mean MJD-OBS
    MJD = [i.header['MJD-OBS'] for i in hdu_in]
    mean_MJD = np.mean(MJD)
    #Calculate mean AIRMASS
    AIRM = [i.header['AIRMASS'] for i in hdu_in]
    mean_AIRM = np.mean(AIRM)
    #Read ulens coordinates
    df = pd.read_csv(objects_csv)
    radeg,decdeg = df[df["objeto"]==objname][["ra_deg", "dec_deg"]].values[0]
    #Update combined image header
    with fits.open(combine_file,'update') as f:
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
    hdusal=fits.open(combine_file,'update')
    hdusal[0].header.insert(26,('OBJ_RA',radeg,'Science object RA in Deg'))
    hdusal[0].header.insert(27,('OBJ_DEC',decdeg,'Science object DEC in Deg'))
    hdusal.close()
    return combine_file

#Calculate mean FWHM in X and Y axis for one image.
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



def sky_to_pixel(img, stars, margin=5):
    """Convert RA/Dec to pixel coordinates, filtering out-of-bounds stars."""
    with fits.open(img) as hdul:
        ny, nx = hdul[0].data.shape
        w = wcs.WCS(hdul[0].header)
        coords = stars[["RA", "DEC"]].values
        pix_coord = w.wcs_world2pix(coords, 0)

        #pix_coord = w.wcs_world2pix(stars.iloc[:, [0, 1]].values,0)

        
        # Vectorized bounds check (margin=5)
        in_bounds = (pix_coord[:, 0] >= margin) & (pix_coord[:, 0] < nx - margin) & \
                    (pix_coord[:, 1] >= margin) & (pix_coord[:, 1] < ny - margin)
        xy = [tuple(pix_coord[i].astype(float)) for i in np.where(in_bounds)[0]]
    
    n_outside = (~in_bounds).sum()
    return xy, n_outside

def filter_images_by_fwhm(image_files, objname, objects_dir, sigma=3, default_fwhm=4.0):
    """Filter images by FWHM using sigma clipping."""

    stars = pd.read_csv(objects_dir / objname / f'{objname}_alig_cat.csv')
    outside_sources = []
    fwhms, xy_list = [], []
    for im_path in image_files[1:]:
        xy, n_outside = sky_to_pixel(im_path, stars)
        outside_sources.append(n_outside)
        xy_list.append(xy)
        if xy:
            fwhm = FWHM_im(im_path, (float(xy[0][0]), float(xy[0][1])))[3]
        else:
            print(f"      ❌ No stars in {im_path}")
            fwhm = default_fwhm
        fwhms.append(fwhm)
       
    print(f"      ⚠️  stars outside images: {outside_sources}")

    fwhms = np.array(fwhms)
    threshold = np.mean(fwhms) + sigma * np.std(fwhms)
    good = fwhms <= threshold

    imfiles2 = [image_files[i+1] for i, g in enumerate(good) if g]
    print(f"      ✓ Kept {len(imfiles2)}/{len(fwhms)} images")
    
    return imfiles2, fwhms[good].tolist()

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

def process_and_combine_images(image_files, objname, filter_band, objects_path, 
                               output_path, objects_csv, restore_bkg=False):
    """
    Procesa imágenes con WCS válido:
    - mide FWHM
    - filtra por seeing
    - hace fotometría
    - escala
    - combina
    
    Parameters
    ----------
    image_files : list
        Lista de imágenes con WCS válido
    objname : str
        Nombre del objeto
    objects_path : Path
        Ruta al directorio de objetos
    filter_band : str
        Banda fotométrica

    
    Returns
    -------
    str
        Nombre de la imagen combinada
    """

    # -------------------- CARGA CATÁLOGO DE ESTRELLAS --------------------
    catalog_path = objects_path / objname / f'{objname}_alig_cat.csv'
    reference_stars = pd.read_csv(catalog_path)

    # -------------------- MEDICIÓN DE FWHM --------------------
    pixel_coords = [sky_to_pixel(image, reference_stars)[0] for image in image_files]

    print(f"         Starting with {len(image_files)} images")
    
    valid_images = []
    fwhm_measurements = []
    # -------------------- FILTRADO POR FWHM --------------------
    for image_path, image_pixel_coords in tqdm(zip(image_files, pixel_coords), 
                                               desc=f"         FWHM Measurement", total=len(image_files)):
        # Si no hay estrellas válidas, omitir imagen
        if len(image_pixel_coords) == 0:
            fname = image_path.name if hasattr(image_path, 'name') else image_path.split('/')[-1]
            print(f"      ❌ No stars found in {fname}")
            continue
        
        first_star_center = tuple(map(float, image_pixel_coords[0]))
        fwhm_result = FWHM_im(image_path, first_star_center)
        print(image_path, fwhm_result)
        # 
        if fwhm_result[3] <= 0:
            fname = image_path.name if hasattr(image_path, 'name') else image_path.split('/')[-1]
            print(f"      ❌ Invalid FWHM in {fname}")
            continue

        valid_images.append(image_path)
        fwhm_measurements.append(fwhm_result[3])
    
    removed_fwhm = len(image_files) - len(valid_images)
    print(f"         Removed {removed_fwhm} images (invalid FWHM/stars). Remaining: {len(valid_images)}")

    if len(valid_images) < 2:
        raise RuntimeError("Not enough valid images to combine")

    # -------------------- FILTRADO POR SEEING --------------------
    mean_fwhm = np.mean(fwhm_measurements)
    std_fwhm = np.std(fwhm_measurements)

    good_seeing_images = []
    good_seeing_fwhms = []

    for image_path, fwhm_value in tqdm(zip(valid_images, fwhm_measurements), 
                                       desc=f"         Seeing Filter", total=len(valid_images)):
        if abs(fwhm_value - mean_fwhm) <= 3 * std_fwhm:
            good_seeing_images.append(image_path)
            good_seeing_fwhms.append(fwhm_value)
            print("good seeing")
        else:
            fname = image_path.name if hasattr(image_path, 'name') else image_path.split('/')[-1]
            print(f"      ⚠️  Rejected (bad seeing): {fname} (FWHM={fwhm_value:.2f})")
    
    removed_seeing = len(valid_images) - len(good_seeing_images)
    print(f"      Removed {removed_seeing} images (bad seeing). Remaining: {len(good_seeing_images)}")

    # -------------------- FOTOMETRÍA --------------------
    final_pixel_coords = [sky_to_pixel(image, reference_stars)[0] for image in good_seeing_images]

    measured_fluxes = []
    measured_backgrounds = []

    for image_path, image_coords, fwhm_value in zip(good_seeing_images, final_pixel_coords, good_seeing_fwhms):
        flux_value, background_value = bg_flux_im(image_path, image_coords, fwhm_value)
        measured_fluxes.append(flux_value)
        measured_backgrounds.append(background_value)

    # -------------------- IMAGEN DE REFERENCIA --------------------
    reference_image_idx = len(good_seeing_images) // 2
    reference_flux_value = measured_fluxes[reference_image_idx]
    ref_image = Path(good_seeing_images[reference_image_idx])

    # -------------------- ALINEADO CON WCS --------------------
    aligned_image_files = algn_with_wcs(good_seeing_images)

    # -------------------- ESCALADO --------------------
    scaled_image_files = []
    scale_factors = []
    for aligned_image, background, flux in zip(aligned_image_files, measured_backgrounds, measured_fluxes):
        header_data_unit = fits.open(aligned_image)[0]
        image_data = fits.getdata(aligned_image).astype(float)

        scaled_data = (image_data - background) * (reference_flux_value / flux)
        scaled_data = np.nan_to_num(scaled_data)

        output_filename = str(aligned_image).replace('.fits', '_scaled.fits')
        fits.writeto(output_filename, scaled_data, header_data_unit.header, overwrite=True)
        scaled_image_files.append(output_filename)
        scale_factors.append(scaled_data)


    # -------------------- COMBINACIÓN --------------------
    with open(output_path/'input_combine.lst', 'w') as combine_list_file:
        combine_list_file.write('\n'.join(scaled_image_files) + '\n')
    
    
    combined_image = comb(objname, filter_band, good_seeing_images, 
                          output_path/'input_combine.lst', objects_csv, output_path)
    
    # -------------------- RESTORE BACKGROUND --------------------
    if restore_bkg:
        data = fits.getdata(combined_image)
        hdr = fits.getheader(combined_image)
    
        data += np.mean(measured_backgrounds)
        fits.writeto(combined_image, data, hdr, overwrite=True)

    # ============================================================
    # 🧾 HEADER COMPLETO (PROVENANCE + PROCESO)
    # ============================================================

    hdul = fits.open(combined_image, mode="update")
    hdr = hdul[0].header

    # --- Identificación ---
    hdr["NCOMB"] = len(scaled_image_files)
    hdr["DATECOMB"] = datetime.now(timezone.utc).isoformat()
    hdr["FILENAME"] = combined_image.name

    # --- Referencia ---
    hdr["REFIMG"] = (Path(ref_image).name, "Reference image")

    # --- Estadísticas ---
    hdr["FWHM_MN"] = float(np.mean(good_seeing_fwhms))
    hdr["FWHM_STD"] = float(np.std(good_seeing_fwhms))
    hdr["BKG_MEAN"] = float(np.mean(measured_backgrounds))
    hdr["SCL_MEAN"] = float(np.mean(scale_factors))
    hdr["SCL_STD"] = float(np.std(scale_factors))


    # --- Flags de proceso ---
    hdr["ALIGNED"] = True
    hdr["SCALED"] = True
    hdr["BKG_SUB"] = True
    hdr["BKG_ADD"] = restore_bkg

    # --- Historia ---
    hdr.add_history("Aligned using WCS")
    hdr.add_history("Background subtracted")
    hdr.add_history("Flux scaled to reference")
    hdr.add_history("Images combined")
    hdr.add_history(f"{len(good_seeing_images)} images used:")
    for file in good_seeing_images:
        hdr.add_history(f"    {file}")
    hdul.flush()
    hdul.close()

    print(f"         ✓ Combined image: {combined_image}")
    print(f"         ✓ Used {len(scaled_image_files)} images")

    return combined_image


import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

def plot_combined(objname, img_files, output_path, show=True):
    """
    Plot combined images of a given object with a shared colorbar.
    
    Parameters
    ----------
    objname : str
        Object name.
    img_files : list
        List of FITS image paths.
    output_path : Path
        Directory where the figure will be saved.
    show : bool
        Whether to display the figure.
    """

    n = len(img_files)

    fig, axes = plt.subplots(
        1, n,
        figsize=(3 * n, 3),
        constrained_layout=True
    )

    if n == 1:
        axes = [axes]

    images = []

    for ax, img in zip(axes, img_files):
        with fits.open(img) as hdul:
            data = hdul[0].data

        # Robust scaling per image
        vmin = np.nanpercentile(data, 5)
        vmax = np.nanpercentile(data, 99)

        im = ax.imshow(
            data,
            origin="lower",
            cmap="gray",
            vmin=vmin,
            vmax=vmax
        )
        images.append(im)

        ax.set_title(str(img).split(".")[0][-6:], fontsize=9)
        ax.axis("off")

    # Shared colorbar (robust and layout-safe)
    cbar = fig.colorbar(
        images[0],
        ax=axes,
        fraction=0.03,
        pad=0.02
    )
    cbar.ax.tick_params(labelsize=6)

    fig.suptitle(f"Combined images of {objname}", fontsize=8)

    fig.savefig(
        output_path / f"combined_{objname}.png",
        dpi=150,
        bbox_inches="tight",
        pad_inches=0.02
    )

    if show:
        plt.show()
    print(f"      ✓ Combined plot saved: {output_path / f'combined_{objname}.png'}")
    plt.close(fig)





def plot_alignment(images, objname, filter_band, obj_ra, obj_dec, 
                          output_name=None, show=False, overwrite=False):
    """
    Plot unificado de control de calidad para el proceso de alineación.
    Muestra: Calibrada, Alineada, Escalada + métricas de calidad.
    
    Parameters
    ----------
    images : ImageFileCollection
        Colección de metadatos de las imágenes de la noche
    objname : str
        Nombre del objeto (para filtrar archivos)
    filter_band : str
        Banda fotométrica (ej: 'I', 'V')
    obj_ra : float
        Right Ascension del objeto en grados
    obj_dec : float
        Declination del objeto en grados
    output_name : str or None
        Nombre del archivo de salida. Si None, se usa 'alignment_<obj>_<band>.png'
    show : bool
        Si True, muestra el plot en pantalla
    overwrite : bool
        Si False y la imagen existe, no hace nada
        
    Returns
    -------
    Path
        Ruta del archivo PNG generado
    """
    
    if output_name is None:
        output_name = f"alignment_{objname}_{filter_band}.png"
    
    night_dir = images.location
    output_path = night_dir / output_name
    
    if output_path.exists() and not overwrite:
        print(f"      ✓ Plot already exists: {output_path} (overwrite=False). Skipping.")
        return output_path

    # ------------------------------------------------------------------
    # Selección y organización de archivos
    # ------------------------------------------------------------------
    df = images.summary.to_pandas()
    
    # Filtro más permisivo: incluye archivos intermedios (_trans, _scaled)
    # que pueden no tener astromet actualizado
    df_obj = df[
        (df["object"] == objname) & 
        (~df["filename"].str.contains("comb", na=False)) &
        (df["filters"] == filter_band) &
        (
            (df["astromet"] == "yes") |  # Archivos con astrometría exitosa
            (df["astromet"] == "failure") |  # Archivos con astrometría fallida
            df["filename"].str.contains("_trans") |  # Archivos alineados
            df["filename"].str.contains("_scaled")  # Archivos escalados
        )
    ]
    
    if len(df_obj) == 0:
        print(f"      No images found for {objname} in {filter_band} band")
        return None
    
    # Patrón para extraer número de versión (ej: i0001, v0042)
    pattern = rf'{filter_band.lower()}(\d{{4}})'
    
    # Diccionario: version -> {"calib": path, "trans": path, "scaled": path, "status": str}
    images_dict = defaultdict(lambda: {
        "calib": None, 
        "trans": None, 
        "scaled": None,
        "status": "unknown",
        "qc_fwhm": None,
        "qc_see": None  # ⭐ Cambiado de qc_seeing a qc_see
    })
    
    # Clasificar archivos por versión
    # Primero: archivos del DataFrame (para metadata como qc_fwhm)
    for _, row in df_obj.iterrows():
        fname = row["filename"]
        match = re.search(pattern, fname)
        if not match:
            continue
        
        version = match.group(1)
        path = Path(night_dir, fname)
        
        # Calibrada (solo WCS, sin transformar)
        if fname.endswith("_wcs.fits") and not fname.endswith("_wcs_trans.fits"):
            images_dict[version]["calib"] = path
            images_dict[version]["status"] = row.get("astromet", "unknown")
            images_dict[version]["qc_fwhm"] = row.get("qc_fwhm", None)
            images_dict[version]["qc_see"] = row.get("qc_see", None)  # ⭐ Cambiado
    
    # Segundo: buscar archivos _trans y _scaled directamente en el filesystem
    # (estos pueden no estar en el DataFrame si fueron creados después de la última indexación)
    for calib_path in [v["calib"] for v in images_dict.values() if v["calib"]]:
        version_match = re.search(pattern, calib_path.name)
        if not version_match:
            continue
        version = version_match.group(1)
        
        # Buscar archivo _trans correspondiente
        trans_path = calib_path.parent / f"{calib_path.stem}_trans.fits"
        if trans_path.exists():
            images_dict[version]["trans"] = trans_path
        
        # Buscar archivo _scaled correspondiente
        # El patrón puede ser complejo, buscar variaciones
        for scaled_candidate in calib_path.parent.glob(f"*{filter_band.lower()}{version}*_scaled.fits"):
            images_dict[version]["scaled"] = scaled_candidate
            break
    
    # Ordenar versiones numéricamente
    versions_sorted = sorted(images_dict.keys(), key=int)
    n_images = len(versions_sorted)
    
    if n_images == 0:
        print(f"      No valid image sequences found for {objname}")
        return None
    
    # ------------------------------------------------------------------
    # Crear figura
    # ------------------------------------------------------------------
    # 3 columnas (calib, aligned, scaled) + 1 para métricas
    fig = plt.figure(figsize=(16, 3.5 * n_images))
    gs = GridSpec(
        nrows=n_images, 
        ncols=4,
        width_ratios=[1, 1, 1, 0.4],
        hspace=0.25,
        wspace=0.15
    )
    
    print(f"      → Generating alignment plots for {n_images} images...")
    
    for i, version in enumerate(tqdm(versions_sorted, desc="      → Processing")):
        img_info = images_dict[version]
        
        calib_path = img_info["calib"]
        trans_path = img_info["trans"]
        scaled_path = img_info["scaled"]
        astro_status = img_info["status"]
        qc_fwhm = img_info["qc_fwhm"]
        qc_see = img_info["qc_see"]  # ⭐ Cambiado
        
        # Lista de paths y títulos
        paths = [calib_path, trans_path, scaled_path]
        titles = ["Calibrated", "Aligned", "Scaled"]
        
        # Estados de cada etapa
        states = []
        
        # ------------------------------------------------------------------
        # Plotear las 3 imágenes
        # ------------------------------------------------------------------
        for j, (path, title) in enumerate(zip(paths, titles)):
            ax = fig.add_subplot(gs[i, j])
            
            # ---- CASO 1: Archivo existe ----
            if path and path.exists():
                try:
                    img_data = fits.getdata(path).astype(float)
                    
                    # Normalización robusta
                    vmin = np.nanpercentile(img_data, 5)
                    vmax = np.nanpercentile(img_data, 99)
                    
                    im = ax.imshow(
                        img_data,
                        origin="lower",
                        cmap="gray",
                        vmin=vmin,
                        vmax=vmax
                    )
                    
                    # Título con información de estado
                    if j == 1:  # Panel de aligned
                        title_text = f"{title} (astromet: {astro_status})"
                    else:
                        title_text = f"{title}"
                    
                    ax.set_title(title_text, fontsize=9)
                    ax.axis("off")
                    
                    # Círculo en posición del objeto
                    if obj_ra is not None and obj_dec is not None:
                        try:
                            w = wcs.WCS(fits.getheader(path))
                            pix_x, pix_y = w.wcs_world2pix([[obj_ra, obj_dec]], 0)[0]
                            
                            # Verificar que está dentro de la imagen
                            ny, nx = img_data.shape
                            if 0 <= pix_x < nx and 0 <= pix_y < ny:
                                circle = plt.Circle(
                                    (pix_x, pix_y),
                                    radius=25,
                                    color="lime",
                                    fill=False,
                                    linewidth=1.5,
                                    alpha=0.8
                                )
                                ax.add_patch(circle)
                            else:
                                # Objeto fuera del FOV
                                ax.text(
                                    0.5, 0.98,
                                    "Object outside FOV",
                                    transform=ax.transAxes,
                                    ha="center", va="top",
                                    fontsize=7,
                                    color="red",
                                    bbox=dict(boxstyle="round,pad=0.3", 
                                            facecolor="black", alpha=0.6)
                                )
                        except Exception:
                            pass  # Si falla WCS, no plotear círculo
                    
                    # Colorbar pequeño
                    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                    cbar.ax.tick_params(labelsize=6)
                    
                    states.append("OK")
                    
                except Exception as e:
                    # Error al leer archivo
                    ax.set_facecolor("black")
                    ax.text(
                        0.5, 0.5,
                        f"ERROR\nCannot read file\n{str(e)[:30]}",
                        color="red",
                        fontsize=8,
                        ha="center", va="center",
                        transform=ax.transAxes
                    )
                    ax.set_title(f"{title} (ERROR)", fontsize=9, color="red")
                    ax.axis("off")
                    states.append("ERROR")
            
            # ---- CASO 2: Archivo no existe ----
            else:
                ax.set_facecolor("#1a1a1a")
                
                # Determinar razón de fallo
                if j == 0:  # Calibrada
                    reason = "Calibration failed"
                elif j == 1:  # Alineada
                    if astro_status == "failure":
                        reason = "Astrometry failed"
                    else:
                        reason = "Alignment not performed"
                elif j == 2:  # Escalada
                    reason = "Scaling not performed"
                else:
                    reason = "File missing"
                
                ax.text(
                    0.5, 0.5,
                    f"NOT AVAILABLE\n{reason}",
                    color="orange",
                    fontsize=9,
                    ha="center", va="center",
                    transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", 
                            facecolor="black", alpha=0.7)
                )
                ax.set_title(f"{title} (N/A)", fontsize=9, color="orange")
                ax.axis("off")
                states.append("N/A")
        
        # ------------------------------------------------------------------
        # Panel de métricas (4ta columna)
        # ------------------------------------------------------------------
        ax_metrics = fig.add_subplot(gs[i, 3])
        ax_metrics.axis("off")
        
        # Construir texto de métricas
        metrics_text = f"Version: {filter_band.lower()}{version}\n"
        metrics_text += "─" * 20 + "\n"
        
        # Estado del pipeline
        metrics_text += f"Calibrated:  {states[0]}\n"
        metrics_text += f"Aligned:     {states[1]}\n"
        metrics_text += f"Scaled:      {states[2]}\n"
        metrics_text += "─" * 20 + "\n"
        
        # QC metrics
        if qc_fwhm is not None and not pd.isna(qc_fwhm):
            try:
                fwhm_val = float(qc_fwhm)
                metrics_text += f"FWHM: {fwhm_val:.2f} pix\n"
            except (ValueError, TypeError):
                metrics_text += "FWHM: N/A\n"
        else:
            metrics_text += "FWHM: N/A\n"
        
        if qc_see is not None and not pd.isna(qc_see):
            try:
                seeing_val = float(qc_see)
                metrics_text += f"Seeing: {seeing_val:.2f} \"\n"
            except (ValueError, TypeError):
                metrics_text += "Seeing: N/A\n"
        else:
            metrics_text += "Seeing: N/A\n"
        
        metrics_text += "─" * 20 + "\n"
        
        # Astrometry status
        if astro_status == "yes":
            astro_color = "lime"
            astro_symbol = "✓"
        elif astro_status == "failure":
            astro_color = "red"
            astro_symbol = "✗"
        else:
            astro_color = "gray"
            astro_symbol = "?"
        
        metrics_text += f"Astrometry: {astro_symbol}\n"
        
        # Calcular estadísticas si existe scaled
        if scaled_path and scaled_path.exists():
            try:
                data = fits.getdata(scaled_path)
                med = np.nanmedian(data)
                std = np.nanstd(data)
                metrics_text += "─" * 20 + "\n"
                metrics_text += f"Med: {med:.1f} ADU\n"
                metrics_text += f"Std: {std:.1f} ADU\n"
            except Exception:
                pass
        
        # Renderizar texto
        ax_metrics.text(
            0.05, 0.95,
            metrics_text,
            transform=ax_metrics.transAxes,
            fontsize=8,
            va="top", ha="left",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", 
                     facecolor="black", alpha=0.3)
        )
    
    # ------------------------------------------------------------------
    # Título general
    # ------------------------------------------------------------------
    fig.suptitle(
        f"Alignment Quality Control — {objname} ({filter_band} band) — Night: {night_dir.name}",
        fontsize=13,
        y=0.995
    )
    
    # ------------------------------------------------------------------
    # Guardar
    # ------------------------------------------------------------------
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1, dpi=150)
    print(f"      ✓ Alignment plot saved: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return output_path