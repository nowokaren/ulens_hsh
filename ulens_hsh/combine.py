from alineacion import center_e,geomap_e,geotran_e,imcombine_e
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy import wcs
from astropy.stats import SigmaClip
from photutils import ModeEstimatorBackground
from photutils import aperture_photometry,CircularAperture,CircularAnnulus
from reproject import reproject_interp
from datetime import datetime as dt
from tqdm.auto import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from fits_io import read_fits_data, img_stats
from matplotlib.gridspec import GridSpec
import re

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
                               output_path, objects_csv):
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

    # -------------------- ALINEADO CON WCS --------------------
    aligned_image_files = algn_with_wcs(good_seeing_images)

    # -------------------- ESCALADO --------------------
    scaled_image_files = []
    for aligned_image, background, flux in zip(aligned_image_files, measured_backgrounds, measured_fluxes):
        header_data_unit = fits.open(aligned_image)[0]
        image_data = fits.getdata(aligned_image).astype(float)

        scaled_data = (image_data - background) * (reference_flux_value / flux)
        scaled_data = np.nan_to_num(scaled_data)

        output_filename = str(aligned_image).replace('.fits', '_scaled.fits')
        fits.writeto(output_filename, scaled_data, header_data_unit.header, overwrite=True)
        scaled_image_files.append(output_filename)


    # -------------------- COMBINACIÓN --------------------
    with open('input_combine.lst', 'w') as combine_list_file:
        combine_list_file.write('\n'.join(scaled_image_files) + '\n')
    
    
    combined_image = comb(objname, filter_band, good_seeing_images, 
                          'input_combine.lst', objects_csv, output_path)
    
    # -------------------- RESTAURAR FONDO --------------------
    combined_header_unit = fits.open(combined_image)[0]
    combined_data = fits.getdata(combined_image) + np.mean(measured_backgrounds)
    
    fits.writeto(combined_image, combined_data, combined_header_unit.header, overwrite=True)

    print(f"         ✓ Successfully combined {len(good_seeing_images)} images")

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




def plot_aligment(dataset, night_dir, objname, filter_band, output_name=None, show=False,
                   overwrite=False):
    """
    Genera un plot de control de calidad de la alineación:
    Calibrated, aligned, 

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

    output_name = f"aligment_{objname}_{filter_band}.png"
    output_path = night_dir / output_name
    if output_path.exists() and not overwrite:
        print(f"      ✓ Plot already exists: {output_path} (overwrite=False). Skipping.")
        return output_path

    # ------------------------------------------------------------------
    # Selección de archivos desde el dataset
    # ------------------------------------------------------------------
    if isinstance(dataset, str) and dataset.endswith(".csv"):

        ds = pd.read_csv(Path(dataset))
        ds_obj = ds[(ds["OBJECT"] == objname) & (ds["ASTROMET"]=="yes") & 
                    (~ds["FILENAME"].str.contains("comb", na=False)) &
                    (ds["FILTERS"] == filter_band)
                    ]
        calib_files = [
            Path(night_dir, fname)
            for fname in ds_obj["FILENAME"] if not "trans" in fname 
            and not "scaled" in fname
        ]
        files = ds_obj["FILENAME"]
        ver = files.str.extract(rf'{filter_band.lower()}(\d{{4}})')[0]
        is_trans = files.str.endswith("_wcs_trans.fits")
        is_wcs   = files.str.endswith("_wcs.fits")
        missing = set(ver[is_wcs]) - set(ver[is_trans])
        mask = is_trans | (is_wcs & ver.isin(missing))
        alig_files = sorted(
            [Path(night_dir, f) for f in files[mask]],
            key=lambda p: int(re.search(rf'{filter_band.lower()}(\d{{4}})', p.name).group(1))
        )
        scaled_files = [
            Path(night_dir, fname)
            for fname in ds_obj["FILENAME"] if fname.endswith("_scaled.fits") 
        ]
        
    else:
        raise ValueError("dataset debe la ruta a un CSV de metadatos.")
        
        
    calib = sorted(calib_files)
    alig = sorted(alig_files)
    scaled = sorted(scaled_files)

    if not (len(calib) == len(alig) == len(scaled)):
        print(len(calib))
        print(len(alig))
        print(len(scaled))
        raise ValueError("La cantidad de RAW, Bias y FlatBias no coincide.")

    n = len(calib)
    
    if n==0:
        print(f"      No images for object {objname}")

    # ------------------------------------------------------------------
    # Figura
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(10, 3*n))
    gs = GridSpec(
        nrows=n, ncols=3,
      #  hspace=0.30, wspace=0.15
    )

    for i in tqdm(range(n), desc=f"      → Generating aligment plots"):
        row = i

        calib_img = read_fits_data(calib[i])
        alig_img  = read_fits_data(alig[i])
        scaled_img = read_fits_data(scaled[i])

        vmin = np.percentile(calib_img, 5)
        vmax = np.percentile(calib_img, 99)

        images = [calib_img, alig_img, scaled_img]
        titles = [
            "Calibrated: " + calib[i].stem,
            "Aligned: " + alig[i].stem,
            "Scaled: " + scaled[i].stem
        ]
        cmaps  = ["gray", "gray", "gray"]
        vmins  = [vmin, vmin, vmin]
        vmaxs  = [vmax, vmax, vmax]

        # Estadísticas
        for j in range(3):
            ax_img = fig.add_subplot(gs[row, j])
            im = ax_img.imshow(images[j], origin="lower",
                               cmap=cmaps[j], vmin=vmins[j], vmax=vmaxs[j])
            ax_img.set_title(titles[j], fontsize=8)
            ax_img.axis("off")

            # Estadísticas
            #s = img_stats(images[j])
            #txt = (f"μ={s['mean']:.2f}  med={s['median']:.2f}  σ={s['std']:.2f}\n"
            #       f"p5={s['p5']:.1f}  p95={s['p95']:.1f}  neg={100*s['neg_frac']:.2f}%")
            #ax_img.text(0.02, -0.10, txt, transform=ax_img.transAxes,
            #            fontsize=8, color="yellow",
            #            bbox=dict(facecolor="black", alpha=0.5, pad=2))

            # Colorbar
            cb = fig.colorbar(im, ax=ax_img, fraction=0.03, pad=0.02)
            cb.ax.tick_params(labelsize=7)

    #fig.subplots_adjust(top=0.94)
    fig.suptitle(
        f"Noche: {night_dir.name}   |   Objeto: {objname}",
        fontsize=12,
        y=0.9
    )

    # ------------------------------------------------------------------
    # Guardar
    # ------------------------------------------------------------------
    if output_name is None:
        output_name = f"aligment_{objname}.png"

    output_path = night_dir / output_name
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.05, dpi=150)
    print(f"      ✓ Plot saved as {output_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path