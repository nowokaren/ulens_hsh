import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS

import os
from astropy.visualization import simple_norm
import requests
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.coordinates import SkyCoord, search_around_sky
from astroquery.vizier import Vizier

def generate_refcat(objname, ra_center, dec_center, 
                    img_path, objects_dir, 
                    fov_frac=0.3, min_mag=9, max_mag=12, 
                    use_catalogs = "all", plot=False, overwrite=False):
    
    
    outpath = objects_dir / objname 
    cat_path = outpath/ f"{objname}_alig_cat.csv"
    if cat_path.exists() and overwrite:
        print(f"         Omitiendo: Archivo {cat_path} ya existe.")
        return True
    elif cat_path.exists() and not overwrite:
        print(f"         Sobrescribiendo archivo {cat_path}")
    else:
        print(f"         Generando archivo {cat_path}...")
    
    # Choose radius for searching reference stars as a fraction of FOV diagonal   
    with fits.open(os.path.join(img_path)) as hdul:
        w_ref = WCS(hdul[0].header)
        ny, nx = hdul[0].data.shape
    
    corners_pix = np.array([[0,0],[nx,0],[0,ny],[nx,ny]])
    ra_c, dec_c = w_ref.pixel_to_world_values(corners_pix[:,0], corners_pix[:,1])
    coord_corners = SkyCoord(ra_c*u.deg, dec_c*u.deg)
    
    _, _, sep2d, _ = search_around_sky(coord_corners, coord_corners, 180*u.deg)
    fov_diag_arcmin = sep2d.max().to(u.arcmin).value
    search_radius_arcmin = fov_diag_arcmin * fov_frac   # radio ≈ diagonal/2
    
    print(f"         FOV diagonal: {fov_diag_arcmin:.1f}' → Search radius: {search_radius_arcmin:.1f}'")
    
    
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
    
    if use_catalogs != "all":
        catalogs = {k: v for k, v in catalogs.items() if k in use_catalogs}
        
    for name, c in catalogs.items():
        mag_limit = f"{min_mag}..{max_mag}"
        try:
            v = Vizier(columns=[c["ra"], c["dec"], c["mag"]],
                       column_filters={c["mag"]: mag_limit}, row_limit=10000)
            r = v.query_region(coord, radius=radius, catalog=c["cat"])
    
            if not r or len(r[0]) == 0:
                print(f"         catalog {name} → 0 refs")
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
            print(f"         catalog {name} → {len(idx)} refs")
    
        except Exception as e:
            print(f"  Error: {e}")
    
    refs = np.array(refs, dtype=object)
    
    if len(refs) == 0:
        raise RuntimeError("No se encontraron estrellas de referencia")

     
    # Save as CSV
    df = pd.DataFrame({
        "ra": refs[:, 0],
        "dec": refs[:, 1],
        "mag": refs[:, 2],
        "catalog": refs[:, 3]
    })
    df.to_csv(cat_path, index=False)
    '''
    np.savetxt(
        cat_path,
        refs[:,:3].astype(float),
        fmt="%.6f %.6f %.3f"
    )
    '''

    print(f"      ✓ Archivo {cat_path} generado correctamente.")
    
    if plot:
        colors = {"Gaia": "yellow", "2MASS": "orange", "APASS": "lime"}
        
        with fits.open(img_path) as hdul:
            data = hdul[0].data
            w = WCS(hdul[0].header)
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
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
    
        ax.set_title(f"{objname} - Estrellas de referencia")
        ax.legend(fontsize=8)
        plot_path = outpath / f"{objname}_align_cat.png"
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close(fig)
        print(f"      ✓ Plot {plot_path} generado correctamente.")
    return cat_path

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
    catalog = pd.read_csv(catalog)
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
    print(f"         ✓ Plot saved: {out_png}")
    
    
