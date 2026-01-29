import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from astropy.io import fits
from astropy.wcs import WCS

import os
from astropy.visualization import simple_norm
import requests

import astropy.units as u
from astropy.coordinates import SkyCoord, search_around_sky
from astroquery.vizier import Vizier
Vizier.vizier_server = "cdsarc.cds.unistra.fr"



from astroquery.gaia import Gaia
import pandas as pd

CATALOGS_ALIG = {
        "Gaia": {
            "cat": "I/355/gaiadr3",
            "ra": "RA_ICRS",
            "dec": "DE_ICRS",
            "mag": "Gmag",
        },
        "2MASS": {
            "cat": "II/246/out",
            "ra": "RAJ2000",
            "dec": "DEJ2000",
            "mag": "Jmag",
        },
        "APASS": {
            "cat": "II/336/apass9",
            "ra": "RAJ2000",
            "dec": "DEJ2000",
            "mag": "Vmag",
        }
    }

CATALOGS_PHOT = {
        # "Gaia_SN": { # usado por el grupo de supernovas pero no existe en astroquery
        # "cat": "J/ApJ/867/105/atlas",
        # "ra": "RAJ2000",
        # "dec": "DEJ2000",
        # "mag": ["g", "r", "i"],
        # "mag_err": ["dg", "dr", "di"],
        # },
         "Gaia_SN": { 
         "cat": "J/ApJ/867/105/refcat2",
         "ra": "RA_ICRS",
         "dec": "DE_ICRS",
         "mag": ["gmag", "rmag", "imag"], # de PAN - STARRS (crossmatch)
         "mag_err": ["e_gmag", "e_rmag", "e_imag"],
         },
        # #	ATLAS all-sky stellar reference catalog, 
        # # ATLAS-REFCAT2 (original column names in green) (992637834 rows)
        "Gaia3": {
            "cat": "I/355/gaiadr3",
            "ra": "RA_ICRS",
            "dec": "DE_ICRS",
            "mag": ["Gmag", "BPmag", "RPmag"], #["phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag"],
            "mag_err": ["e_Gmag", "e_BPmag", "e_RPmag"] #["phot_g_mean_mag_error", "phot_bp_mean_mag_error", "phot_rp_mean_mag_error"],
        },
}


def generate_refcat(objname, ra_center, dec_center, 
                    img_path, objects_dir, 
                    fov_frac=0.3, min_mag=9, max_mag=12, 
                    max_mag_err = 0.25,
                    use_catalogs = "all", plot=False, 
                    type="alig", overwrite=False):
    

    outpath = objects_dir / objname 
    cat_path = outpath/ f"{objname}_{type}_cat.csv"
    if cat_path.exists() and not overwrite:
        print(f"         Omitiendo: Archivo {cat_path} ya existe.")
        return cat_path
    elif cat_path.exists() and overwrite:
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
    

    refs = []
    catalogs = CATALOGS_ALIG if type == "alig" else CATALOGS_PHOT

    if use_catalogs != "all":
        catalogs = {k: v for k, v in catalogs.items() if k in use_catalogs}
        
    for name, c in catalogs.items():
        mag_limit = f"{min_mag}..{max_mag}"
        if type == "alig":
            columns = [c["ra"], c["dec"], c["mag"]]
            column_filters = {c["mag"]: mag_limit}
        elif type == "phot":
            columns = [c["ra"], c["dec"]] + c["mag"] + c["mag_err"]
            column_filters = {}
            for mag_col in c.get("mag", []):
                column_filters[mag_col] = mag_limit
            for err_col in c.get("mag_err", []):
                column_filters[err_col] = f"<{max_mag_err}"   # ajusta si querés otro valor
        
        #try:
        v = Vizier(columns=columns,
                    column_filters=column_filters, row_limit=10000)
        r = v.query_region(coord, radius=radius, catalog=c["cat"])

        if not r or len(r[0]) == 0:
            print(f"         • Catalog {name} → 0 refs")
            continue

        t = r[0]
        df = t.to_pandas()
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        coord_refs = SkyCoord(df[c["ra"]].values * u.deg,
                    df[c["dec"]].values * u.deg)

        sep = coord.separation(coord_refs).arcmin
        df = df[sep > 0.5]
        if type == "phot":
            if use_catalogs == "Gaia3":
                df = gaia_to_vi(df)
            elif use_catalogs == "Gaia_SN":
                df = gaia_to_bvri(df)
        elif type == "alig":
            df = df.rename(columns = {c["mag"]: "mag"})
        df = df.rename(columns = {c["ra"]:"RA", c["dec"]:"DEC"})
        df["catalog"] = name
        refs.append(df)
        print(f"            • Catalog {name} → {len(df)} refs")
    
        #except Exception as e:
        #    print(f"  Error: {e}")

    if len(refs) == 0:
        raise RuntimeError("Empty reference catalog")
    

    df_out = pd.concat(refs, ignore_index=True)
    df_out.to_csv(cat_path, index=False)


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

        for cat in np.unique(df_out["catalog"]):
            m = df_out["catalog"].values == cat
            x, y = w.world_to_pixel_values(df_out["RA"].astype(float),
                                            df_out["DEC"].astype(float))
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
        print(f"      ✓ Plot saved: {plot_path} ")

    return cat_path

def plot_catalog_on_image(
    fits_file,
    catalog,
    obj_ra,
    obj_dec,
    out_png,
    ra_col="RA",
    dec_col="DEC",
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
    print(f"      ✓ Plot saved: {out_png}")
    
    

def gaia_to_vi(df):
    """
    Transform Gaia G, BP, RP magnitudes to Johnson–Cousins V and I.

    Based on ESA Gaia photometric transformations.

    Returns
    -------
    pandas.DataFrame
        Input DataFrame with added columns:
        G, BP, RP, color, V, I, and propagated errors
    """

    df = df.copy()

    # Rename for clarity
    if "phot_g_mean_mag" in df.columns:
        df["Gmag"]  = df["phot_g_mean_mag"]
        df["BPmag"] = df["phot_bp_mean_mag"]
        df["RPmag"] = df["phot_rp_mean_mag"]

    # Color
    df["color"] = df["BPmag"] - df["RPmag"]

    # Valid color range
    df = df[(df["color"] > -0.5) & (df["color"] < 2.75)]

    # --- Transformations ---
    df["V"] = (
        df["Gmag"]
        + 0.01760
        + 0.00686 * df["color"]
        + 0.1732  * df["color"]**2
    )

    df["I"] = (
        df["Gmag"]
        - 0.02085
        - 0.7419  * df["color"]
        + 0.09631 * df["color"]**2
    )

    # --- Gaia magnitude errors ---
    if "phot_g_mean_flux_over_error" in df.columns:
        df["e_Gmag"]  = 1.0857 / df["phot_g_mean_flux_over_error"]
        df["e_BPmag"] = 1.0857 / df["phot_bp_mean_flux_over_error"]
        df["e_RPmag"] = 1.0857 / df["phot_rp_mean_flux_over_error"]

    df["color_err"] = np.sqrt(df["e_BPmag"]**2 + df["e_RPmag"]**2)

    # --- Error propagation ---
    a1_V, a2_V = 0.00686, 0.1732
    a1_I, a2_I = -0.7419, 0.09631

    sigma_pol_V = 0.045858
    sigma_pol_I = 0.04956

    df["V_err"] = np.sqrt(
        df["e_Gmag"]**2
        + (a1_V + 2*a2_V*df["color"])**2 * df["color_err"]**2
        + sigma_pol_V**2
    )

    df["I_err"] = np.sqrt(
        df["e_Gmag"]**2
        + (a1_I + 2*a2_I*df["color"])**2 * df["color_err"]**2
        + sigma_pol_I**2
    )

    return df


def gaia_to_bvri(df,
                 g_col='gmag', r_col='rmag', i_col='imag',
                 dg_col='e_gmag', dr_col='e_rmag', di_col='e_imag',
                 ra_col='ra', dec_col='dec'):
    """
    Transform Gaia-like photometry (g, r, i) to Johnson-Cousins BVRI
    using Tonry et al. (2012).

    Coefficients are hardcoded from Coef.tbl.

    Parameters
    ----------
    df : pandas.DataFrame
        Input catalog with Gaia magnitudes and errors.
    g_col, r_col, i_col : str
        Column names for magnitudes.
    dg_col, dr_col, di_col : str
        Column names for magnitude uncertainties.
    ra_col, dec_col : str
        Column names for sky coordinates.

    Returns
    -------
    pandas.DataFrame
        Catalog with RA, Dec, B, V, R, I and associated uncertainties.
    """

    # --- Coeficientes Tonry+2012 (Coef.tbl) ---
    # rows: B, V, R, I
    # columns: [C0, C1, sigma_C]
    COEF = np.array([
        [ 0.213,  0.587, 0.034],  # B
        [ 0.006,  0.474, 0.012],  # V
        [-0.138, -0.131, 0.015],  # R
        [-0.367, -0.149, 0.016],  # I
    ])

    g  = df[g_col].to_numpy()
    r  = df[r_col].to_numpy()
    i  = df[i_col].to_numpy()
    dg = df[dg_col].to_numpy()
    dr = df[dr_col].to_numpy()
    di = df[di_col].to_numpy()

    color = g - r
    dcolor2 = dg**2 + dr**2

    B = g + COEF[0, 0] + COEF[0, 1] * color
    dB = np.sqrt(dg**2 + COEF[0, 2]**2 + (COEF[0, 1]**2) * dcolor2)

    V = r + COEF[1, 0] + COEF[1, 1] * color
    dV = np.sqrt(dr**2 + COEF[1, 2]**2 + (COEF[1, 1]**2) * dcolor2)

    R = r + COEF[2, 0] + COEF[2, 1] * color
    dR = np.sqrt(dr**2 + COEF[2, 2]**2 + (COEF[2, 1]**2) * dcolor2)

    I = i + COEF[3, 0] + COEF[3, 1] * color
    dI = np.sqrt(di**2 + COEF[3, 2]**2 + (COEF[3, 1]**2) * dcolor2)

    out = pd.DataFrame({
        ra_col: df[ra_col],
        dec_col: df[dec_col],
        'B': B, 'B_err': dB,
        'V': V, 'V_err': dV,
        'R': R, 'R_err': dR,
        'I': I, 'I_err': dI
    })
    return out






'''



from astroquery.utils.tap.core import TapPlus
from astroquery.vizier import Vizier
import warnings

TAP_VIZIER_URL = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap"   # o el que te funcione

from astroquery.utils.tap.core import TapPlus
import warnings

TAP_URL = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap"  # oficial y actual en 2026

def query_vizier_fallback(coord, radius, catalog, columns=None, column_filters=None, row_limit=-1):
    """
    Intenta Vizier nativo → si falla → TAP con ADQL cone-search correcto
    """
    # ────────────── Intento 1: Vizier nativo ──────────────
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            v = Vizier(
                columns=columns or ["**"],
                column_filters=column_filters or {},
                row_limit=row_limit,
                vizier_server="cdsarc.cds.unistra.fr",   # mirror más estable
                timeout=90
            )
            result = v.query_region(coord, radius=radius, catalog=catalog)
            if result and len(result) > 0 and len(result[0]) > 0:
                print(f"   → Vizier nativo OK ({catalog}, {len(result[0])} filas)")
                return result[0]
    except Exception as e:
        print(f"   Vizier falló ({catalog}): {str(e)[:120]} → probando TAP...")

    # ────────────── Intento 2: TAP ──────────────
    try:
        tap = TapPlus(url=TAP_URL, verbose=False)

        # Nombres de columnas estándar en VizieR/TAP (casi siempre estos)
        ra_col  = "RA_ICRS"    if "gaia" in catalog.lower() else "RAJ2000"
        dec_col = "DE_ICRS"    if "gaia" in catalog.lower() else "DEJ2000"

        # ADQL cone-search (forma más portable y estándar)
        cone_condition = (
            f"CONTAINS(POINT('ICRS', \"{ra_col}\", \"{dec_col}\"), "
            f"CIRCLE('ICRS', {coord.ra.deg}, {coord.dec.deg}, {radius.to(u.deg).value})) = 1"
        )

        where_clauses = [cone_condition]

        # Filtros de magnitud
        if column_filters:
            for col, filt in column_filters.items():
                if ".." in filt:
                    lo, hi = [float(x) for x in filt.split("..")]
                    where_clauses.append(f'"{col}" BETWEEN {lo} AND {hi}')
                elif "<" in filt:
                    val = float(filt.replace("<", ""))
                    where_clauses.append(f'"{col}" < {val}')
                elif ">" in filt:
                    val = float(filt.replace(">", ""))
                    where_clauses.append(f'"{col}" > {val}')

        where_str = " AND ".join(where_clauses)

        # Columnas
        cols_str = ", ".join(f'"{c}"' for c in (columns or ["*"]))

        adql = f"""
        SELECT TOP {row_limit if row_limit > 0 else 999999}
            {cols_str.replace("*", "*")}
        FROM "{catalog}"
        WHERE {where_str}
        """

        print(f"   Ejecutando ADQL:\n{adql[:300]}...")  # debug

        job = tap.launch_job(adql)
        table = job.get_results()

        if len(table) > 0:
            print(f"   → TAP OK ({catalog}, {len(table)} filas)")
            return table
        else:
            print(f"   TAP devolvió 0 filas para {catalog}")

    except Exception as e:
        print(f"   TAP falló ({catalog}): {str(e)[:150]}")

    return None

from astroquery.vizier import Vizier
from astroquery.utils.tap.core import TapPlus
import warnings

TAP_URL = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap"  # estable en 2026

def query_vizier_native(coord, radius, catalog, columns=None, column_filters=None, row_limit=-1):
    """Query vía astroquery.vizier nativo (HTTP/HTTPS)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            v = Vizier(
                columns=columns or ["**"],
                column_filters=column_filters or {},
                row_limit=row_limit,
                vizier_server="cdsarc.cds.unistra.fr",  # mirror rápido
                timeout=120  # más tiempo para catálogos grandes
            )
            result = v.query_region(coord, radius=radius, catalog=catalog)
            if result and len(result) > 0 and len(result[0]) > 0:
                print(f"   → Vizier native OK ({catalog}, {len(result[0])} filas)")
                return result[0]
    except Exception as e:
        print(f"   Vizier native falló ({catalog}): {str(e)[:120]}")
    return None

def query_vizier_tap(coord, radius, catalog, columns=None, column_filters=None, 
                     row_limit=-1, job_async=True):
    """Query vía TAP (ADQL). Usa async por defecto para evitar delays."""
    try:
        tap = TapPlus(url=TAP_URL, verbose=False)

        # Nombres de columna/tabla ajustados por catálogo (para 2MASS/APASS usa prefijo completo)
        if "gaia" in catalog.lower():
            ra_col, dec_col = "RA_ICRS", "DE_ICRS"
            table_prefix = ""  # sin prefijo
        elif "246" in catalog:  # 2MASS
            ra_col, dec_col = "RAJ2000", "DEJ2000"
            table_prefix = f'"{catalog}".'  # "II/246/out".RAJ2000
        elif "336" in catalog:  # APASS
            ra_col, dec_col = "RAJ2000", "DEJ2000"
            table_prefix = f'"{catalog}".'  # prefijo completo
        else:
            ra_col, dec_col = "RAJ2000", "DEJ2000"
            table_prefix = ""

        cone_condition = (
            f"CONTAINS(POINT('ICRS', {table_prefix}{ra_col}, {table_prefix}{dec_col}), "
            f"CIRCLE('ICRS', {coord.ra.deg}, {coord.dec.deg}, {radius.to(u.deg).value})) = 1"
        )

        where_clauses = [cone_condition]
        if column_filters:
            for col, filt in column_filters.items():
                if ".." in filt:
                    lo, hi = [float(x) for x in filt.split("..")]
                    where_clauses.append(f'{table_prefix}"{col}" BETWEEN {lo} AND {hi}')

        where_str = " AND ".join(where_clauses)

        cols_str = ", ".join(f'{table_prefix}"{c}"' for c in (columns or ["*"]))

        adql = f"""
        SELECT TOP {row_limit if row_limit > 0 else 999999}
            {cols_str}
        FROM "{catalog}"
        WHERE {where_str}
        """

        print(f"   Ejecutando ADQL (async={job_async}):\n{adql[:300]}...")

        if job_async:
            job = tap.launch_job_async(adql)
            job.wait()  # espera, pero async permite no bloquear el thread principal si es necesario
        else:
            job = tap.launch_job(adql)
        
        table = job.get_results()
        if len(table) > 0:
            print(f"   → TAP OK ({catalog}, {len(table)} filas)")
            return table
        else:
            print(f"   TAP devolvió 0 filas ({catalog})")
    except Exception as e:
        print(f"   TAP falló ({catalog}): {str(e)[:150]}")
    return None

def query_sources(coord, radius, catalog, columns=None, column_filters=None, row_limit=-1, method="fallback"):
    """
    Wrapper configurable: elige método vía config.

    """
    DO_TASK = True
    try:
        result = query_vizier_sources(
            coord, radius, catalog, columns, column_filters, row_limit
        )
        if result is not None:
            return result
    except Exception as e:
        print(f"   Vizier (sin indicar server) falló para {catalog}: {str(e)[:120]}")

    if DO_TASK:
        try:
            result = query_vizier_native(
                coord, radius, catalog, columns, column_filters, row_limit
            )
            if result is not None:
                return result
        except Exception as e:
            print(f"   Vizier (indicando server) falló para {catalog}: {str(e)[:120]}")
    
    if DO_TASK:
        try:
            result = query_vizier_tap(
                coord, radius, catalog, columns, column_filters, row_limit,
                job_async=False
            )
            if result is not None:
                return result
        except Exception as e:
            print(f"   TAP sync falló para {catalog}: {str(e)[:120]}")   
            
    if DO_TASK:
        try:
            result = query_vizier_tap(
                coord, radius, catalog, columns, column_filters, row_limit,
                job_async=True
            )
            if result is not None:
                return result
        except Exception as e:
            print(f"   TAP async falló para {catalog}: {str(e)[:120]}") 
    
    if DO_TASK:  
        print(f"   Todos los métodos fallaron para {catalog}")
        return None

def query_vizier_sources(ra, dec, settings=None):
    """
    Query reference stars around a given position using Vizier.

    Returns
    -------
    pandas.DataFrame
        Columns include:
        - RA, DEC
        - native magnitude columns (e.g. Gmag, Vmag, g, r, i, dg, dr, di, ...)
        - catalog
    """
    if settings is None:
        settings = {}

    radius = settings.get("radius", 0.7) * u.deg
    min_mag = settings.get("min_mag", 9)
    max_mag = settings.get("max_mag", 15)
    row_limit = settings.get("row_limit", -1)
    use_catalogs = settings.get("use_catalogs", "all")

    coord = SkyCoord(ra * u.deg, dec * u.deg)



    if use_catalogs != "all":
        catalogs = {k: v for k, v in CATALOGS_COLUMNS.items() if k in use_catalogs}

    dfs = []
    for name, c in catalogs.items():
        try:
            mag_cols = c["mag"]
            err_cols = c.get("mag_err", [])

            columns = [c["ra"], c["dec"]] + mag_cols + err_cols
            mag_filter = {m: f"{min_mag}..{max_mag}" for m in mag_cols}
            print(columns, mag_filter, row_limit)
            v = Vizier(
                columns=columns,
                column_filters=mag_filter,
                row_limit=row_limit,
            )

            r = v.query_region(coord, radius=radius, catalog=c["cat"])
            if not r or len(r[0]) == 0:
                continue

            t = r[0].to_pandas()

            # Rename RA/DEC to standard names
            t = t.rename(columns={
                c["ra"]: "ra",
                c["dec"]: "dec"
            })

            # Drop rows with NaNs in any magnitude column
            t = t.dropna(subset=mag_cols)

            t["catalog"] = name
            dfs.append(t)

        except Exception:
            continue

    if len(dfs) == 0:
        raise RuntimeError("Empty reference catalog – try increasing search radius or changing catalogs")
    return pd.concat(dfs, ignore_index=True)



def query_gaia_sources(
    ra,
    dec,
    settings = None
):
    """
    Query Gaia DR3 around (ra, dec) and return clean stellar sources.

    Parameters
    ----------
    ra, dec : float
        ICRS coordinates in degrees
    radius_deg : float
        Search radius in degrees
    min_mag, max_mag : float
        G magnitude limits
    min_g_snr, min_bp_snr, min_rp_snr : float
        Flux-over-error thresholds

    Returns
    -------
    pandas.DataFrame
        Gaia sources with photometry and basic quality cuts applied
    """
    if settings is None:
        settings = {}
    radius = settings.get("radius", 0.7)
    min_mag = settings.get("min_mag", 6)
    max_mag = settings.get("max_mag", 15)

    min_bp_snr=settings.get("min_bp_snr", 50)
    min_rp_snr=settings.get("min_rp_snr", 50)
    min_g_snr=settings.get("min_g_snr", 100)
    

    query = f"""
    SELECT
        source_id,
        ra, dec,
        phot_g_mean_mag,
        phot_g_mean_flux_over_error,
        phot_bp_mean_mag,
        phot_bp_mean_flux_over_error,
        phot_rp_mean_mag,
        phot_rp_mean_flux_over_error
    FROM gaiadr3.gaia_source
    WHERE
        CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {ra}, {dec}, {radius})
        ) = 1
        AND phot_g_mean_mag BETWEEN {min_mag} AND {max_mag}
        AND duplicated_source = 'false'
        AND phot_g_mean_flux_over_error  > {min_g_snr}
        AND phot_bp_mean_flux_over_error > {min_bp_snr}
        AND phot_rp_mean_flux_over_error > {min_rp_snr}
    """

    job = Gaia.launch_job(query)
    df = job.get_results().to_pandas()

    return df
# ==========================
# Catalog query registry
# ==========================

CATALOG_QUERIES = {
    "vizier": query_vizier_sources,
    "vizier_tap": query_vizier_tap,
    "vizier_native": query_vizier_native,
    "gaia": query_gaia_sources,
    "all": query_sources,
}

# ==========================
# Photometric transforms
# ==========================

PHOT_TRANSFORMS = {
    "ESA": gaia_to_vi,
    "Tonry": gaia_to_bvri,
}
def create_photometry_catalog(
    ra,
    dec,
    object_name,
    output_path,
    phot_cfg,
    overwrite=False,
):
    """
    Build photometric reference catalog according to config.

    Parameters
    ----------
    ra, dec : float
        Object coordinates
    radius_arcmin : float
        Search radius
    phot_cfg : dict
        photometry.catalog section from config.yaml

    Returns
    -------
    pandas.DataFrame
    """
    
    query_method = phot_cfg["query_method"]
    transform_name = phot_cfg.get("gaia_to_vi", "raw")
    radius = phot_cfg.get("radius", 0.7)  # degrees 
    overwrite = phot_cfg.get("overwrite", False)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    filename = f"{object_name}_{query_method}_{transform_name}.csv"
    outfile = output_path / filename

    if outfile.exists() and not overwrite:
        print(f"         Omitiendo: Archivo {outfile} ya existe.")
        return outfile
    
    # --- Query ---
    query_func = CATALOG_QUERIES[query_method]

    df = query_func(
        ra=ra,
        dec=dec,
        settings=phot_cfg
    )

    # --- Transform stage ---
    if transform_name != "raw":
        trans_func = PHOT_TRANSFORMS[transform_name]
        df = trans_func(df)
            
    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        raise RuntimeError("Empty catalog returned")

    df.to_csv(outfile, index=False)

    return df
'''