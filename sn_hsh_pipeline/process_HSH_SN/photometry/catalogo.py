import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import mastcasjobs
import astropy
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import simple_norm
from photutils import CircularAperture

######################################
## FUNCIONES
######################################

def wcs_to_pix(filename,radec):
    hdulist = fits.open(filename)
    w = WCS(hdulist[0].header)
    pix = w.wcs_world2pix(radec,0,ra_dec_order=True)
    return pix
'''
def job(RA,DEC,filt1,filt2,filt3):
    wsid= 1084646065
    pwd="dakepa22"
    p1 = 'r.objid, '
    p2 = 'r.RA, '
    p3 = 'r.Dec, '
    p4 = 'r.'+filt1+', '
    p5 = 'r.d'+filt1+', '
    p6 = 'r.'+filt2+', '
    p7 = 'r.d'+filt2+', '
    p8 = 'r.'+filt3+', '
    p9 = 'r.d'+filt3+', '
    p10 = 'r.dupvar'
    query ="""select """+p1+p2+p3+p4+p5+p6+p7+p8+p9+p10+"""
        from refcat2 as r,
        dbo.fGetNearbyObjEq("""+str(RA)+""","""+str(DEC)+""",0.07) as n
            where r.objid=n.objid"""
    print(query)
    jobs = mastcasjobs.MastCasJobs(userid=wsid, password=pwd,context="HLSP_ATLAS_REFCAT2")
    return jobs.quick(query, task_name="my task")
'''
from astroquery.vizier import Vizier
import astropy.units as u
from astropy.coordinates import SkyCoord

def job(RA, DEC, filt1, filt2, filt3):
    # Catálogo ATLAS-REFCAT2 en VizieR
    catalog = "J/ApJ/867/105/atlas"
    
    # Columnas equivalentes (ajusta si querés más: objID, RAJ2000, DEJ2000, gmag, e_gmag, etc.)
    columns = ['objID', 'RAJ2000', 'DEJ2000', 
               filt1+'mag', 'e_'+filt1+'mag',
               filt2+'mag', 'e_'+filt2+'mag',
               filt3+'mag', 'e_'+filt3+'mag',
               'dup']  # dup para variables (equivalente a dupvar)
    
    coord = SkyCoord(ra=RA*u.deg, dec=DEC*u.deg)
    radius = 0.07 * u.deg  # ~4.2 arcmin, como el original (0.07 deg)
    
    v = Vizier(columns=columns, row_limit=500)  # Limita a 500 refs
    res = v.query_region(coord, radius=radius, catalog=catalog)
    
    if len(res) == 0:
        print("No refs encontradas – prueba radio mayor o catálogo alternativo")
        return pd.DataFrame()  # Empty DF
    
    res_cat = res[0].to_pandas()
    # Renombra columnas para compatibilidad con el resto del script
    res_cat.rename(columns={
        'RAJ2000': 'RA',
        'DEJ2000': 'Dec',
        filt1+'mag': filt1,
        'e_'+filt1+'mag': 'd'+filt1,
        filt2+'mag': filt2,
        'e_'+filt2+'mag': 'd'+filt2,
        filt3+'mag': filt3,
        'e_'+filt3+'mag': 'd'+filt3,
        'dup': 'dupvar'  # dup=2 para variables, como original
    }, inplace=True)
    
    return res_cat

def remove_near_objects(df,delta_pix=10):
    """ Remueve objetos más cercanos entre sí que 'delta_pix'
    
    Args:
    df: un data frame con las coordenadas de los objetos en píxeles.
    delta_pix: máxima distancia permitida entre objetos.
    
    Returns:
    index = lista con los índices de los objetos a remover.
    
    """
    index = []
    for i in range(len(df)):
        for j in range(len(df)):
            if i != j:
                dist = np.sqrt((df['x'].iloc[i] - df['x'].iloc[j])**2 + (df['y'].iloc[i] - df['y'].iloc[j])**2)
                if dist < delta_pix: 
                    index.append(i)
                    
    df.drop(index, inplace=True)
    df = df.reset_index(drop=True)
    return df

def remove_objects_near_sn(df,sn_coords,delta_pix=12):
    """ Remueve objetos cercanos a la SN.
    Remueve aquellos más cercanos que 'delta_pix'
    
    Args:
    df: un data frame con las coordenadas de los objetos en píxeles.
    delta_pix: máxima distancia permitida entre objetos.
    
    Returns:
    index = lista con los índices de los objetos a remover.
    
    """
    index = []
    for i in range(len(df)):
        dist = np.sqrt((df['x'].iloc[i] - sn_coords[0])**2 + (df['y'].iloc[i] - sn_coords[1])**2)
        if dist < delta_pix: 
            index.append(i)
                    
    df.drop(index, inplace=True)
    df = df.reset_index(drop=True)
    return df

######################################
## MAIN
######################################

##- SN
sn_name = 'OGLE-2025-BLG-0451'
sn = 'OGLE-2025-BLG-0451'

##- Lectura de imagen
comb_images = ['%s_r_comb.fits'%sn_name,'%s_v_comb.fits'%sn_name,'%s_i_comb.fits'%sn_name]
for image in comb_images:
    if os.path.exists(image):
        hdulist = fits.open(image)[0]
        data = hdulist.data.astype(float)
        break

## Lectura de coordenadas del centro de la imagen y de la SN
df = pd.read_csv('../data/objetos.csv')
pos_sn_ra, pos_sn_dec = df[df["objeto"]=="OGLE-2025-BLG-0451"][["ra_deg", "dec_deg"]].values[0]
with fits.open(image) as hdul:
    w_ref = WCS(hdul[0].header)
    ny, nx = hdul[0].data.shape
    
corners_pix = np.array([nx/2,ny/2])
pos_center_ra, pos_center_dec = w_ref.pixel_to_world_values(nx/2,ny/2)
coord_corners = SkyCoord(ra_c*u.deg, dec_c*u.deg)
    
#filename_sn_coords = '%s/%s_sn.coo'%(path,sn)
#pos_sn_ra, pos_sn_dec, pos_center_ra, pos_center_dec = np.loadtxt(filename_sn_coords, usecols=[0,1,2,3], unpack=True)
#pos_sn_wcs = np.array([pos_sn_ra,pos_sn_dec]).reshape(1,2)
#pos_sn_pix = wcs_to_pix(image, pos_sn_wcs)
#pos_center_wcs = np.array([pos_center_ra,pos_center_dec]).reshape(1,2)
#pos_center_pix = wcs_to_pix(image, pos_center_wcs)

##- Catálogo
##- Si el catálogo existe no hace nada, sino llama a refcat
if os.path.exists('%s/%s_cat.csv'%(path,sn)):
    print("El catálogo ya existe en '%s'"%path)
else:
    filter1 = 'r'
    filter2 = 'g'
    filter3 = 'i'
    res_cat = job(pos_center_ra,pos_center_dec,filter1,filter2,filter3)
    res_cat = res_cat.to_pandas()

    ##- Localiza y elimina elementos con dupvar=1 (variables) y mayores que 2 y con errores mayores que 0.25 mag
    mask = np.logical_or((res_cat['dupvar'] == 0),(res_cat['dupvar'] == 2))
    res_cat = res_cat[mask]
    res_cat = res_cat[res_cat['g'] < 19]
    res_cat = res_cat[res_cat['r'] < 19]
    res_cat = res_cat[res_cat['i'] < 19]
    res_cat = res_cat[res_cat['dg'] < 0.25]
    res_cat = res_cat[res_cat['dr'] < 0.25]
    res_cat = res_cat[res_cat['di'] < 0.25]
    res_cat.reset_index(drop=True, inplace=True)

    ##- Convierte las coordenadas del catálogo a pixeles
    positions_comp_wcs = res_cat[['RA', 'Dec']].to_numpy()
    positions_comp_pix = wcs_to_pix(image,positions_comp_wcs)
    positions_comp_pix = pd.DataFrame(positions_comp_pix, columns=('x','y'))
    res_cat['x'] = positions_comp_pix['x']
    res_cat['y'] = positions_comp_pix['y']

    ##- Remueve objetos cercanos entre sí y cercanos a la SN
    res_cat = remove_near_objects(res_cat)
    res_cat = remove_objects_near_sn(res_cat, pos_sn_pix[0])
    res_cat.to_csv('%s/%s_cat.csv'%(path,sn), index=False)


    ##- Plotea las imagenes marcando las estrellas de comparación
    fig = plt.figure(dpi=180, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)
    norm_img = simple_norm(data, 'power', power=5., percent=99.)
    ax.imshow(data, norm=norm_img, cmap='gray', origin='lower')
    ax.plot(positions_comp_pix['x'], positions_comp_pix['y'], 'o', c='red', ms=1.5)
    ax.plot(pos_sn_pix[0][0], pos_sn_pix[0][1], 'o', c='blue', ms=2)
    fig.savefig('%s/%s_catalogo.png'%(path,sn), format='png', dpi='figure')
