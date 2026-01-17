from pathlib import Path
import requests
import json
from astropy.io import fits
from astropy.io.fits import getheader
from astropy.io.fits import getval
from astropy import stats
from astropy.time import Time


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

def apply_astrometry(input_fits, output_path, api_key="wuupmjpswkcbncws"):
    """
    Sube una imagen FITS a Astrometry.net, obtiene el WCS y genera
    un archivo <original>_wcs.fit con la solución astrométrica aplicada
    en el directorio 'output_path'.

    Aplica los keywords WCS críticos uno por uno para asegurar que
    el WCS final sea válido.
    """
    input_fits = Path(input_fits)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # ------------------ LOGIN ------------------
    print(f"   [Astrometry.net] Iniciando sesión...")
    r = requests.post(
        "http://nova.astrometry.net/api/login",
        data={"request-json": json.dumps({"apikey": api_key})}
    ).json()
    session = r.get("session")
    if not session:
        raise RuntimeError("No se pudo iniciar sesión en Astrometry.net")
    print(f"   [Astrometry.net] Sesión iniciada OK")

    # ------------------ SUBIR IMAGEN ------------------
    print(f"   [Astrometry.net] Subiendo imagen {input_fits.name}...")
    with open(input_fits, "rb") as f:
        files = {"file": f}
        payload = {"request-json": json.dumps({"session": session})}
        r = requests.post("http://nova.astrometry.net/api/upload", files=files, data=payload).json()

    if "subid" not in r:
        raise RuntimeError(f"Error al subir la imagen: {r}")
    subid = r["subid"]
    print(f"   [Astrometry.net] Imagen subida (SUBID {subid})")

    # ------------------ ESPERAR JOB ------------------
    print(f"   [Astrometry.net] Esperando asignación de JOB...")
    jobid = None
    while jobid is None:
        time.sleep(5)
        jobs = requests.get(f"http://nova.astrometry.net/api/submissions/{subid}").json().get("jobs", [])
        if jobs and jobs[0] is not None:
            jobid = jobs[0]
    print(f"   [Astrometry.net] JOB asignado: {jobid}")

    # ------------------ ESPERAR RESOLUCIÓN ------------------
    print(f"   [Astrometry.net] Resolviendo campo...")
    status = "processing"
    while status == "processing" or status == "solving":
        time.sleep(4)
        status = requests.get(f"http://nova.astrometry.net/api/jobs/{jobid}").json().get("status")
    if status != "success":
        raise RuntimeError(f"Astrometry.net no pudo resolver la imagen (status={status})")
    print(f"   [Astrometry.net] Solución WCS obtenida!")

    # ------------------ DESCARGAR HEADER WCS ------------------
    wcs_url = f"http://nova.astrometry.net/wcs_file/{jobid}"
    print(f"   [Astrometry.net] Descargando solución WCS...")
    wcs_raw = requests.get(wcs_url).content

    tmp_wcs = input_fits.with_name(input_fits.stem + "_tmp_wcs.fit")
    with open(tmp_wcs, "wb") as f:
        f.write(wcs_raw)

    # ------------------ APLICAR HEADER WCS ------------------
    print(f"   [Astrometry.net] Combinando WCS con la imagen original...")

    # Abrir la imagen original
    with fits.open(input_fits, mode="readonly") as orig_hdu:
        data = orig_hdu[0].data
        orig_hdr = orig_hdu[0].header.copy()

    # Abrir la solución WCS
    with fits.open(tmp_wcs) as wcs_hdu:
        wcs_hdr = wcs_hdu[0].header

    # Definir los keywords críticos
    critical_keys = [
        "CRPIX1","CRPIX2",
        "CRVAL1","CRVAL2",
        "CTYPE1","CTYPE2",
        "CUNIT1","CUNIT2",
        "CD1_1","CD1_2","CD2_1","CD2_2",
        "PC1_1","PC1_2","PC2_1","PC2_2",
        "EQUINOX"
    ]

    # Copiar solo los keywords críticos al header original
    for key in critical_keys:
        if key in wcs_hdr:
            orig_hdr[key] = wcs_hdr[key]
    orig_hdr['ASTROMET'] = ('yes', 'Astrometric solution applied')

    # Guardar la imagen final
    output_fits = output_path / f"{input_fits.stem}_wcs.fits"
    fits.writeto(output_fits, data, orig_hdr, overwrite=True)

    print(f"   [Astrometry.net] Imagen final guardada en: {output_fits}")

    # Limpiar archivo temporal
    if tmp_wcs.exists():
        tmp_wcs.unlink()


    return output_fits

def run_astrometry(dataset, output_dir, api_key, overwrite=True):
    """
    Aplica astrometría a todas las imágenes científicas del dataset
    que no tengan WCS.
    """

    output_dir = Path(output_dir)
    images = dataset.get("images_flat", []) 
    if len(images) == 0:
        print("   → No science images with flat correction")
    
    results = []

    for img in images:
        img = Path(img)

        out = output_dir / f"{img.stem}_wcs.fits"

        if out.exists() and not overwrite:
            print(f"   → Skipping existing {out.name}")
            continue

        try:
            result = apply_astrometry(img, output_dir, api_key)
            results.append(result)
        except Exception as e:
            print(f"   → Failed on {img.name}: {e}")

    return results