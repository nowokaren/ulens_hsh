import time
import json
import requests
from pathlib import Path
from astropy.io import fits

def apply_astrometry(input_fits, output_path, api_key="AN_API_KEY"):
    """
    Sube una imagen FITS a Astrometry.net, obtiene el WCS y genera
    un archivo <original>_wcs.fit con la solución astrométrica aplicada
    en el directorio 'output_path'.

    Parámetros
    ----------
    input_fits : str o Path
        Ruta del archivo FITS original.

    output_path : str o Path
        Carpeta donde guardar el archivo resultante con WCS.

    api_key : str
        API key de Astrometry.net.
    """

    input_fits = Path(input_fits)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # LOGIN
    # ------------------------------------------------------------
    print(f"[Astrometry.net] Iniciando sesión...")
    r = requests.post(
        "http://nova.astrometry.net/api/login",
        data={"request-json": json.dumps({"apikey": api_key})}
    ).json()

    session = r.get("session")
    if not session:
        raise RuntimeError("No se pudo iniciar sesión en Astrometry.net")
    print(f"[Astrometry.net] Sesión iniciada OK")

    # ------------------------------------------------------------
    # SUBIR IMAGEN
    # ------------------------------------------------------------
    print(f"[Astrometry.net] Subiendo imagen {input_fits.name}...")
    files = {"file": open(input_fits, "rb")}
    payload = {"request-json": json.dumps({"session": session})}

    r = requests.post("http://nova.astrometry.net/api/upload",
                      files=files, data=payload).json()

    if "subid" not in r:
        raise RuntimeError(f"Error al subir la imagen: {r}")

    subid = r["subid"]
    print(f"[Astrometry.net] Imagen subida (SUBID {subid})")

    # ------------------------------------------------------------
    # ESPERAR A QUE SE CREE EL JOB
    # ------------------------------------------------------------
    print(f"[Astrometry.net] Esperando asignación de JOB...")
    jobid = None
    while jobid is None:
        time.sleep(5)
        jobs = requests.get(
            f"http://nova.astrometry.net/api/submissions/{subid}"
        ).json().get("jobs", [])

        if jobs and jobs[0] is not None:
            jobid = jobs[0]

    print(f"[Astrometry.net] JOB asignado: {jobid}")

    # ------------------------------------------------------------
    # ESPERAR A QUE TERMINE EL PROCESO
    # ------------------------------------------------------------
    print(f"[Astrometry.net] Resolviendo campo...")

    status = "processing"
    while status == "processing":
        time.sleep(4)
        status = requests.get(
            f"http://nova.astrometry.net/api/jobs/{jobid}"
        ).json().get("status")

    if status != "success":
        raise RuntimeError(f"Astrometry.net no pudo resolver la imagen (status={status})")

    print(f"[Astrometry.net] Solución WCS obtenida!")

    # ------------------------------------------------------------
    # DESCARGAR HEADER WCS
    # ------------------------------------------------------------
    wcs_url = f"http://nova.astrometry.net/wcs_file/{jobid}"
    print(f"[Astrometry.net] Descargando solución WCS...")
    wcs_raw = requests.get(wcs_url).content

    tmp_wcs = input_fits.with_name(input_fits.stem + "_tmp_wcs.fit")
    with open(tmp_wcs, "wb") as f:
        f.write(wcs_raw)

    # ------------------------------------------------------------
    # APLICAR EL HEADER WCS AL FITS ORIGINAL
    # ------------------------------------------------------------
    print(f"[Astrometry.net] Combinando WCS con la imagen original...")

    orig_hdu = fits.open(input_fits)
    data = orig_hdu[0].data
    hdr = orig_hdu[0].header

    wcs_header = fits.getheader(tmp_wcs)

    # claves que podrían aparecer, incluyen SIP opcional
    wcs_keys = [
        "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
        "CTYPE1", "CTYPE2",
        "CD1_1", "CD1_2", "CD2_1", "CD2_2",
        "CDELT1", "CDELT2",
        "CROTA1", "CROTA2",
        "A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER"
    ]

    # agregar claves SIP y PV si existen
    wcs_keys += [k for k in wcs_header.keys() if k.startswith(("A_", "B_", "AP_", "BP_", "PV"))]

    for key in wcs_keys:
        if key in wcs_header:
            hdr[key] = wcs_header[key]

    # ------------------------------------------------------------
    # GUARDAR RESULTADO FINAL EN output_path
    # ------------------------------------------------------------
    output_fits = output_path / f"{input_fits.stem}_wcs.fit"
    fits.writeto(output_fits, data, hdr, overwrite=True)

    print(f"[Astrometry.net] Imagen final guardada en: {output_fits}")

    # limpiar temporales
    tmp_wcs.unlink(missing_ok=True)

    return output_fits

