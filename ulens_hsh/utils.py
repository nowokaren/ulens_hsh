from pathlib import Path
from astropy.coordinates import SkyCoord
import astropy.units as u
import yaml
import pandas as pd
import sys

def load_target_coordinates(imdir):
    """
    Lee automáticamente la RA/Dec del objeto de interés desde:
    ../../objetos/<OBJETO>/<OBJETO>.coo

    Acepta dos formatos:
    - decimal:     RA DEC  (en grados)
    - sexagesimal: HH:MM:SS  DD:MM:SS

    Devuelve RA, Dec en grados (floats).
    """

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
    
def load_config(config_file):
    with open(config_file, "r") as f:
        return yaml.safe_load(f)

    

class Tee:
    """
    Duplica stdout: imprime en pantalla y escribe en archivo.
    """
    def __init__(self, filename):
        self.file = open(filename, "a", buffering=1)  # append
        self.stdout = sys.stdout

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)

    def flush(self):
        self.stdout.flush()
        self.file.flush()





