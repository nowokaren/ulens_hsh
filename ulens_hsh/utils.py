from pathlib import Path
from astropy.coordinates import SkyCoord
import astropy.units as u
import yaml
import pandas as pd
import sys

def load_target_coordinates(objname, objects_csv = "objetos.csv"):
    """
    Lee automáticamente la RA/Dec del objeto de interés desde un archivo CSV.

    Acepta dos formatos:
    - decimal:     RA DEC  (en grados)
    - sexagesimal: HH:MM:SS  DD:MM:SS

    Devuelve RA, Dec en grados (floats).
    """

    posiciones = pd.read_csv(objects_csv).query(f"objeto == '{objname}'")
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





