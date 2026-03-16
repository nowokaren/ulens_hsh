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

import shutil
from pathlib import Path

def version_config_file(config_file, night_dir):
    config_file = Path(config_file)
    night_dir = Path(night_dir)

    base = config_file.stem

    existing = sorted(night_dir.glob(f"{base}_v*.yaml"))

    if not existing:
        version = 1
    else:
        nums = [int(f.stem.split("_v")[-1]) for f in existing]
        version = max(nums) + 1

    new_name = f"{base}_v{version}.yaml"
    dest = night_dir / new_name

    shutil.copy(config_file, dest)

    return new_name
    

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


'''
def append_last_row(
    csv_origen,
    csv_destino,
    n_rows = 1
):
    csv_origen = Path(csv_origen)
    csv_destino = Path(csv_destino)


    df_src = pd.read_csv(csv_origen)

    if df_src.empty:
        raise ValueError("El CSV origen está vacío")

    last_row = df_src.tail(n_rows)

    if csv_destino.exists():
        df_dst = pd.read_csv(csv_destino)
        all_cols = sorted(set(df_dst.columns) | set(last_row.columns))
        df_dst = df_dst.reindex(columns=all_cols)
        last_row = last_row.reindex(columns=all_cols)
        df_out = pd.concat([df_dst, last_row], ignore_index=True)
    else:
        df_out = last_row.copy()
    df_out.to_csv(csv_destino, index=False)
    return df_out
'''

def append_last_row(
    csv_origen,
    csv_destino,
    n_rows=1,
    config_version=None
):

    csv_origen = Path(csv_origen)
    csv_destino = Path(csv_destino)

    df_src = pd.read_csv(csv_origen)

    if df_src.empty:
        raise ValueError("El CSV origen está vacío")

    last_row = df_src.tail(n_rows).copy()

    if config_version is not None:
        last_row["CONFIG_FILE"] = config_version

    if csv_destino.exists():

        df_dst = pd.read_csv(csv_destino)

        # si la columna no existe la creamos
        if "CONFIG_FILE" not in df_dst.columns:
            df_dst["CONFIG_FILE"] = None

        all_cols = sorted(set(df_dst.columns) | set(last_row.columns))

        df_dst = df_dst.reindex(columns=all_cols)
        last_row = last_row.reindex(columns=all_cols)

        df_out = pd.concat([df_dst, last_row], ignore_index=True)

    else:
        if "CONFIG_FILE" not in last_row.columns:
            last_row["CONFIG_FILE"] = config_version
        df_out = last_row.copy()

    df_out.to_csv(csv_destino, index=False)

    return df_out





