from fits_io import scan_dataset, update_headers, dataset_metadata
from fits_io import identify_objects, cleanup_intermediate_files
from reduccion import run_reduction, plot_reduction
from astrometria import run_astrometry

from pathlib import Path
from utils import load_config, Tee
from datetime import datetime
import sys


# -----------------------------------------------------------------------------
# Load settings
# -----------------------------------------------------------------------------

config_file="config.yaml"
cfg = load_config(config_file)

BASE = Path(cfg["paths"]["BASE"])
data_dir = Path(BASE, cfg["paths"]["data_dir"])
object_dir = Path(BASE, cfg["paths"]["objects_dir"])
night = cfg["night"]
night_dir = Path(BASE, data_dir, night)

gain = cfg["instrument"]["gain"]
rdnoise = cfg["instrument"]["rdnoise"]

steps = cfg["steps"]

print(f"Running pipeline in night dir: {night_dir}")
# -------------------------------------------------------------------------
# Redirect stdout/stderr to a single log file (append)
# -------------------------------------------------------------------------
log_file = night_dir / "pipeline.log"
sys.stdout = Tee(log_file)
sys.stderr = sys.stdout

print("\n"*4 + "="*80)
print(f"New pipeline run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)


# -------------------------------------------------------------------------
# 1) Scan dataset
# -------------------------------------------------------------------------
dataset = scan_dataset(night_dir)

objects = identify_objects(dataset["images_raw"])


# -------------------------------------------------------------------------
# 2) Update headers
# -------------------------------------------------------------------------
if steps.get("update_headers", False):
    print("→ Updating FITS headers")
    update_headers(dataset["all"], gain=gain, rdnoise=rdnoise)

# -------------------------------------------------------------------------
# 3) Export metadata (and correct OBJECT name in header)
# -------------------------------------------------------------------------
if steps.get("export_metadata", False):
    output_file = cfg["metadata"].get("dataset_metadata_file", "images_data.csv")
    print(f"→ Exporting metadata to {output_file}")
    raw_metadata_file = dataset_metadata(dataset, night_dir, output_file=output_file,
                                         objects_csv="objetos.csv")

# -------------------------------------------------------------------------
# 4) Reduction
# -------------------------------------------------------------------------
if steps.get("reduction", False):
    print("→ Running reduction")

    reduction_cfg = cfg.get("reduction", {})
    reduction_result = run_reduction(
        dataset=dataset,
        zero_correction=reduction_cfg.get("zero_correction", True),
        flat_correction=reduction_cfg.get("flat_correction", True),
        dark_correction=reduction_cfg.get("dark_correction", False),
    )
else:
    reduction_result = None
    
# -------------------------------------------------------------------------
# 4.a) Plot de reducción
# -------------------------------------------------------------------------
if cfg["qc"].get("reduction_images", False):
    print("→ Generating reduction plots")


    for objname, images in objects.items():
        print(f"   → Object: {objname}")

        plot_reduction(
            dataset=dataset,
            night_dir=night_dir,
            objname=objname,
            output_name=f"reduction_{objname}.png",
            show=False
        )
    

# -------------------------------------------------------------------------
# 5) Astrometry
# -------------------------------------------------------------------------
if steps.get("astrometry", False):
    print("→ Running astrometry")

    astro_cfg = cfg.get("astrometry", {})
    run_astrometry(
        dataset=dataset,
        output_dir=night_dir,
        api_key=astro_cfg["api_key"],
        overwrite=astro_cfg.get("overwrite", True),
    )

'''
# -------------------------------------------------------------------------
# 6) Cleanup
# -------------------------------------------------------------------------
if steps.get("cleanup", False):
    print("→ Cleaning intermediate files")
    cleanup_intermediate_files(data_dir)
'''
print("✓ Pipeline finished successfully")



