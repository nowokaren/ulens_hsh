from fits_io import scan_dataset, update_headers, dataset_metadata, object_in_fov
from fits_io import identify_objects, cleanup_intermediate_files 
from fits_io import flag_object_in_fov, load_dataset_objects
from reduccion import run_reduction, plot_reduction
from astrometria import run_astrometry
from stars_catalog import generate_refcat
import pandas as pd
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
objects_csv = Path(BASE, "objetos.csv")

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

#objects = identify_objects(dataset["images_raw"])


# -------------------------------------------------------------------------
# 2) Update headers
# -------------------------------------------------------------------------
if steps.get("update_headers", False):
    print("→ Updating FITS headers")
    update_headers(dataset["all"], gain=gain, rdnoise=rdnoise)

# -------------------------------------------------------------------------
# 3) Export metadata
# -------------------------------------------------------------------------
if steps.get("export_metadata", False):
    output_file = cfg["metadata"].get("dataset_metadata_file", "images_data.csv")
    print(f"→ Exporting metadata to {output_file}")
    raw_metadata_file = dataset_metadata(dataset, night_dir, output_file=output_file,
                                         objects_csv="objetos.csv")
    # Load objects list
    objects = load_dataset_objects(night_dir, output_file)

    

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
    for objname in objects:
        print(f"   → Object: {objname}")

        plot_reduction(
            dataset=output_file,
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
dataset = scan_dataset(night_dir)
metadata_file = dataset_metadata(dataset, night_dir,
                                output_file=output_file)
# -------------------------------------------------------------------------
# 5.5) Images contains its object?
# -------------------------------------------------------------------------
if steps.get("contains_obj", False):
    print("→ Flagging not contained images")
    df = flag_object_in_fov(metadata_file,
                       objects_csv,
                       night_dir,
                       overwrite=False)
metadata_file = dataset_metadata(dataset, night_dir,
                                output_file=output_file)
                       
                       
# -------------------------------------------------------------------------
# 6) Combination
# -------------------------------------------------------------------------
if steps.get("combination", False):
    print("→ Combining images per object and filter")
    filters = cfg.get("filters", ["I", "V"])
    for objname in objects:
        print(f"   → Object: {objname}")
        
        for filt in filters:
            print(f"      → Filter: {filt}")

'''
# -------------------------------------------------------------------------
# 6) Cleanup
# -------------------------------------------------------------------------
if steps.get("cleanup", False):
    print("→ Cleaning intermediate files")
    cleanup_intermediate_files(data_dir)
'''
print("✓ Pipeline finished successfully")



