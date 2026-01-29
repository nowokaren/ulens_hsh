from fits_io import scan_dataset, update_headers, dataset_metadata, object_in_fov
from fits_io import identify_objects, cleanup_intermediate_files 
from fits_io import flag_object_in_fov, load_dataset_objects
from reduccion import run_reduction, plot_reduction
from astrometria import run_astrometry
from stars_catalog import generate_refcat, plot_catalog_on_image
from combine import process_and_combine_images, plot_combined, remove_aligment_tempfiles, plot_aligment
import pandas as pd
from pathlib import Path
import os
from utils import load_config, Tee, load_target_coordinates, append_last_row
from datetime import datetime
import sys
from autophot.autophot_main import run_automatic_autophot
import tempfile
import shutil
from autophot.prep_input import load





# -----------------------------------------------------------------------------
# Load settings
# -----------------------------------------------------------------------------

config_file="config.yaml"
cfg = load_config(config_file)

BASE = Path(cfg["paths"]["BASE"])
data_path = Path(BASE, cfg["paths"]["data_dir"])
object_path = Path(data_path, cfg["paths"]["objects_dir"])
night = cfg["night"]
night_dir = Path(data_path, night)
objects_csv = Path(object_path, "objetos.csv")
images_file_name = cfg["paths"]["images_data_file"]
images_file = Path(night_dir, images_file_name)

gain = cfg["instrument"]["gain"]
rdnoise = cfg["instrument"]["rdnoise"]

steps = cfg["steps"]

print(f"Running pipeline in night dir: {night_dir}")
# -------------------------------------------------------------------------
# Redirect stdout/stderr to a single log file (append)
# -------------------------------------------------------------------------
log_file = night_dir / "pipeline.log"
#sys.stdout = Tee(log_file)
#sys.stderr = sys.stdout

print("\n"*2 + "="*80)
print(f"New pipeline run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)


# -------------------------------------------------------------------------
# 1) Scan dataset
# -------------------------------------------------------------------------
dataset = scan_dataset(night_dir)

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
    print(f"→ Exporting metadata to {images_file}")
    metadata_file = dataset_metadata(dataset, night_dir, output_file=images_file,
                                         objects_csv=objects_csv)
    # Load objects list
    objects = load_dataset_objects(night_dir, images_file)
    
    # Create a folder in objects dir for each object
    for objname in objects:
        obj_dir = Path(object_path, objname)
        obj_dir.mkdir(parents=True, exist_ok=True)


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
    print("→ [QC] Generating reduction plots")
    for objname in objects:
        print(f"   → Object: {objname}")

        plot_reduction(
            dataset=images_file,
            night_dir=night_dir,
            objname=objname,
            output_name=f"reduction_{objname}.png",
            show=False,
            overwrite=True
        )
    print("✓ [QC] Reduction plots generated")

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
                                    output_file=images_file)
# -------------------------------------------------------------------------
# 5.a) Images contains its object?
# -------------------------------------------------------------------------
if steps.get("contains_obj", False):
    print("→ Flagging not contained images")
    df = flag_object_in_fov(metadata_file,
                       objects_csv,
                       night_dir,
                       overwrite=False)
    metadata_file = dataset_metadata(dataset, night_dir,
                                    output_file=images_file)

# -------------------------------------------------------------------------
# 6) Combination
# -------------------------------------------------------------------------
if steps.get("combine", False):
    print("→ Combining images per object and filter")
    
    filters = cfg.get("filters", ["I", "V"])
    for objname in objects:
        print(f"   → Object: {objname}")
        print(f"      → Generating reference catalog")
        # Load metadata for the object
        ra, dec = load_target_coordinates(objname, objects_csv)
        alig_cfg = cfg.get("aligment", {})["catalog"]
        # Load image path for the object
        ds = pd.read_csv(Path(night_dir, images_file))
        img_files = ds[(ds["OBJECT"]==objname)&(ds["ASTROMET"]=="yes")]["FILENAME"].values
        img_path = night_dir / img_files[0]

        alig_cat_path = generate_refcat(
            objname=objname, ra_center=ra, dec_center=dec,
            img_path=img_path, objects_dir=object_path,
            fov_frac=alig_cfg.get("fov_frac", 0.3),
            min_mag=alig_cfg.get("min_mag", 9),
            max_mag=alig_cfg.get("max_mag", 12),
            use_catalogs=alig_cfg.get("use_catalogs", "all"),
            plot=alig_cfg.get("plot", False),
            overwrite=alig_cfg.get("overwrite", False),
        )

        for filt in filters:
            print(f"      → Filter: {filt}")
            images_to_combine = [night_dir / im for im in img_files if im[-14].upper() == filt.upper()]
            if len(images_to_combine) == 0:
                print(f"         ! No images found for filter {filt}. Skipping.")
                continue
            print(f"         ✓ Found {len(images_to_combine)} images for filter {filt}.")
            combined_im = process_and_combine_images(images_to_combine, objname,  filt, object_path, night_dir, objects_csv)
            # Plot aligment reference catalog on combined images (unnecessary step)
            plot_catalog_on_image(fits_file=combined_im,
                                  catalog=alig_cat_path,
                                  obj_ra=ra,
                                  obj_dec=dec,
                                  out_png=Path(night_dir, f"{objname}{filt}_comb_alig_cat.png"),
                                  title=f"{objname} – Aligment catalog")
            if cfg["qc"].get("aligment_images", False):
                dataset = scan_dataset(night_dir)
                metadata_file = dataset_metadata(dataset, night_dir,
                                    output_file=images_file)
                print("      → Plotting aligment images")
                plot_aligment(
                    dataset=str(images_file),
                    night_dir=night_dir,
                    objname=objname,
                    filter_band=filt,
                    output_name=f"aligment_{objname}.png",
                    show=False,
                    overwrite=True
                )
            if cfg["combine"].get("remove_temp_files", False):
                print("      → Removing temporary files")
                removed = remove_aligment_tempfiles(filt, night_dir)
    dataset = scan_dataset(night_dir)
    metadata_file = dataset_metadata(dataset, night_dir,
                                    output_file=images_file)
    

if cfg["qc"].get("combined_images", False):
    print("→ Generating combined plots")
    for objname in objects:
        print(f"   → Object: {objname}")
        image_files = [file for file in dataset["images_combined"] if objname in file.name]
        if len(image_files) == 0:
            print(f"      ! No combined images found for object {objname}. Skipping.")
            continue  
        plot_combined(objname, image_files,  night_dir, show=False)
        
        
if steps.get("photometry", False):
    print("→ Photometry on science images")
    phot_cfg = cfg["photometry"]
    img_files = []
    if phot_cfg["on_exp"]:
        img_files += dataset["images_astro"]
    if phot_cfg["on_comb"]:
        img_files += dataset["images_combined"]
         
    for objname in objects:
        print(f"   → Object: {objname}")
        print(f"      → Generating reference catalog")
        ra, dec = load_target_coordinates(objname, objects_csv)
        phot_cat_cfg = cfg["photometry"]["catalog"]
        img_path = night_dir / img_files[0]
        phot_cat_path = generate_refcat(
            objname=objname, ra_center=ra, dec_center=dec,
            img_path=img_path, objects_dir=object_path,
            fov_frac=phot_cat_cfg.get("fov_frac", 0.3),
            min_mag=phot_cat_cfg.get("min_mag", 9),
            max_mag=phot_cat_cfg.get("max_mag", 12),
            max_mag_err=phot_cat_cfg.get("max_mag_err", 0.25),
            use_catalogs=phot_cat_cfg.get("use_catalogs", False),
            plot=phot_cat_cfg.get("plot", False),
            type="phot",
            overwrite=phot_cat_cfg.get("overwrite", False)
        )
        print(f"      → Plotting catalog on combined image")
        combined_im = [f for f in dataset["images_combined"] if objname in f.name][0]
        plot_catalog_on_image(fits_file=combined_im,
                        catalog=phot_cat_path,
                        obj_ra=ra,
                        obj_dec=dec,
                        out_png=Path(object_path, objname, f"{objname}_phot_cat.png"),
                        title=f"{objname} – Aligment catalog")
        
        print("   → Photometry settings")
        image_files = []
        if phot_cfg["on_comb"]:
            image_files+=[f for f in dataset["images_combined"] if objname.lower() in str(f).lower()]
        if phot_cfg["on_exp"]:
            image_files+=[f for f in dataset["images_astro"] if objname.lower() in str(f).lower()]

        refcat_csv = os.path.join(BASE, "data", "objetos", objname, f"{objname}_phot_cat.csv")
        fits_dir = os.path.join(BASE, "data", night, objname, "phot")
        os.makedirs(fits_dir, exist_ok=True)
        for im in image_files:
            shutil.copy(im, Path(fits_dir, im.name))
        wdir = os.path.join(BASE, "outputs", night, objname)
        os.makedirs(wdir, exist_ok=True)

        #autophot_input = load()
        autophot_input = phot_cfg["autophot"]
        autophot_input["wdir"] = wdir
        autophot_input["fits_dir"] = fits_dir

        autophot_input["target_name"] = objname
        autophot_input["target_ra"] = ra/15
        autophot_input["target_dec"] = dec

        autophot_input["catalog"]["use_catalog"] = "custom"
        autophot_input["catalog"]["catalog_custom_fpath"] = refcat_csv
        shutil.copy("telescope.yml", Path(wdir, "telescope.yml"))

        try:
            run_automatic_autophot(autophot_input)
            os.chdir(BASE)
            print(f"         ✓ AutoPhOT completado para {objname}")
        except Exception as e:
            print(f"         Error AutoPhOT {objname}: {str(e)}")
            continue

        append_last_row(
            Path(night_dir, objname, "phot_REDUCED","REDUCED.csv"),
            Path(object_path, objname, "lightcurve.csv"),
            n_rows=len(image_files)
        )
        
        # Delete calibrated intermediate science images ? bias, flats
        # Delete duplicated images copied to run photometry?
 
print("✓ Pipeline finished successfully")
