# autophot usage in HSH pipeline

This pipeline relies on a locally modified version of `autophot`
(https://github.com/Astro-Sean/autophot).

## Installation method

`autophot` is installed in editable mode from a local clone:

```bash
pip install -e /home/knowogrodski/Documentos/observation_HSH/process_HSH_SN/autophot

## Version and provenance

- **Upstream repository:** Astro-Sean/autophot  
- **Local branch:** `hsh-fixes`  
- **Base branch:** `master`  
- **Commit hash:** 4bf4ced5f6dbc0a8f8a00881ea02e950c188eb2e

## Local modifications

The following files were modified to support HSH data:

- `autophot/packages/aperture.py`
- `autophot/packages/psf.py`
- `autophot/packages/call_catalog.py`
- `autophot/packages/run.py`
- `autophot/packages/main.py`

## Rationale

These modifications address:

- Aperture handling for HSH images  
- Catalog query robustness  
- Pipeline stability for HSH data
