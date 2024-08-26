# pht-ml

Identify long period exoplanets from TESS light curves using deep learning. See paper for more details.

## Data

We use volunteer scores from Planet Hunters TESS as soft labels for training the deep learning models. See `src/data.py` for details.

### Preprocessing Pipeline

This repo uses preprocessed PDCSAP_FLUX for the input light curves from .fits files/ See `src/utils/preprocess_lcs.py` for usage. We use a binning factor of 7. These raw fluxes are then median normalized and centred on zero. 

## Usage

All hyperparameters are controlled via command line arguments. Use `python src/main.py` to run an experiment. 


## Related Work

- [Astronet](https://iopscience.iop.org/article/10.3847/1538-3881/aa9e09/meta)
- [Ramjet](https://iopscience.iop.org/article/10.3847/1538-3881/abf4c6#ajabf4c6s3)
- [Planet Hunters Tess](https://academic.oup.com/mnras/article/501/4/4669/6027708#225349447)
