# pht-ml

This repository contains code to train machine learning models to classify planetary transits from TESS light curves.

## Data

### Light Curves and Synthetic Data

.fits files were downloaded from the [TESS archive](https://archive.stsci.edu/missions-and-data/tess). We use the ETE-6 dataset to generate synthetic data.

### Targets

We primarily use [Planet Hunters TESS](https://mast.stsci.edu/phad/) aggregated volunteer scores as targets to train the network.

### Preprocessing Pipeline

This repo uses preprocessed PDCSAP_FLUX for the input light curves from .fits files. See `src/utils/preprocess_lcs.py` for usage. We use a binning factor of 7. These raw fluxes are then median normalized and centred on zero, and a series of augmentations are used for training purposes (see `data.py` and `transforms.py`).

## Models

We use a 1-D CNN we call *PlaNet*, inspired by the [Ramjet](https://iopscience.iop.org/article/10.3847/1538-3881/abf4c6#ajabf4c6s3) architecture. See `src/models` for implementations.

## Usage

Install the requirements, and login to wandb (for logging): `pip install -r requirements.txt; wandb login <YOUR_API_KEY>`.

All experimental conditions and hyperparameters are set via command line arguments (for example, you can vary the proportion of synthetic data used to train the network). Use `python src/main.py` to run an experiment.

## Related Work

- [Astronet](https://iopscience.iop.org/article/10.3847/1538-3881/aa9e09/meta)
- [Ramjet](https://iopscience.iop.org/article/10.3847/1538-3881/abf4c6#ajabf4c6s3)
- [Planet Hunters Tess](https://academic.oup.com/mnras/article/501/4/4669/6027708#225349447)
