# pht-ml

Identify long period exoplanets from TESS light curves using deep learning.


## Data

TESS data (link). Trained using

### Preprocessing Pipeline

Binned, median, subtracted etc.


## Usage

All hyperparameters are controlled via command line arguments. 


## General TODO for code

- [x] get to max GPU utilisation
- [ ] Seperate out injected transit data
- [ ] add some heuristics for easy/hard classifications (e.g. depth of transit, SNR for stars). then add this to the loss?
- [ ] Add example light curves, targets, predictions from val set as logs on wandb to help debugging


## Related Papers and Repos

- Astronet
- Ramjet
- 