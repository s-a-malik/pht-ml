
import os

from glob import glob

import astropy.io.fits as pf
from tqdm.autonotebook import trange

import numpy as np
import pandas as pd


def preprocess_lcs(lc_root_path, save_path, sectors):
    """Preprocesses light curves into csv files with names containing the metadata.
    Loads fits files from each sector, takes PDCSAP_FLUX and time and saves to csv files, no additional preprocessing.
    Metadata in the file name in case this is useful: tic_id, sector, samera, ccd, tess magnitude, effective temperature, radius
    File naming convention: {save_path}/Sector{sector}/tic-{tic_id}_sec-{sector}_cam-{camera}_chi-{ccd}_tessmag-{}_teff-{teff}_srad-{radius}.csv
    Params:
    - lc_root_path (str): path to lc fits files
    - save_path (str): path to save csv files
    - sectors (List(int)): sectors to preprocess
    """

    for sector in sectors:
        print(f"Preprocessing sector {sector}")
        # make the directory if it doesn't exist
        os.makedirs(os.path.join(save_path, "Sector{}".format(sector)), exist_ok=True)
        # get the list of files in the sector
        fits_files = glob(os.path.join(lc_root_path, f"planethunters/Rel{sector}/Sector{sector}/**/*.fit*"), recursive=True)
        print(f"Found {len(fits_files)} files")
        with trange(len(fits_files)) as t: 
            for i, fits_file in enumerate(fits_files):
                # read the file
                time, flux, file_name = _read_lc(fits_file)
                if time is None:
                    continue
                # save the file
                file_name = os.path.join(save_path, "Sector{}".format(sector), file_name)
                np.savetxt(file_name, flux, delimiter=",") # csv
                t.update()
                if i == 2:
                    break



def _read_lc(lc_file):
    """Read light curve file (copy from data.py)
    Returns:
    - time (np.array): time array
    - flux (np.array): flux array
    - file_name (str): file name
    """
    # open the file in context manager - catching corrupt files
    try:
        with pf.open(lc_file) as hdul:
            d = hdul[1].data
            time = d["TIME"]   # currently not using time
            flux = d["PDCSAP_FLUX"]  # the processed flux
            
            t0 = time[0]  # make the time start at 0 (so that the timeline always runs from 0 to 27.8 days)
            time -= t0

            tic = int(hdul[0].header["TICID"])
            sec = int(hdul[0].header["SECTOR"])
            cam = int(hdul[0].header["CAMERA"])
            chi = int(hdul[0].header["CCD"])
            tessmag = hdul[0].header["TESSMAG"]
            teff = hdul[0].header["TEFF"]
            srad = hdul[0].header["RADIUS"]
            file_name = f"tic-{tic}_sec-{sec}_cam-{cam}_chi-{chi}_tessmag-{tessmag}_teff-{teff}_srad-{srad}.csv"
    except:
        print("Error in fits file: ", lc_file)
        return None, None, None

    return time, flux, file_name



if __name__ == "__main__":
    LC_ROOT_PATH = "/mnt/zfsusers/shreshth/pht_project/data/TESS"
    LABELS_ROOT_PATH = "/mnt/zfsusers/shreshth/pht_project/data/pht_labels"
    SAVE_PATH = "/mnt/zfsusers/shreshth/pht_project/data/lc_csvs"
    SECTORS = [10]

    preprocess_lcs(LC_ROOT_PATH, SAVE_PATH, SECTORS)
