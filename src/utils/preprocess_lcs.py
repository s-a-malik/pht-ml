"""Preprocess LC and injected planet files for reducing compute/memory requirements during training.
"""
import os

from glob import glob

import astropy.io.fits as pf
from tqdm.autonotebook import trange

import numpy as np
import pandas as pd


def preprocess_planets_flux(planet_root_path, save_path, bin_factor):
    """Bin the planet fluxes into a csv file.
    Params:
    - planet_root_path (str): path to planet fits files
    - save_path (str): path to save csv files
    - bin_factor (int): bin factor for light curves
    """
    os.makedirs(save_path, exist_ok=True)
    planet_files = glob(f"{planet_root_path}/Planets_*.txt")
    print(f"Found {len(planet_files)} files")
    with trange(len(planet_files)) as t: 
        for planet_file in planet_files:
            # read the file
            flux = np.genfromtxt(planet_file, delimiter=',')
            
            # bin flux
            N = len(flux)
            n = int(np.floor(N / bin_factor) * bin_factor)
            X = np.zeros((1, n))
            X[0, :] = flux[:n]
            Xb = rebin(X, (1, int(n / bin_factor)))
            flux_binned = Xb[0]

            _, file_name = os.path.split(planet_file)
            file_name += f"_binfac-{bin_factor}.csv"

            # save the file
            file_name = os.path.join(save_path, file_name)
            pd.DataFrame({"flux": flux_binned}).to_csv(file_name, index=False)
            # np.savetxt(file_name, flux, delimiter=",") # csv
            t.update()



def preprocess_lcs(lc_root_path, save_path, sectors, bin_factor):
    """Preprocesses light curves into csv files with names containing the metadata.
    Loads fits files from each sector, takes PDCSAP_FLUX and time, bins flux, and saves to csv files. 
    Metadata in the file name in case this is useful: tic_id, sector, samera, ccd, tess magnitude, effective temperature, radius
    File naming convention: {save_path}/Sector{sector}/tic-{tic_id}_sec-{sector}_cam-{camera}_chi-{ccd}_tessmag-{}_teff-{teff}_srad-{radius}_binfac-{bin_factor}.csv
    Params:
    - lc_root_path (str): path to lc fits files
    - save_path (str): path to save csv files
    - sectors (List(int)): sectors to preprocess
    - bin_factor (int): bin factor for light curves
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
                
                # bin flux
                N = len(time)
                n = int(np.floor(N / bin_factor) * bin_factor)
                X = np.zeros((2, n))
                X[0, :] = time[:n]
                X[1, :] = flux[:n]
                Xb = rebin(X, (2, int(n / bin_factor)))
                time_binned = Xb[0]
                flux_binned = Xb[1]

                file_name += f"_binfac-{bin_factor}.csv"

                # save the file
                file_name = os.path.join(save_path, "Sector{}".format(sector), file_name)
                pd.DataFrame({"time": time_binned, "flux": flux_binned}).to_csv(file_name, index=False)
                # np.savetxt(file_name, flux, delimiter=",") # csv
                t.update()


def rebin(arr, new_shape):
    """Function to bin the data to make it easier to visualise.
    """
    shape = (
        new_shape[0],
        arr.shape[0] // new_shape[0],
        new_shape[1],
        arr.shape[1] // new_shape[1],
    )
    return arr.reshape(shape).mean(-1).mean(1)


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
            file_name = f"tic-{tic}_sec-{sec}_cam-{cam}_chi-{chi}_tessmag-{tessmag}_teff-{teff}_srad-{srad}"
    except:
        print("Error in fits file: ", lc_file)
        return None, None, None

    return time, flux, file_name



if __name__ == "__main__":
    LC_ROOT_PATH = "/mnt/zfsusers/shreshth/pht_project/data/TESS"
    PLANETS_ROOT_PATH = "/mnt/zfsusers/shreshth/kepler_share/kepler2/TESS/ETE-6/injected/Planets"
    LABELS_ROOT_PATH = "/mnt/zfsusers/shreshth/pht_project/data/pht_labels"
    SAVE_PATH = "/mnt/zfsusers/shreshth/pht_project/data/lc_csvs"
    PLANETS_SAVE_PATH = "/mnt/zfsusers/shreshth/pht_project/data/planet_csvs"
    # SECTORS = [10]
    SECTORS = list(range(10, 15))
    # SECTORS = [37]
    BIN_FACTOR = 3

    # preprocess_planets_flux(PLANETS_ROOT_PATH, PLANETS_SAVE_PATH, BIN_FACTOR)
    preprocess_lcs(LC_ROOT_PATH, SAVE_PATH, SECTORS, BIN_FACTOR)

