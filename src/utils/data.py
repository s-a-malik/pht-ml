"""data.py
Utility functions and classes for data manipulation.
"""

import os
import time
import glob

import torch

import numpy as np
import pandas as pd

import astropy.io.fits as pf


SECTORS = list(range(10, 39))
# SECTORS = [10, 11]
SHORTEST_LC = 17546

class LCData(torch.utils.data.Dataset):
    """Light curve dataset
    """

    def __init__(self, lc_root_path, labels_root_path, transform=None):
        super(LCData, self).__init__()

        self.lc_root_path = lc_root_path
        self.labels_root_path = labels_root_path
        self.transform = transform

        # get list of all lc files
        self.lc_file_list = []
        for sector in SECTORS:
            print(f"sector: {sector}")
            new_files = glob.glob(os.path.join(lc_root_path, f"Rel{sector}/Sector{sector}/**/*.fit*"), recursive=True)
            print("num. files found: ", len(new_files))
            self.lc_file_list += new_files

            # length of light curve in this sector:
            x, tic, sec = self._read_lc(self.lc_file_list[-1])
            print("length of light curve in sector ", sec, ": ", x.shape, ", TIC: ", tic)

        # get all the labels
        self.labels_df = pd.DataFrame()
        for sector in SECTORS:
            self.labels_df = pd.concat([self.labels_df, pd.read_csv(f"{labels_root_path}/summary_file_sec{sector}.csv")], axis=0)
        print("num. labels: ", len(self.labels_df))
        # TODO add simulated data

    def __len__(self):
        return len(self.lc_file_list)


    def __getitem__(self, idx):
        start = time.time()
        # get lc file
        lc_file = self.lc_file_list[idx]

        # read lc file
        x, tic, sec = self._read_lc(lc_file)

        # TODO fill in nans more effectively in the light curve
        x = torch.nan_to_num(x)

        # TODO make all the same length - each sector is slightly different length (could do this in the collate fn also)
        x = x[:SHORTEST_LC]

        # get label for this lc file (if exists) match sector 
        # filter by sector, then TIC ID
        # print(self.labels_df.loc[(self.labels_df["TIC_ID"] == tic) & (self.labels_df["sector"] == sec), "maxdb"])
        y = self.labels_df.loc[(self.labels_df["TIC_ID"] == tic) & (self.labels_df["sector"] == sec), "maxdb"].values[0]
        print(f"tic: {tic}, sec: {sec}, x.shape: {x.shape}, x: {x}, y: {y}")
        # TODO check if label does not exist
        if not y:
            print(y, "label not found for TIC: ", tic, " in sector: ", sec)

        end = time.time()
        print("time to get data", end - start)
        # return tensors
        return (torch.from_numpy(x), torch.tensor(tic), torch.tensor(sec)), torch.tensor(y)

    def _read_lc(self, lc_file):
        """Read light curve file
        """
            # open the file in context manager
        with pf.open(lc_file) as hdul:
            d = hdul[1].data
            t = d["TIME"]
            f2 = d["PDCSAP_FLUX"]  # the processed flux
            # print(f2)
            f2 /= np.nanmedian(f2)
            # print(f2)
            f2 = f2.astype(np.float64)  # to fix numpy => torch byte error
            # print(f2)

            t0 = t[0]  # make the time start at 0 (so that the timeline always runs from 0 to 27.8 days)
            t -= t0

            tic = int(hdul[0].header["TICID"])
            sec = int(hdul[0].header["SECTOR"])
            cam = int(hdul[0].header["CAMERA"])
            chi = int(hdul[0].header["CCD"])
            tessmag = hdul[0].header["TESSMAG"]
            teff = hdul[0].header["TEFF"]
            srad = hdul[0].header["RADIUS"]

        return f2, tic, sec

    # load file path lists and count number of curves (save in an array?)

    # split these into train, val, test by index. 

    # then on get item, load the file. (need to ensure loading is fast - preload batches)

# TODO function to get file from tic id - needed for specific lookup 


# TODO collate fn to return a good batch of simulated and real data
def collate_fn(batch):
    pass


# transformations of light curves to make them consistent


# splitting of data into train, val, test use file path lists.


def get_data_loader(data_root_path, labels_root_path, batch_size, shuffle=True, num_workers=0, pin_memory=False):
    """Get data loader
    """
    dataset = LCData(data_root_path, labels_root_path)
    data_loader = torch.utils.data.DataLoader(dataset,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                num_workers=num_workers,
                                                pin_memory=pin_memory)
    return data_loader




if __name__ == "__main__":
    
    LC_ROOT_PATH = "/mnt/zfsusers/shreshth/pht_project/data/TESS/planethunters"
    LABELS_ROOT_PATH = "/mnt/zfsusers/shreshth/pht_project/data/pht_labels"

    # lc_data = LCData(LC_ROOT_PATH, LABELS_ROOT_PATH)
    # print(len(lc_data))
    data_loader = get_data_loader(LC_ROOT_PATH, LABELS_ROOT_PATH, batch_size=4, shuffle=False, num_workers=0, pin_memory=False)
    for i, (x, y) in enumerate(data_loader):
        # print(i, x, y)
        # if i == 5:
        #     break



    # use pin_memory to speed up loading on GPU
