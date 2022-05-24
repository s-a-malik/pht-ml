"""data.py
Utility functions and classes for data manipulation.
"""

import os
import argparse
from ast import literal_eval
import csv
import time
from glob import glob
import functools

from tqdm.autonotebook import trange

import torch
import torchvision

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import AutoMinorLocator

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split as split

import astropy.io.fits as pf
from astropy.table import Table

# import transforms
from utils import transforms
from utils.utils import plot_lc


TRAIN_SECTORS_DEBUG = [10]
TRAIN_SECTORS_FULL = [10,11,12,13]

TEST_SECTORS_DEBUG = [14]
TEST_SECTORS_FULL = [14]

# SHORTEST_LC = 17500 # from sector 10-38. Used to trim all the data to the same length.
SHORTEST_LC = 18900 # binned 7 sector 10-14

#### DATASET CLASSES

class LCData(torch.utils.data.Dataset):
    """Light curve dataset
    """

    def __init__(
        self,
        data_root_path="/mnt/zfsusers/shreshth/pht_project/data",
        data_split="train",
        bin_factor=7,
        synthetic_prob=0.0,
        eb_prob=0.0,
        single_transit_only=True,
        transform=None,
        store_cache=True
        ):
        """
        Params:
        - data_root_path (str): path to data directory 
        - data_split (bool): which data split to load (train/test/train_debug/test_debug)
        - bin_factor (int): binning factor light curves to use
        - synthetic_prob (float): proportion of data to be synthetic transits
        - eb_prob (float): proportion of data to be synthetic eclipsing binaries
        - single_transit_only (bool): only use single transits in synthetic data
        - transform (callable): transform to apply to the data
        - store_cache (bool): whether to store all the data in RAM in advance
        """
        super(LCData, self).__init__()

        self.data_root_path = data_root_path
        self.data_split = data_split
        self.bin_factor = bin_factor
        self.synthetic_prob = synthetic_prob
        self.eb_prob = eb_prob
        self.single_transit_only = single_transit_only
        self.transform = transform
        self.store_cache = store_cache
        
        if self.data_split == "train":
            sectors = TRAIN_SECTORS_FULL
        elif self.data_split == "test":
            sectors = TEST_SECTORS_FULL
        elif self.data_split == "train_debug":
            sectors = TRAIN_SECTORS_DEBUG
        elif self.data_split == "test_debug":
            sectors = TEST_SECTORS_DEBUG
        else:
            raise ValueError(f"Invalid data split {data_split}")

        ####### LC data

        # get list of all lc files
        self.lc_file_list = []
        for sector in sectors:
            # print(f"sector: {sector}")
            new_files = glob(f"{self.data_root_path}/lc_csvs/Sector{sector}/*binfac-{self.bin_factor}.csv", recursive=True)
            print("num. files found: ", len(new_files))
            self.lc_file_list += new_files

        print("total num. LC files found: ", len(self.lc_file_list))

        ####### Label data

        # get all the labels
        self.labels_df = pd.DataFrame()
        for sector in sectors:
            self.labels_df = pd.concat([self.labels_df, pd.read_csv(f"{self.data_root_path}/pht_labels/summary_file_sec{sector}.csv")], axis=0)
        print("num. total labels (including simulated data): ", len(self.labels_df))

        # removing simulated data
        self.labels_df = self.labels_df[~self.labels_df["subject_type"]]
        print("num. real transits labels: ", len(self.labels_df))
        # check how many non-zero labels
        print("num. non-zero labels: ", len(self.labels_df[self.labels_df["maxdb"] != 0.0]))
        print("strong non-zero labels (score > 0.5): ", len(self.labels_df[self.labels_df["maxdb"] > 0.5]))
       
        ##### planetary transits 
        if self.synthetic_prob > 0.0:
            self.pl_data = self._get_pl_data()
            print(f"using {self.synthetic_prob} proportion of synthetic data. Single transit only? {self.single_transit_only}")

        ##### cache data
        if self.store_cache:
            self.cache = {}
            print("filling cache")
            with trange(len(self)) as t:
                for i in range(len(self)):
                    self.__getitem__(i)
                    t.update()


    def __len__(self):
        return len(self.lc_file_list)


    # maxsize is the max number of samples to cache - each LC is ~2MB. If you have a lot of memory, you can increase this
    # @functools.lru_cache(maxsize=1000)  # Cache loaded data
    def __getitem__(self, idx):
        """Returns:
        - x (dict): dictionary with keys:
            - flux (np.array): light curve
            - tic (int): TIC
            - sec (int): sector
            - cam (int): camera
            - chi (int): chi
            - tessmag (float): TESS magnitude
            - teff (float): effective temperature
            - srad (float): stellar radius
            - binfac (float): binning factor
            - tic_inj (int): TIC of injected planet (-1 if not injected)
            - depth (float): transit depth (-1 if not injected)
            - duration (float): transit duration (-1 if not injected)
            - period (float): transit period (-1 if not injected)
        - y (float): volunteer confidence score (1 if synthetic transit)
        """
        # check if we have this data cached
        if self.store_cache:
            if idx in self.cache:
                return self.cache[idx]

        # get lc file
        lc_file = self.lc_file_list[idx]
        x = _read_lc_csv(lc_file)
        # if corrupt return None and skip c.f. collate_fn
        if x is None:
            return {"flux": None}, None

        # get label for this lc file (if exists), match sector 
        y = self.labels_df.loc[(self.labels_df["TIC_ID"] == x["tic"]) & (self.labels_df["sector"] == x["sec"]), "maxdb"].values
        if len(y) == 1:
            y = torch.tensor(y[0], dtype=torch.float)
        elif len(y) > 1:
            # print(y, "more than one label for TIC: ", x["tic"], " in sector: ", x["sec"])
            y = None
            # self.no_label_tics.append((tic, sec))
        else:
            # print(y, "label not found for TIC: ", x["tic"], " in sector: ", x["sec"])
            y = None
            # self.no_label_tics.append((tic, sec))

        # probabilistically add synthetic transits, only if labels are zero.
        if (np.random.rand() < self.synthetic_prob) and (y == 0.0):
            # print("injecting transit")
            pl_inj = self.pl_data[np.random.randint(len(self.pl_data))]
            # print(pl_inj)
            x["flux"] = self._inject_transit(x["flux"], pl_inj["flux"])
            x["tic_inj"] = pl_inj["tic_id"]
            x["depth"] = pl_inj["depth"]
            x["duration"] = pl_inj["duration"]
            x["period"] = pl_inj["period"]
            y = torch.tensor(1.0, dtype=torch.float)
            # plot_lc(x["flux"], save_path=f"/mnt/zfsusers/shreshth/pht_project/data/examples/test_dataloader_unnorm_{idx}.png")

        else:
            x["tic_inj"] = -1
            x["depth"] = -1
            x["duration"] = -1
            x["period"] = -1

        # TODO add EB synthetics

        if self.transform:
            x["flux"] = self.transform(x["flux"])
        # turn into float
        x["flux"] = torch.tensor(x["flux"], dtype=torch.float)
        
        if self.store_cache:
            # add to cache 
            self.cache[idx] = (x, y)

        return x, y


    def _get_pl_data(self):
        """Loads the planetary transits data.
        """
        # simulated transit info
        pl_table = Table.read(f"{self.data_root_path}/planet_csvs/ete6_planet_data.txt", format='ascii',comment='#')
        pl_files = glob(f"{self.data_root_path}/planet_csvs/Planets_*binfac-{self.bin_factor}.csv")
        print(f"found {len(pl_files)} planet flux files for binfac {self.bin_factor}")
        
        # load planetary transits into RAM
        pl_data = []   # list of dicts with metadata
        # TODO better to use a dict to speed up lookup or df to save memory?
        print("loading planet metadata...")
        with trange(len(pl_files)) as t:
            for i, pl_file in enumerate(pl_files):
                # extract tic id
                tic_id = int(pl_file.split("/")[-1].split("_")[1].split(".")[0])
                # check if we should include this planet
                if not self._is_planet_in_data_split(tic_id):
                    continue

                # look up in table
                pl_row = pl_table[pl_table['col1'] == tic_id]

                pl_depth = pl_row['col10'][0]  # transit depth
                pl_dur = pl_row['col9'][0]     # transit duration
                pl_per = pl_row['col3'][0]     # transit period                  
                pl_flux = np.genfromtxt(str(pl_file), skip_header=1)
                if len(pl_flux) == 0:
                    print(f"WARNING: no data for tic {tic_id}", pl_row)
                    print(f"skipping...")
                    continue
                if self.single_transit_only:
                    # take only the transit
                    pl_flux = self._extract_single_transit(pl_flux)
                    if len(pl_flux) == 0:
                        print(f"WARNING: no transit found for tic {tic_id}", pl_row)
                        print(f"skipping...")
                        continue
                
                pl_data.append({"flux": pl_flux, "tic_id": tic_id, "depth": pl_depth, "duration": pl_dur, "period": pl_per})
                t.update()

    
        print(f"Loaded {len(pl_data)} simulated transits for {self.data_split} data split")
        # print("examples", pl_data[-5:])

        return pl_data


    def _is_planet_in_data_split(self, tic_id):
        """Checks if a planet flux should be included in this data for simulation.
        Currently just uses the tic_id to select 1/4th of the available data
        """
        if self.data_split in ["train", "train_debug"]:
            if tic_id % 4 == 0:
                return True
            else:
                return False
        elif self.data_split in ["test", "test_debug"]:
            if tic_id % 4 == 0:
                return False
            else:
                return True


    def _extract_single_transit(self, x):
        """Extract a single transit from the planet flux
        Params:
        - x (np.array): flux of the light curve
        Returns:
        - transit (np.array): extracted single transit (shape variable)
        """
        # print("extracting single transit")
        # get the first dip
        start_idx = np.argmax(x<1)
        # get the end of the dip
        length = np.argmax(x[start_idx:]==1)
        # take one extra from either side
        if start_idx > 0:
            transit = x[start_idx-1:start_idx+length+1]
        else:
            transit = x[start_idx:start_idx+length+1]

        return transit


    def _inject_transit(self, base_flux, injected_flux):
        """Inject a transit into a base light curve. 
        N.B. Need to ensure both fluxes correspond to the same cadence.
        Params:
        - base_flux (np.array): base LC to inject into
        - injected_flux (np.array): transit to inject (different length to base)
        """
        if len(injected_flux) >= len(base_flux):
            injected_flux = injected_flux[:len(base_flux)-1]
        
        # ensure the injected flux is not in a missing data region. Only if single transit as the full curve may have a lot of missing data
        if self.single_transit_only:
            missing_data = True
            while missing_data:
                # add injected flux section to random part of base flux
                start_idx = np.random.randint(0, len(base_flux)-len(injected_flux))
                # check if there is missing data in the injected flux
                # print("checking for missing data")

                # if there is 20% missing data in the transit, try again
                # TODO maybe adjust this parameter?      
                missing_data = np.count_nonzero(np.isnan(base_flux[start_idx:start_idx+len(injected_flux)])) / len(injected_flux) > 0.2
        else:
            start_idx = np.random.randint(0, len(base_flux)-len(injected_flux))

        base_flux[start_idx:start_idx+len(injected_flux)] = base_flux[start_idx:start_idx+len(injected_flux)] * injected_flux

        return base_flux



##### UTILS

# TODO collate fn to return a good batch of simulated and real data (do this from the data loader
def collate_fn(batch):
    """Collate function for filtering out corrupted data in the dataset
    Assumes that missing data are NoneType
    """
    batch = [x for x in batch if x[0]["flux"] is not None]   # filter on missing flux 
    batch = [x for x in batch if x[1] is not None]           # filter on missing labels
    return torch.utils.data.dataloader.default_collate(batch)


def _read_lc_csv(lc_file):
    """Read LC flux from preprocessed csv
    Params:
    - lc_file (str): path to lc_file
    Returns:
    - x (dict): dictionary with keys:
        - flux (np.array): light curve
        - tic (int): TIC
        - sec (int): sector
        - cam (int): camera
        - chi (int): chi
        - tessmag (float): TESS magnitude
        - teff (float): effective temperature
        - srad (float): stellar radius
        - binfac (float): binning factor
    """
    try:
        # read the csv file
        df = pd.read_csv(lc_file)
        # get the flux
        x = {}
        x["flux"] = df["flux"].values

        # parse the file name
        file_name = lc_file.split("/")[-1]
        params = file_name.split("_")
        for i, param in enumerate(params):
            if i == len(params) - 1:
                # remove .csv
                x[param.split("-")[0]] = literal_eval(param.split("-")[1][:-4])
            else:
                x[param.split("-")[0]] = literal_eval(param.split("-")[1])
            # convert None to -1
            x[param.split("-")[0]] = -1 if x[param.split("-")[0]] is None else x[param.split("-")[0]]
    except:
        # print("failed to read file: ", lc_file)
        x = None
    return x


def _read_lc(lc_file):
    """Read light curve .fits file
    """
    # open the file in context manager - catching corrupt files
    try:
        with pf.open(lc_file) as hdul:
            d = hdul[1].data
            t = d["TIME"]   # currently not using time
            f2 = d["PDCSAP_FLUX"]  # the processed flux
            
            t0 = t[0]  # make the time start at 0 (so that the timeline always runs from 0 to 27.8 days)
            t -= t0

            tic = int(hdul[0].header["TICID"])
            sec = int(hdul[0].header["SECTOR"])
            cam = int(hdul[0].header["CAMERA"])
            chi = int(hdul[0].header["CCD"])
            tessmag = hdul[0].header["TESSMAG"]
            teff = hdul[0].header["TEFF"]
            srad = hdul[0].header["RADIUS"]
    except:
        print("Error in fits file: ", lc_file)
        return None, None, None

    return f2, tic, sec

        
def get_data_loaders(args):
    """Get data loaders given argparse arguments
    """
    # unpack arguments
    data_root_path = args.data_path
    val_size = args.val_size
    seed = args.seed
    bin_factor = args.bin_factor
    synthetic_prob = args.synthetic_prob
    eb_prob = args.eb_prob
    batch_size = args.batch_size
    num_workers = args.num_workers
    cache = not args.no_cache
    # max_lc_length = args.max_lc_length
    max_lc_length = int(SHORTEST_LC / bin_factor)
    multi_transit = args.multi_transit
    pin_memory = True
    debug = args.debug

    # composed transform
    training_transform = torchvision.transforms.Compose([
        transforms.NormaliseFlux(),
        transforms.RandomDelete(prob=0.0, delete_fraction=0.1),
        transforms.RandomShift(prob=0.0, permute_fraction=0.1),
        # transforms.BinData(bin_factor=3),  # bin before imputing
        transforms.ImputeNans(method="zero"),
        transforms.Cutoff(length=max_lc_length)
    ])

    # test tranforms - do not randomly delete or permute
    test_transform = torchvision.transforms.Compose([
        transforms.NormaliseFlux(),
        transforms.RandomDelete(prob=0.0, delete_fraction=0.1),
        transforms.RandomShift(prob=0.0, permute_fraction=0.1),
        # transforms.BinData(bin_factor=3),  # bin before imputing
        transforms.ImputeNans(method="zero"),
        transforms.Cutoff(length=max_lc_length)
    ])


    # TODO choose type of data set - set an argument for this (e.g. simulated/real proportions)
    train_dataset = LCData(
        data_root_path=data_root_path,
        data_split="train_debug" if debug else "train",
        bin_factor=bin_factor,
        synthetic_prob=synthetic_prob,
        eb_prob=eb_prob,
        single_transit_only=not multi_transit,
        transform=training_transform,
        store_cache=cache
    )
    indices = [i for i in range(len(train_dataset))]
    train_idx, val_idx = split(indices, random_state=seed, test_size=val_size)
    train_set = torch.utils.data.Subset(train_dataset, train_idx)
    val_set = torch.utils.data.Subset(train_dataset, val_idx)

    test_set = LCData(
        data_root_path=data_root_path,
        data_split="test_debug" if debug else "test",
        bin_factor=bin_factor,
        synthetic_prob=0.5,             # TODO have synthetics in test as well? yes probably
        eb_prob=0.0,
        single_transit_only=not multi_transit,       # irrelevant for test set
        transform=test_transform,
        store_cache=cache
    )

    print(f'Size of training set: {len(train_set)}')
    print(f'Size of val set: {len(val_set)}')
    print(f'Size of test set: {len(test_set)}')

    train_dataloader = torch.utils.data.DataLoader(train_set,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=num_workers,
                                                pin_memory=pin_memory,
                                                collate_fn=collate_fn)
    val_dataloader = torch.utils.data.DataLoader(val_set,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=num_workers,
                                                pin_memory=pin_memory,
                                                collate_fn=collate_fn)
    test_dataloader = torch.utils.data.DataLoader(test_set,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=num_workers,
                                                pin_memory=pin_memory,
                                                collate_fn=collate_fn)

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == "__main__":
    
    # parse data args only
    ap = argparse.ArgumentParser(description="test dataloader")
    ap.add_argument("--data-path", type=str, default="/mnt/zfsusers/shreshth/pht_project/data")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--val-size", type=float, default=0.2)
    ap.add_argument("--bin-factor", type=int, default=7)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--synthetic-prob", type=float, default=0.5)
    ap.add_argument("--eb-prob", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--multi-transit", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    train_dataloader, val_dataloader, test_dataloader = get_data_loaders(args)
    with trange(len(train_dataloader)) as t:
        for i, (x, y) in enumerate(train_dataloader):
            if i % 100 == 0:
                print(i, x, y)
            # print(x["flux"].shape, y.shape)
            # for j in range(len(x)):
                # simulated = "sim" if x["tic_inj"][j] != -1 else "real"
                # print(simulated)
                # plot_lc(x["flux"][j], save_path=f"/mnt/zfsusers/shreshth/pht_project/data/examples/test_dataloader_{j}_{simulated}.png")
                # if j == 10:
                    # break
            # break
            t.update()
    