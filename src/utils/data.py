"""data.py
Utility functions and classes for data manipulation.
"""

import os
import argparse
from ast import literal_eval
import time
from glob import glob
import functools
from copy import deepcopy

from tqdm.autonotebook import trange

import torch
import torchvision

import numpy as np
import pandas as pd

from astropy.table import Table

if __name__ == "__main__":
    import transforms
    from utils import plot_lc
else:
    from utils import transforms
    from utils.utils import plot_lc

# sector 11 looks dodgy, sector 16 empty

TRAIN_SECTORS_DEBUG = [10]
TRAIN_SECTORS_STANDARD = [10,11,12,13,14,15,16,17,18,19,20]
TRAIN_SECTORS_FULL = [10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]

VAL_SECTORS_DEBUG = [12]
VAL_SECTORS_STANDARD = [21,22,23]
VAL_SECTORS_FULL = [30,31,32,33,34,35]

TEST_SECTORS_DEBUG = [14]
TEST_SECTORS_STANDARD = [24]
TEST_SECTORS_FULL = [36,37,38]

SHORTEST_LC = 17500 # from sector 10-38. Used to trim all the data to the same length.
# SHORTEST_LC = 18900 # binned 7 sector 10-14

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
        min_snr=0.5,
        single_transit_only=True,
        transform=None,
        preprocessing=None,
        store_cache=True,
        plot_examples=False,
        ):
        """
        Params:
        - data_root_path (str): path to data directory 
        - data_split (str): which data split to load
        - bin_factor (int): binning factor light curves to use
        - synthetic_prob (float): proportion of data to be synthetic transits
        - eb_prob (float): proportion of data to be synthetic eclipsing binaries
        - min_snr (float): minimum signal-to-noise ratio to include transits
        - single_transit_only (bool): only use single transits in synthetic data
        - transform (callable): transform to apply to the data in getitem
        - preprocessing (callable): preprocessing to apply to the data (before caching)
        - store_cache (bool): whether to store all the data in RAM in advance
        - plot_examples (bool): whether to plot the light curves for debugging
        """
        super(LCData, self).__init__()

        self.data_root_path = data_root_path
        self.data_split = data_split
        self.bin_factor = bin_factor
        self.synthetic_prob = synthetic_prob
        self.eb_prob = eb_prob
        self.min_snr = min_snr
        self.single_transit_only = single_transit_only
        self.transform = transform
        self.store_cache = store_cache
        self.preprocessing = preprocessing
        self.plot_examples = plot_examples
        
        self.sectors = self._get_sectors()

        ####### LC data

        # get list of all lc files
        self.lc_file_list = []
        for sector in self.sectors:
            # print(f"sector: {sector}")
            new_files = glob(f"{self.data_root_path}/lc_csvs_cdpp/Sector{sector}/*binfac-{self.bin_factor}.csv", recursive=True)
            print("num. files found: ", len(new_files))
            self.lc_file_list += new_files
        print("total num. LC files found: ", len(self.lc_file_list))

        ####### Label data

        # get all the labels
        self.labels_df = pd.DataFrame()
        for sector in self.sectors:
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

        ##### eclipsing binaries
        if self.eb_prob > 0.0:
            self.eb_data = self._get_eb_data()
            print(f"using {self.eb_prob} proportion of synthetic eclipsing binaries")

        ##### cache data
        self.cache = {}
        if self.store_cache:
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
        if idx in self.cache:
            (x, y) = self.cache[idx]
        else:
            # get lc file
            lc_file = self.lc_file_list[idx]
            x = _read_lc_csv(lc_file)
            # if corrupt return None and skip c.f. collate_fn
            if x["flux"] is None:
                if self.store_cache:
                    self.cache[idx] = (x, None)
                return x, None

            if self.plot_examples:
                plot_lc(x["flux"], save_path=f"/mnt/zfsusers/shreshth/pht_project/data/examples/test_dataloader_raw_{idx}.png")

            # preprocessing
            if self.preprocessing:
                x["flux"] = self.preprocessing(x["flux"])
                if self.plot_examples:
                    plot_lc(x["flux"], save_path=f"/mnt/zfsusers/shreshth/pht_project/data/examples/test_dataloader_preprocessed_{idx}.png")

            # get label for this lc file (if exists), match sector 
            y = self.labels_df.loc[(self.labels_df["TIC_ID"] == x["tic"]) & (self.labels_df["sector"] == x["sec"]), "maxdb"].values
            if len(y) == 1:
                y = torch.tensor(y[0], dtype=torch.float)
            elif len(y) > 1:
                y = None
            else:
                y = None

            if self.store_cache:
                # add to cache 
                self.cache[idx] = (deepcopy(x), deepcopy(y))

        # probabilistically add synthetic transits, only if labels are zero.
        rand_num = np.random.rand()
        if (rand_num < self.synthetic_prob) and (y == 0.0):
            x = self._add_synthetic_transit(x)
            y = torch.tensor(1.0, dtype=torch.float)
        # add a small delta in case eb_prob is 0.0
        elif (self.synthetic_prob < rand_num < self.synthetic_prob + self.eb_prob + 1e-8) and (y == 0.0) and (self.eb_prob > 0.0):
            x = self._add_synthetic_eclipse_binary(x)
            y = torch.tensor(0.0, dtype=torch.float)
        else:
            x["tic_inj"] = -1
            x["depth"] = -1
            x["duration"] = -1
            x["period"] = -1
            x["snr"] = -1
            x["eb_prim_depth"] = -1
            x["eb_sec_depth"] = -1
            x["eb_period"] = -1
            x["class"] = "real"

        # if transit additions failed, return None
        if x["flux"] is None:
            return x, None

        if self.plot_examples:
            plot_lc(x["flux"], save_path=f"/mnt/zfsusers/shreshth/pht_project/data/examples/test_dataloader_injected_{idx}.png")

        if self.transform:
            x["flux"] = self.transform(x["flux"])
            if self.plot_examples:
                plot_lc(x["flux"], save_path=f"/mnt/zfsusers/shreshth/pht_project/data/examples/test_dataloader_transformed_{idx}.png")

        return x, y


    def _get_sectors(self):
        """
        Returns:
        - sectors (list): list of sectors
        """
        if self.data_split == "train_standard":
            return TRAIN_SECTORS_STANDARD
        elif self.data_split == "val_standard":
            return VAL_SECTORS_STANDARD
        elif self.data_split == "test_standard":
            return TEST_SECTORS_STANDARD
        if self.data_split == "train_full":
            return TRAIN_SECTORS_FULL
        elif self.data_split == "val_full":
            return VAL_SECTORS_FULL
        elif self.data_split == "test_full":
            return TEST_SECTORS_FULL
        elif self.data_split == "train_debug":
            return TRAIN_SECTORS_DEBUG
        elif self.data_split == "val_debug":
            return VAL_SECTORS_DEBUG
        elif self.data_split == "test_debug":
            return TEST_SECTORS_DEBUG
        else:
            raise ValueError(f"Invalid data split {self.data_split}")

    def _get_eb_data(self):
        """Loads the eclipsing binary data
        """
        # simulated transit info
        eb_table = Table.read(f"{self.data_root_path}/eb_csvs/ete6_eb_data.txt", format='ascii',comment='#')
        eb_files = glob(f"{self.data_root_path}/eb_csvs/EBs_*binfac-{self.bin_factor}.csv")
        print(f"found {len(eb_files)} eb flux files for binfac {self.bin_factor}")
        
        # load planetary transits into RAM
        eb_data = {}   # dict of dicts with metadata
        print("loading eb metadata...")
        idx = 0
        with trange(len(eb_files)) as t:
            for eb_file in eb_files:
                # extract tic id
                tic_id = int(eb_file.split("/")[-1].split("_")[1].split(".")[0])
                # check if we should include this planet
                if not self._is_planet_in_data_split(tic_id):
                    continue

                # look up in table
                eb_row = eb_table[eb_table['col1'] == tic_id]
                eb_prim_depth = eb_row['col8'][0]
                eb_sec_depth = eb_row['col9'][0]
                eb_per = eb_row['col3'][0]     # transit period                  
                eb_flux = np.genfromtxt(str(eb_file), skip_header=1)

                if len(eb_flux) == 0:
                    print(f"WARNING: no data for tic {tic_id}", eb_row)
                    print(f"skipping...")
                    continue
                
                eb_data[idx] = {"flux": eb_flux, "tic_id": tic_id, "eb_prim_depth": eb_prim_depth, "eb_sec_depth": eb_sec_depth, "eb_period": eb_per}
                idx += 1
                t.update()

        print(f"Loaded {len(eb_data)} simulated transits for {self.data_split} data split")

        return eb_data


    def _add_synthetic_eclipse_binary(self, x):
        """Adds synthetic eclipsing binary data.
        """
        eb_inj = self.eb_data[np.random.randint(len(self.eb_data))]    
        x["flux"] = self._inject_transit(x["flux"], eb_inj["flux"], single_transit=False)
        x["tic_inj"] = eb_inj["tic_id"]
        x["eb_prim_depth"] = eb_inj["eb_prim_depth"]
        x["eb_sec_depth"] = eb_inj["eb_sec_depth"]
        x["eb_period"] = eb_inj["eb_period"]
        x["depth"] = -1
        x["duration"] = -1
        x["period"] = -1
        x["snr"] = -1
        x["class"] = "eb"

        return x


    def _get_pl_data(self):
        """Loads the planetary transits data.
        """
        # simulated transit info
        pl_table = Table.read(f"{self.data_root_path}/planet_csvs/ete6_planet_data.txt", format='ascii',comment='#')
        pl_files = glob(f"{self.data_root_path}/planet_csvs/Planets_*binfac-{self.bin_factor}.csv")
        print(f"found {len(pl_files)} planet flux files for binfac {self.bin_factor}")
        
        # load planetary transits into RAM
        pl_data = {}   # dict of dicts with metadata
        print("loading planet metadata...")
        idx = 0
        with trange(len(pl_files)) as t:
            for pl_file in pl_files:
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
                
                # check transit duration as well (from simulation)
                if pl_dur > 4: 
                    print(f"duration {pl_dur} too long for tic {tic_id}")
                    continue
                
                # if pl_depth < 1000:
                #     print(f"depth {pl_depth} too low for tic {tic_id}")
                #     continue

                
                pl_data[idx] = {"flux": pl_flux, "tic_id": tic_id, "depth": pl_depth, "duration": pl_dur, "period": pl_per}
                idx += 1
                t.update()

        print(f"Loaded {len(pl_data)} simulated transits for {self.data_split} data split")

        return pl_data


    def _is_planet_in_data_split(self, tic_id):
        """Checks if a planet flux should be included in this data for simulation.
        Currently just uses the tic_id to select 1/4th of the available data for training/val.
        """
        if "train" in self.data_split:
            if tic_id % 4 == 0:
                return True
            else:
                return False
        elif "val" in self.data_split:
            if tic_id % 4 == 1:
                return True
            else:
                return False
        elif "test" in self.data_split:
            if (tic_id % 4 == 2) or (tic_id % 4 == 3):
                return True
            else:
                return False


    def _add_synthetic_transit(self, x):
        """Adds a synthetic transit to the data.
        """
        bad_snr = True
        num_bad = 0
        while bad_snr:
            pl_inj = self.pl_data[np.random.randint(len(self.pl_data))]
            # check closest cdpp of base flux to planet duration
            durs = np.array([0.5, 1, 2])
            durs_ = ["cdpp05", "cdpp1", "cdpp2"]
            j = np.argmin(abs(pl_inj["duration"] - durs))
            # check if we have cdpp data for this star
            if durs_[j] in x:
                x_cdpp = float(x[durs_[j]])     
                pl_snr = pl_inj["depth"] / x_cdpp
            else:
                # if not, just inject anyway (backwards compatibility)
                print(f"WARNING: no {durs_[j]} data for tic {x['tic']}")
                pl_snr = 100.0

            # if the SNR is lower than our threshhold, skip this target entirely. 
            # min_snr = 0.5 in the argparse - ask Nora.
            # max_snr = 15.0 in the argparse - ask Nora.
            if (pl_snr < self.min_snr) or (pl_snr > 25):   
            # if pl_snr < self.min_snr:
                bad_snr = True
            else:
                bad_snr = False
            if bad_snr:
                num_bad += 1
                # print("bad SNR: ", pl_snr, " for TIC: ", x["tic"], " in sector: ", x["sec"], "and planet tic id:", pl_inj["tic_id"])
                if num_bad > 5:
                    # print("too many bad SNRs. Skipping this target.")
                    x["flux"] = None
                    return x
        
        x["flux"] = self._inject_transit(x["flux"], pl_inj["flux"], single_transit=self.single_transit_only)
        x["tic_inj"] = pl_inj["tic_id"]
        x["depth"] = pl_inj["depth"]
        x["duration"] = pl_inj["duration"]
        x["period"] = pl_inj["period"]
        x["snr"] = pl_snr
        x["eb_prim_depth"] = -1
        x["eb_sec_depth"] = -1
        x["eb_period"] = -1
        x["class"] = "synth_planet"
    
        return x

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


    def _inject_transit(self, base_flux, injected_flux, single_transit=False):
        """Inject a transit into a base light curve. keep nans in the same place.
        N.B. Need to ensure both fluxes correspond to the same cadence.
        Params:
        - base_flux (np.array): base LC to inject into
        - injected_flux (np.array): transit to inject (different length to base)
        """
        if len(injected_flux) >= len(base_flux):
            injected_flux = injected_flux[:len(base_flux)-1]
        
        # ensure the injected flux is not in a missing data region. Only if single transit as the full curve may have a lot of missing data
        if single_transit:
            missing_data = True
            while missing_data:
                # add injected flux section to random part of base flux
                start_idx = np.random.randint(0, len(base_flux)-len(injected_flux))
                # if there is 20% missing data in the transit, try again
                # TODO maybe adjust this parameter?      
                missing_data = np.count_nonzero(np.isnan(base_flux[start_idx:start_idx+len(injected_flux)])) / len(injected_flux) > 0.2
        else:
            start_idx = np.random.randint(0, len(base_flux)-len(injected_flux))

        base_flux[start_idx:start_idx+len(injected_flux)] = base_flux[start_idx:start_idx+len(injected_flux)] * injected_flux

        return base_flux

##### UTILS

def collate_fn(batch):
    """Collate function for filtering out corrupted data in the dataset
    Assumes that missing data are NoneType
    """
    batch = [(x,y) for (x,y) in batch if x["flux"] is not None]   # filter on missing flux 
    batch = [(x,y) for (x,y) in batch if y is not None]           # filter on missing labels
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
        - cdpp(05,1,2) (float): CDPP at 0.5, 1, 2 hour time scales
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
        x = {"flux": None}
    return x

        
def get_data_loaders(args):
    """Get data loaders given argparse arguments
    """
    # unpack arguments
    data_root_path = args.data_path
    bin_factor = args.bin_factor
    synthetic_prob = args.synthetic_prob
    eb_prob = args.eb_prob
    batch_size = args.batch_size
    num_workers = args.num_workers
    cache = not args.no_cache
    aug_prob = args.aug_prob
    permute_fraction = args.permute_fraction
    delete_fraction = args.delete_fraction
    outlier_std = args.outlier_std
    rolling_window = args.rolling_window
    noise_std = args.noise_std
    min_snr = args.min_snr
    # max_lc_length = args.max_lc_length
    max_lc_length = int(SHORTEST_LC / bin_factor)
    multi_transit = args.multi_transit
    pin_memory = True
    data_split = args.data_split
    plot_examples = args.plot_examples

    # preprocessing = torchvision.transforms.Compose([
    #     # transforms.RemoveOutliersPercent(percent_change=0.15),
    #     transforms.RemoveOutliers(window=rolling_window, std_dev=outlier_std),
    # ])
    preprocessing = None

    # composed transform
    training_transform = torchvision.transforms.Compose([
        transforms.NormaliseFlux(),
        transforms.MirrorFlip(prob=aug_prob),
        transforms.RandomDelete(prob=aug_prob, delete_fraction=delete_fraction),
        transforms.RandomShift(prob=1.0, permute_fraction=permute_fraction),    # always permute to remove sector bias
        transforms.GaussianNoise(prob=aug_prob, window=rolling_window, std=noise_std),
        transforms.ImputeNans(method="zero"),
        transforms.Cutoff(length=max_lc_length),
        transforms.ToFloatTensor()
    ])

    # test tranforms - do not randomly delete or permute
    val_transform = torchvision.transforms.Compose([
        transforms.NormaliseFlux(),
        transforms.ImputeNans(method="zero"),
        transforms.Cutoff(length=max_lc_length),
        transforms.ToFloatTensor()
    ])

    test_transform = torchvision.transforms.Compose([
        transforms.NormaliseFlux(),
        transforms.ImputeNans(method="zero"),
        transforms.Cutoff(length=max_lc_length),
        transforms.ToFloatTensor()
    ])


    # TODO choose type of data set - set an argument for this (e.g. simulated/real proportions)
    train_set = LCData(
        data_root_path=data_root_path,
        data_split=f"train_{data_split}",
        bin_factor=bin_factor,
        synthetic_prob=synthetic_prob,
        eb_prob=eb_prob,
        min_snr=min_snr,
        single_transit_only=not multi_transit,
        transform=training_transform,
        preprocessing=preprocessing,
        store_cache=cache,
        plot_examples=plot_examples
    )

    # same amount of synthetics in val set as in train set
    val_set = LCData(
        data_root_path=data_root_path,
        data_split=f"val_{data_split}",
        bin_factor=bin_factor,
        synthetic_prob=synthetic_prob,
        eb_prob=eb_prob,
        min_snr=min_snr,
        single_transit_only=not multi_transit,
        transform=val_transform,
        preprocessing=preprocessing,
        store_cache=cache,
        plot_examples=plot_examples
    )

    # no synthetics in test set
    test_set = LCData(
        data_root_path=data_root_path,
        data_split=f"test_{data_split}",
        bin_factor=bin_factor,
        synthetic_prob=0.0,
        eb_prob=0.0,
        min_snr=min_snr,
        single_transit_only=not multi_transit,       # irrelevant for test set
        transform=test_transform,
        preprocessing=preprocessing,
        store_cache=cache,
        plot_examples=plot_examples
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
                                                shuffle=True,               # shuffle val set as well to get different batches for prediction saving
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
    ap.add_argument("--data-split", type=str, default="debug")
    ap.add_argument("--bin-factor", type=int, default=7)
    ap.add_argument("--synthetic-prob", type=float, default=1.0)
    ap.add_argument("--eb-prob", type=float, default=0.0)
    ap.add_argument("--aug-prob", type=float, default=1.0, help="Probability of augmenting data with random defects.")
    ap.add_argument("--permute-fraction", type=float, default=0.25, help="Fraction of light curve to be randomly permuted.")
    ap.add_argument("--delete-fraction", type=float, default=0.1, help="Fraction of light curve to be randomly deleted.")
    ap.add_argument("--outlier-std", type=float, default=3.0, help="Remove points more than this number of rolling standard deviations from the rolling mean.")
    ap.add_argument("--rolling-window", type=int, default=100, help="Window size for rolling mean and standard deviation.")
    ap.add_argument("--min-snr", type=float, default=1.0, help="Min signal to noise ratio for planet injection.")
    ap.add_argument("--noise-std", type=float, default=0.1, help="Standard deviation of noise added to light curve for training.")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--multi-transit", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--plot-examples", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_dataloader, val_dataloader, test_dataloader = get_data_loaders(args)
    with trange(len(train_dataloader)) as t:
        for i, (x, y) in enumerate(train_dataloader):
            if i == 0:
                print(i, x, y)
                # print(x["flux"].shape, y.shape)
                for j in range(len(x)):
                    if x["snr"][j] != -1:
                        simulated = "planet"
                    elif x["eb_period"][j] != -1:
                        simulated = "eb" 
                    else:
                        simulated = "real"
                    print(simulated)
                    # plot_lc(x["flux"][j], save_path=f"/mnt/zfsusers/shreshth/pht_project/data/examples/test_dataloader_{j}_{simulated}.png")
                    if j == 5:
                        break
                break
            t.update()
    