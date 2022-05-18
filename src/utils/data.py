"""data.py
Utility functions and classes for data manipulation.
"""

import os
from ast import literal_eval
import csv
import time
from glob import glob
import functools

from tqdm.autonotebook import trange

import torch
from torchvision import transforms

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import AutoMinorLocator

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split as split

import astropy.io.fits as pf
from astropy.table import Table


# SECTORS = list(range(10, 39))
# without 35 and 37
# SECTORS = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 38]
# SECTORS = [10, 11]
# SECTORS = [37]
# SECTORS = [10, 11, 12, 13]

TRAIN_SECTORS = [10,11,12,13]
TEST_SECTORS = [14]

SHORTEST_LC = 17546 # from sector 10-38. Used to trim all the data to the same length.


#### TRANSFORMS

class NormaliseFlux(object):
    """Normalise the flux
    TODO check what normalisation is best, add optional args
    """
    def __init__(self):
        pass

    def __call__(self, x):
        x /= np.nanmedian(x)
        # median at 0
        x -= np.nanmedian(x)
        x = x.astype(np.float64)  # to fix numpy => torch byte error
        return x


class Cutoff(object):
    """Restrict LC data to a certain length. Cut start or end randomly
    """
    def __init__(self, length=SHORTEST_LC):
        self.length = length
    
    def __call__(self, x):
        # choose a random start to cut
        start = np.random.randint(0, len(x) - self.length)
        return x[start:start+self.length]


class ImputeNans(object):
    """Impute nans in the light curve
    """
    def __init__(self, method="zero"):
        self.method = method

    def __call__(self, x):
        if self.method == "zero":
            return np.nan_to_num(x)
        else:
            raise NotImplementedError("Only zero imputation is implemented")


class BinData(object):
    """Bin light curves. (Not used now we pre-bin the data)
    """

    def __init__(self, bin_factor=1):
        self.bin_factor = bin_factor
    
    def __call__(self, x):
        ## bin data
        N = len(x)
        n = int(np.floor(N / self.bin_factor) * self.bin_factor)
        X = np.zeros((1, n))
        X[0, :] = x[:n]
        Xb = self._rebin(X, (1, int(n / self.bin_factor)))
        x_binned = Xb[0]
        return x_binned

    def _rebin(self, arr, new_shape):
    
        shape = (
            new_shape[0],
            arr.shape[0] // new_shape[0],
            new_shape[1],
            arr.shape[1] // new_shape[1],
        )
        return arr.reshape(shape).mean(-1).mean(1)


class RandomDelete(object):
    """Randomly delete a continuous section of the data
    """
    def __init__(self, prob=0.0, delete_fraction=0.1):
        self.prob = prob
        self.delete_fraction = delete_fraction

    def __call__(self, x):

        if np.random.rand() < self.prob:
            N = len(x)
            n = int(np.floor(N * self.delete_fraction))
            start = np.random.randint(0, N - n)
            x[np.arange(start, start + n)] = 0.0

        return x


class RandomShift(object):
    """Randomly swap a chunk of the data with another chunk
    """
    def __init__(self, prob, permute_fraction=0.1):
        self.prob = prob
        self.permute_fraction = permute_fraction

    def __call__(self, x):
            
        if np.random.rand() < self.prob:
            N = len(x)
            n = int(np.floor(N * self.permute_fraction))
            start1 = np.random.randint(0, N - n)
            end1 = start1 + n
            start2 = np.random.randint(0, N - n)
            end2 = start2 + n
            x[start1:end1], x[start2:end2] = x[start2:end2], x[start1:end1]

        return x


class MirrorFlip(object):
    """Mirror flip the data
    """
    def __init__(self, prob):
        self.prob = prob
    
    def __call__(self, x):
        if np.random.rand() < self.prob:
            x = np.flip(x, axis=0)
        return x


# composed transform
training_transform = transforms.Compose([
    NormaliseFlux(),
    RandomDelete(prob=0.0, delete_fraction=0.1),
    RandomShift(prob=0.0, permute_fraction=0.1),
    # BinData(bin_factor=3),  # bin before imputing
    ImputeNans(method="zero"),
    Cutoff(length=int(SHORTEST_LC/7))
])

# test tranforms - do not randomly delete or permute
test_transform = transforms.Compose([
    NormaliseFlux(),
    RandomDelete(prob=0.0, delete_fraction=0.1),
    RandomShift(prob=0.0, permute_fraction=0.1),
    # BinData(bin_factor=3),  # bin before imputing
    ImputeNans(method="zero"),
    Cutoff(length=int(SHORTEST_LC/7))
])

#### DATASET CLASSES

class LCData(torch.utils.data.Dataset):
    """Light curve dataset
    """

    def __init__(self, data_root_path, sectors, synthetic_prob=0.0, eb_prob=0.0, single_transit_only=True, transform=training_transform):
        """
        Params:
        - data_root_path (str): path to data directory 
        - sectors (list[int]): list of sectors to load
        - synthetic_prob (float): proportion of data to be synthetic transits
        - eb_prob (float): proportion of data to be synthetic eclipsing binaries
        - single_transit_only (bool): only use single transits in synthetic data
        - transform (callable): transform to apply to the data
        """
        super(LCData, self).__init__()

        self.data_root_path = data_root_path
        self.sectors = sectors
        self.synthetic_prob = synthetic_prob
        self.eb_prob = eb_prob
        self.single_transit_only = single_transit_only
        self.transform = transform

        self.cache = {} # cache for __getitem__

        ##### planetary transits 
        # simulated transit info
        pl_table = Table.read(f"{data_root_path}/planet_csvs/ete6_planet_data.txt", format='ascii',comment='#')
        pl_files = glob(f"{data_root_path}/planet_csvs/Planets_*.txt")
        print(f"found {len(pl_files)} planet flux files")
        
        # load planetary transits into RAM
        print("adding metadata...")
        self.pl_data = []   # list of dicts with metadata
        # TODO better to use a dict to speed up lookup or df to save memory?

        with tqdm(total=len(pl_files)) as t:
            for i, pl_file in enumerate(pl_files):
                # extract tic id
                tic_id = int(pl_file.split("/")[-1].split("_")[1].split(".")[0])
                print(f"tic_id: {tic_id}")
                # look up in table
                pl_row = pl_table[pl_table['col1'] == tic_id]
                print(pl_row)

                pl_depth = pl_row['col10'][0]  # transit depth
                pl_dur = pl_row['col9'][0]     # transit duration
                pl_per = pl_row['col3'][0]     # transit period
                print(f"depth: {pl_depth}, duration: {pl_dur}, period: {pl_per}")
                                
                pl_flux = np.genfromtxt(str(pl_file) skip_header=1)
                print(pl_flux.shape)

                if self.single_transit_only:
                    # take only the transit
                    pl_flux = _extract_transit(pl_flux)
               
                self.pl_data.append({"flux": pl_flux, "tic_id": tic_id, "depth": pl_depth, "duration": pl_dur, "period": pl_per})
                t.update()
                if i > 10:
                    break
    
        print(f"Loaded {len(pl_data)} simulated transits")

        # TODO use the planet metadata to augment the loss function


        ####### LC data

        # get list of all lc files
        self.lc_file_list = []
        for sector in sectors:
            # print(f"sector: {sector}")
            new_files = glob(f"{self.data_root_path}/Sector{sector}/*.csv"), recursive=True)
            print("num. files found: ", len(new_files))
            self.lc_file_list += new_files

        print("total num. files found: ", len(self.lc_file_list))

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


    def __len__(self):
        return len(self.lc_file_list)


    # maxsize is the max number of samples to cache - each LC is ~2MB. If you have a lot of memory, you can increase this
    # @functools.lru_cache(maxsize=1000)  # Cache loaded data
    def __getitem__(self, idx):
        """Returns:
        - input (dict): dictionary with keys:
            - flux (float): light curve
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
        # get lc file
        lc_file = self.lc_file_list[idx]

        # read lc file
        x, tic, sec = _read_lc_csv(lc_file)
        # if corrupt return None and skip c.f. collate_fn
        if x is None:
            return {"flux": None}, None


        # get label for this lc file (if exists) match sector 
        y = self.labels_df.loc[(self.labels_df["TIC_ID"] == tic) & (self.labels_df["sector"] == sec), "maxdb"].values
        if len(y) == 1:
            y = torch.tensor(y[0], dtype=torch.float)
        elif len(y) > 1:
            # print(y, "more than one label for TIC: ", tic, " in sector: ", sec)
            y = None
            # self.no_label_tics.append((tic, sec))
        else:
            # print(y, "label not found for TIC: ", tic, " in sector: ", sec)
            y = None
            # self.no_label_tics.append((tic, sec))

        # TODO probabilistically add synthetic transits, only if labels are zero.
        if np.random.rand() < self.synthetic_transit_prob:
            pl_inj = self.pl_data[np.random.randint(len(self.pl_data))]
            x = _inject_transit(x, pl_inj["flux"])

        if self.transform:
            x = self.transform(x)
        # print(x.shape)

        # add to cache 

        # TODO change this
        # return (torch.tensor(x, dtype=torch.float), torch.tensor(tic), torch.tensor(sec), False), y
        return {"flux": x, "tic_id": tic, "sector": sec}, y


# TODO function to get file from tic id - needed for specific lookup 


##### UTILS

# TODO collate fn to return a good batch of simulated and real data (do this from the data loader
def collate_fn(batch):
    """Collate function for filtering out corrupted data in the dataset
    Assumes that missing data are NoneType
    """
    batch = [x for x in batch if x[0]["flux"] is not None]   # filter on missing fits files
    batch = [x for x in batch if x[1] is not None]      # filter on missing labels
    return torch.utils.data.dataloader.default_collate(batch)


def _extract_transit(x):
    """Extract a single transit from the planet flux
    Params:
    - x (np.array): flux of the light curve
    Returns:
    - transit (np.array): extracted single transit (shape variable)
    """
    # get the first dip
    start_idx = np.argmax(x<1)
    # get the end of the dip
    length = np.argmax(x[start_idx:]==1)
    # take one extra from either side
    transit = x[start_idx-1:start_idx+length+1]

    return transit


def _inject_transit(base_flux, injected_flux):
    """Inject a transit into a base light curve. 
    N.B. Need to ensure both fluxes correspond to the same cadence.
    Params:
    - base_flux (np.array): base LC to inject into
    - injected_flux (np.array): transit to inject (different length to base)
    """
    print("injecting transit")
    if len(injected_flux) > len(base_flux):
        injected_flux = injected_flux[:len(base_flux)]
    
    # ensure the injected flux is not in a missing data region
    missing_data = True
    while missing_data:
        # add injected flux section to random part of base flux
        start_idx = np.random.randint(0, len(base_flux)-len(injected_flux))
        # check if there is missing data in the injected flux
        print("checking for missing data")
        print(base_flux[start_idx:start_idx+len(injected_flux)])
        if base_flux[start_idx] != 0.0:
            missing_data = False

    print("start idx: ", start_idx, "length", len(base_flux), len(injected_flux))
    base_flux[start_idx:start_idx+len(injected_flux)] *= injected_flux

    return base_flux


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

    # read the csv file
    df = pd.read_csv(lc_file, delimiter=',', skip_header=1)
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


def plot_lc(x, save_path="/mnt/zfsusers/shreshth/pht_project/data/examples/test_light_curve.png"):
    """Plot light curve for debugging
    Params:
    - x (np.array): light curve
    """

    # plot it
    fig, ax = plt.subplots(figsize=(16, 5))
    plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.05)

    ## plot the binned and unbinned LC
    ax.plot(list(range(len(x))), x,
        color="royalblue",
        marker="o",
        markersize=1,
        lw=0,
        label="unbinned",
    )
    ## label the axis.
    ax.xaxis.set_label_coords(0.063, 0.06)  # position of the x-axis label

    ## define tick marks/axis parameters
    minorLocator = AutoMinorLocator()
    ax.xaxis.set_minor_locator(minorLocator)
    ax.tick_params(direction="in", which="minor", colors="w", length=3, labelsize=13)

    minorLocator = AutoMinorLocator()
    ax.yaxis.set_minor_locator(minorLocator)
    ax.tick_params(direction="in", length=3, which="minor", colors="grey", labelsize=13)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    ax.tick_params(axis="y", direction="in", pad=-30, color="white", labelcolor="white")
    ax.tick_params(axis="x", direction="in", pad=-17, color="white", labelcolor="white")

    # ax.set_xlabel("Time (days)", fontsize=10, color="white")

    ax.set_facecolor("#03012d")

    ## save the image
    plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor())

        
def get_data_loaders(args):
    """Get data loaders given argparse arguments
    """
    # unpack arguments
    data_root_path = args.data_path
    val_size = args.val_size
    test_size = args.test_size
    seed = args.seed
    batch_size = args.batch_size
    num_workers = args.num_workers
    pin_memory = False

    # TODO choose type of data set - set an argument for this (e.g. simulated/real proportions)


    real_dataset = LCData(data_root_path, labels_root_path)
    sim_dataset = SimulatedData(data_root_path, labels_root_path)
    dataset = torch.utils.data.ConcatDataset([real_dataset, sim_dataset])
    # dataset = sim_dataset
    print(f"dataset size {len(dataset)}: {len(real_dataset)} real, {len(sim_dataset)} simulated")


    indices = [i for i in range(len(dataset))]
    train_idx, test_idx = split(indices, random_state=seed, test_size=test_size)
    train_and_val_set = torch.utils.data.Subset(dataset, train_idx)
    test_set = torch.utils.data.Subset(dataset, test_idx)


    indices = [i for i in range(len(train_and_val_set))]
    train_idx, val_idx = split(indices, random_state=seed, test_size=val_size/(1-test_size))
    train_set = torch.utils.data.Subset(train_and_val_set, train_idx)
    val_set = torch.utils.data.Subset(train_and_val_set, val_idx)

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

    # TODO fix the transforms of the non-training dataloaders
    # val_dataloader.dataset.transform = None

    # for missing ones
    # full_dataloader = torch.utils.data.DataLoader(dataset,
    #                                             batch_size=batch_size,
    #                                             shuffle=False,
    #                                             num_workers=num_workers,
    #                                             pin_memory=pin_memory,
    #                                             collate_fn=collate_fn)

    return train_dataloader, val_dataloader, test_dataloader
    # return full_dataloader

if __name__ == "__main__":
    
    LC_ROOT_PATH = "/mnt/zfsusers/shreshth/pht_project/data/TESS"
    LABELS_ROOT_PATH = "/mnt/zfsusers/shreshth/pht_project/data/pht_labels"

    # lc_data = LCData(LC_ROOT_PATH, LABELS_ROOT_PATH)
    # sim_data = SimulatedData(LC_ROOT_PATH, LABELS_ROOT_PATH)
    # print(sim_data[0])
    # print(sim_data[10])

    # print(len(lc_data))
    train_dataloader, val_dataloader, test_dataloader = get_data_loaders(LC_ROOT_PATH, LABELS_ROOT_PATH, val_size=0.2, test_size=0.2, seed=0, batch_size=1024, num_workers=0, pin_memory=False)
    # # train_dataloader = get_data_loaders(LC_ROOT_PATH, LABELS_ROOT_PATH, val_size=0.2, test_size=0.2, seed=0, batch_size=1024, num_workers=0, pin_memory=False)
    with trange(len(train_dataloader)) as t:
        for i, (x, y) in enumerate(train_dataloader):
            # print(i, x, y)
            print(i, x, y)
            print(x[0].shape, y.shape)
            for j in range(len(x)):
                print(x[-1][j])
                simulated = "sim" if x[-1][j] else "real"
                print(simulated)
                plot_lc(x[0][j], save_path=f"/mnt/zfsusers/shreshth/pht_project/data/examples/test_dataloader_{j}_{simulated}.png")
                if j == 10:
                    break
            break
            t.update()
    
    # # save no label tics to file
    # print("no label tics: ", train_dataloader.dataset.no_label_tics)
    # with open('/mnt/zfsusers/shreshth/pht_project/data/pht_labels/no_label_tics.csv','w') as out:
    #     csv_out = csv.writer(out)
    #     csv_out.writerow(['tic_id','sec'])
    #     for row in train_dataloader.dataset.no_label_tics:
    #         csv_out.writerow(row)







# class SimulatedData(torch.utils.data.Dataset):
#     """Simulated LC dataset for training
#     """
#     def __init__(self, lc_root_path, labels_root_path, transform=training_transform):
#         super(SimulatedData, self).__init__()

#         self.lc_root_path = lc_root_path
#         self.labels_root_path = labels_root_path
#         # TODO do this differently for testing and training (e.g. shuffling stuff)
#         self.transform = transform 

#         # get the labels and planet info
#         self.labels_df = pd.DataFrame()
#         for sector in SECTORS:
#             self.labels_df = pd.concat([self.labels_df, pd.read_csv(f"{labels_root_path}/summary_file_sec{sector}.csv")], axis=0)
#         self.simulated_transits_df = self.labels_df[self.labels_df["subject_type"]]
#         print("num. simulated transits labels: ", len(self.simulated_transits_df))

#         # simulated transit info
#         self.pl_table = Table.read(f"{lc_root_path}/ETE-6/injected/ete6_planet_data.txt", format='ascii',comment='#')


#     def __len__(self):
#         return len(self.simulated_transits_df)

#     # maxsize is the max number of samples to cache - each LC is ~2MB. If you have a lot of memory, you can increase this
#     @functools.lru_cache(maxsize=1000)  # Cache loaded data
#     def __getitem__(self, idx):
#         """TODO also return the snr to make a differential loss function
#         Returns:
#         - input (tuple)
#             - x (float): light curve
#             - tic (int): TIC
#             - sec (int): sector
#             - sim (bool): True if simulated data
#         - y (float): score
#         """

#         this_simulated_transit = self.simulated_transits_df.iloc[idx]
#         # print(this_simulated_transit)
#         y = this_simulated_transit["maxdb"]
#         y = torch.tensor(y, dtype=torch.float)

#         tic_base = int(this_simulated_transit["TIC_LC"])
#         tic_inj = int(this_simulated_transit["TIC_inj"])
#         snr = this_simulated_transit["SNR"]
#         base_tic_sector = this_simulated_transit["sector"]
#         # print(y, tic_base, tic_inj, snr, base_tic_sector)

#         # read base lc file
#         base_lc = glob("{}/planethunters/Rel*{:d}/Sector{}/**/*{:d}*.fit*".format(self.lc_root_path, base_tic_sector, base_tic_sector, tic_base), recursive=True)
#         # print(base_lc)
#         if len(base_lc) == 0:
#             print(f"no lc file found for TIC: {tic_base} in sector: {base_tic_sector}")
#             return (None, None, None, True), None
#         base_lc = base_lc[0]
#         x, tic, sec = _read_lc(base_lc)
#         # if corrupt return None and skip c.f. collate_fn
#         if x is None:
#             return (None, None, None, True), None

#         # debug plot
#         # plot_lc(x, save_path=f"./{tic}_{sec}_real.png")

#         # read injected lc file
#         injected_pl = glob("{}/ETE-6/injected/Planets/Planets_*{:d}.txt".format(self.lc_root_path, tic_inj))
#         # print(injected_pl)
#         if len(injected_pl) == 0:
#             print(f"no injected lc file found for TIC: {tic_inj}")
#             return (None, None, None, True), None
#         injected_pl = injected_pl[0]
#         inj_pl = np.genfromtxt(str(injected_pl))
#         # print(f"injected shape {inj_pl.shape}")

#         # plot_lc(inj_pl, save_path=f"./{tic_inj}_inj.png")

#         # inject planet
#         if len(inj_pl)>len(x):
#             inj_pl = inj_pl[:len(x)]
#         x = inj_pl * x

#         # plot_lc(x, save_path=f"./{tic}_{sec}_real_inj_{tic_inj}.png")

#         # transform
#         if self.transform:
#             x = self.transform(x)
#         # print(x.shape)
#         # plot_lc(x, save_path=f"./{tic}_{sec}_real_inj_{tic_inj}_transformed.png")

#         # get injected planet info 
#         # plt_tic = self.pl_table[(self.pl_table['col1'] == tic_inj)][0]  # transit tic
#         # pl_depth = plt_tic['col10']  # transit depth
#         # pl_dur = plt_tic['col9']     # transit duration
#         # pl_per = plt_tic['col3']     # transit period


#         # return tensors
#         return (torch.tensor(x, dtype=torch.float), torch.tensor(tic), torch.tensor(sec), True), y
