"""data.py
Utility functions and classes for data manipulation.
"""

import os
from ast import literal_eval
import csv
import time
import glob
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


# SECTORS = list(range(10, 39))
# without 35 and 37
# SECTORS = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 38]
# SECTORS = [10, 11]
SECTORS = [37]
SHORTEST_LC = 17546 # from sector 10-38. Used to trim all the data to the same length


#### TRANSFORMS

class NormaliseFlux(object):
    """Normalise the flux
    TODO check what normalisation is best, add optional args
    """
    def __init__(self):
        pass

    def __call__(self, x):
        x /= np.nanmedian(x)  # TODO why?
        # median at 0
        x -= np.nanmedian(x)
        x = x.astype(np.float64)  # to fix numpy => torch byte error
        return x


class Cutoff(object):
    """Restrict LC data to a certain length.
    """
    def __init__(self, length=SHORTEST_LC):
        self.length = length
    
    def __call__(self, x):
        return x[:self.length]


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


# class AddEbTransit(object):
#     """Add random eclipsing binary transits to the light curves
#     """
    # def __init__(self, period, duration, amplitude, phase):


# class AddPlTransit(object):
#     """Add random planetary transits to the light curves
#     """
    # def __init__(self, period, duration, amplitude, phase):


# composed transform
training_transform = transforms.Compose([
    NormaliseFlux(),
    ImputeNans(method="zero"),
    Cutoff()
])

# TODO test tranforms


#### DATASET CLASSES

class LCData(torch.utils.data.Dataset):
    """Light curve dataset
    """

    def __init__(self, lc_root_path, labels_root_path, transform=training_transform):
        super(LCData, self).__init__()

        self.lc_root_path = lc_root_path
        self.labels_root_path = labels_root_path
        # TODO do this differently for testing and training (e.g. shuffling stuff)
        self.transform = transform 

        # get list of all lc files
        self.lc_file_list = []
        for sector in SECTORS:
            # print(f"sector: {sector}")
            new_files = glob.glob(os.path.join(lc_root_path, f"planethunters/Rel{sector}/Sector{sector}/**/*.fit*"), recursive=True)
            print("num. files found: ", len(new_files))
            self.lc_file_list += new_files

            # length of light curve in this sector:
            x, tic, sec = _read_lc(self.lc_file_list[-1])
            print("length of light curve in sector ", sec, ": ", x.shape, ", TIC: ", tic)
        print("total num. files found: ", len(self.lc_file_list))

        # no labels for these tics
        with open(os.path.join(labels_root_path, "no_label_tics.txt"), "r") as f:
            tic_sec_list = literal_eval(f.read())
            self.no_label_tics = [tic for tic, sec in tic_sec_list]
        print(self.no_label_tics[:10], len(self.no_label_tics))
        self.lc_file_list = [x for x in self.lc_file_list if int(x.split("/")[-1].split("-")[2]) not in self.no_label_tics]
        print("removed tics with no labels, total num files now: ", len(self.lc_file_list))

        # get all the labels
        self.labels_df = pd.DataFrame()
        for sector in SECTORS:
            self.labels_df = pd.concat([self.labels_df, pd.read_csv(f"{labels_root_path}/summary_file_sec{sector}.csv")], axis=0)
        print("num. total labels (including simulated data): ", len(self.labels_df))

        # check how many non-zero labels
        print("num. non-zero labels: ", len(self.labels_df[self.labels_df["maxdb"] != 0.0]))
        print("strong non-zero labels (score > 0.5): ", len(self.labels_df[self.labels_df["maxdb"] > 0.5]))


    def __len__(self):
        return len(self.lc_file_list)


    # maxsize is the max number of samples to cache - each LC is ~2MB. If you have a lot of memory, you can increase this
    @functools.lru_cache(maxsize=1000)  # Cache loaded data
    def __getitem__(self, idx):
        # start = time.time()
        # get lc file
        lc_file = self.lc_file_list[idx]

        # read lc file
        x, tic, sec = _read_lc(lc_file)
        # if corrupt return None and skip c.f. collate_fn
        if x is None:
            return (x, tic, sec), None

        if self.transform:
            x = self.transform(x)

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

        # end = time.time()
        # print("time to get data", end - start)
        # return tensors
        return (torch.tensor(x, dtype=torch.float), torch.tensor(tic), torch.tensor(sec), False), y


# TODO function to get file from tic id - needed for specific lookup 


# class EBData(torch.utils.data.Dataset):
#     """Eclipsing Binary dataset for training to remove false positives
#     """

#     def __init__(self, lc_root_path, labels_root_path, transform=transform):
#         super(EBData, self).__init__()

#         pass

#     def __len__(self):
#         return 0
    
#     def __getitem__(self, idx):
#         return None, None


class SimulatedData(torch.utils.data.Dataset):
    """Simulated LC dataset for training
    """
    def __init__(self, lc_root_path, labels_root_path, transform=training_transform):
        super(LCData, self).__init__()

        self.lc_root_path = lc_root_path
        self.labels_root_path = labels_root_path
        # TODO do this differently for testing and training (e.g. shuffling stuff)
        self.transform = transform 

        # get all the simulated data files
        self.labels_df = pd.DataFrame()
        for sector in SECTORS:
            self.labels_df = pd.concat([self.labels_df, pd.read_csv(f"{labels_root_path}/summary_file_sec{sector}.csv")], axis=0)
        self.simulated_transits_df = self.labels_df[self.labels_df["subject_type"]]
        print("num. simulated transits labels: ", len(self.simulated_transits_df))
        print(self.simulated_transits_df.head())

        # check how many non-zero labels
        print("num. non-zero labels: ", len(self.labels_df[self.labels_df["maxdb"] != 0.0]))
        print("strong non-zero labels (score > 0.5): ", len(self.labels_df[self.labels_df["maxdb"] > 0.5]))


    def __len__(self):
        return len(self.lc_file_list)


    # maxsize is the max number of samples to cache - each LC is ~2MB. If you have a lot of memory, you can increase this
    @functools.lru_cache(maxsize=1000)  # Cache loaded data
    def __getitem__(self, idx):
        this_simulated_transit = self.simulated_transits_df.iloc[idx]
        print(this_simulated_transit)
        y = this_simulated_transit["maxdb"]
        tic_base = this_simulated_transit["TIC_LC"]
        tic_inj = this_simulated_transit["TIC_inf"]
        snr = this_simulated_transit["SNR"]
        print(y, tic_base, tic_inj, snr)

        # reconstruct the light curve files
        # find sector of the base TIC
        base_tic_sector = self.labels_df.loc[self.labels_df["TIC_ID"] == tic_base, "sector"].values[0]

        # read base lc file
        base_lc = glob("{}/planethunters/Rel{:d}/Sector{}/**/*{:d}.fit*".format(self.lc_root_path, base_tic_sector, base_tic_sector, tic_base), recursive=True)
        print(base_lc)
        base_lc = base_lc[0]
        x, tic, sec = _read_lc(base_lc)
        # if corrupt return None and skip c.f. collate_fn
        if x is None:
            return (x, tic, sec), None

        # read injected lc file
        injected_pl = glob("{}/ETE-6/injected/Planets/Planets_*{:d}.txt".format(self.lc_root_path, tic_inj))
        print(injected_pl)
        inj_pl = np.genfromtxt(str(injected_pl[0]))
        print(f"injected shape {inj_pl.shape}")

        # inject planet
        if len(inj_pl)>len(x):
            inj_pl = inj_pl[:len(x)]
        x = inj_pl * x

        # transform
        if self.transform:
            x = self.transform(x)

        # return tensors
        return (torch.tensor(x, dtype=torch.float), torch.tensor(tic), torch.tensor(sec), True), y


### UTILS

# TODO collate fn to return a good batch of simulated and real data (do this from the data loader
def collate_fn(batch):
    """Collate function for filtering out corrupted data in the dataset
    Assumes that missing data are NoneType
    """
    batch = [x for x in batch if x[0][0] is not None]   # filter on missing fits files
    batch = [x for x in batch if x[1] is not None]      # filter on missing labels

    return torch.utils.data.dataloader.default_collate(batch)


def _read_lc(lc_file):
    """Read light curve file
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


def get_data_loaders(data_root_path, labels_root_path, val_size, test_size, seed, batch_size, num_workers=0, pin_memory=False):
    """Get data loaders
    TODO pin_memory and num_workers sensibly
    """

    # TODO choose type of data set - set an argument for this (e.g. simulated/real proportions)

    dataset = LCData(data_root_path, labels_root_path)

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
    sim_data = SimulatedData(LC_ROOT_PATH, LABELS_ROOT_PATH)


    # print(len(lc_data))
    # train_dataloader, val_dataloader, test_dataloader = get_data_loaders(LC_ROOT_PATH, LABELS_ROOT_PATH, val_size=0.2, test_size=0.2, seed=0, batch_size=1024, num_workers=0, pin_memory=False)
    # # train_dataloader = get_data_loaders(LC_ROOT_PATH, LABELS_ROOT_PATH, val_size=0.2, test_size=0.2, seed=0, batch_size=1024, num_workers=0, pin_memory=False)
    # with trange(len(train_dataloader)) as t:
    #     for i, (x, y) in enumerate(train_dataloader):
    #         # print(i, x, y)
    #         if i % 100 == 0:
    #             print(i, x, y)
    #             print(x[0].shape, y.shape)

    #             # plot it
    #             fig, ax = plt.subplots(figsize=(16, 5))
    #             plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.05)

    #             ## plot the binned and unbinned LC
    #             ax.plot(list(range(len(x[0][0]))), x[0][0],
    #                 color="royalblue",
    #                 marker="o",
    #                 markersize=1,
    #                 lw=0,
    #                 label="unbinned",
    #             )
    #             ## label the axis.
    #             ax.xaxis.set_label_coords(0.063, 0.06)  # position of the x-axis label

    #             ## define tick marks/axis parameters

    #             minorLocator = AutoMinorLocator()
    #             ax.xaxis.set_minor_locator(minorLocator)
    #             ax.tick_params(direction="in", which="minor", colors="w", length=3, labelsize=13)

    #             minorLocator = AutoMinorLocator()
    #             ax.yaxis.set_minor_locator(minorLocator)
    #             ax.tick_params(direction="in", length=3, which="minor", colors="grey", labelsize=13)
    #             ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    #             ax.tick_params(axis="y", direction="in", pad=-30, color="white", labelcolor="white")
    #             ax.tick_params(axis="x", direction="in", pad=-17, color="white", labelcolor="white")

    #             ax.set_xlabel("Time (days)", fontsize=10, color="white")

    #             # ax.set_axis_bgcolor("#03012d")  # depending on what version of Python you're using.
    #             ax.set_facecolor("#03012d")

    #             ## save the image
    #             im_name = "./test_lc_dataloader_" + str(i) + ".png"
    #             path = "/mnt/zfsusers/shreshth/pht_project/data/examples"
    #             plt.savefig("%s/%s" % (path, im_name), format="png")
                
    #         t.update()
    
    # # save no label tics to file
    # print("no label tics: ", train_dataloader.dataset.no_label_tics)
    # with open('/mnt/zfsusers/shreshth/pht_project/data/pht_labels/no_label_tics.csv','w') as out:
    #     csv_out = csv.writer(out)
    #     csv_out.writerow(['tic_id','sec'])
    #     for row in train_dataloader.dataset.no_label_tics:
    #         csv_out.writerow(row)
