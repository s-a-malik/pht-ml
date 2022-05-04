"""data.py
Utility functions and classes for data manipulation.
"""

import os
import time
import glob
from tqdm.autonotebook import trange

import torch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import astropy.io.fits as pf


# SECTORS = list(range(10, 39))
SECTORS = [10, 11]
SHORTEST_LC = 17546 # from sector 10-38. Used to trim all the data to the same length

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
            # print(f"sector: {sector}")
            new_files = glob.glob(os.path.join(lc_root_path, f"Rel{sector}/Sector{sector}/**/*.fit*"), recursive=True)
            # print("num. files found: ", len(new_files))
            self.lc_file_list += new_files

            # length of light curve in this sector:
            x, tic, sec = self._read_lc(self.lc_file_list[-1])
            print("length of light curve in sector ", sec, ": ", x.shape, ", TIC: ", tic)
        print("total num. files found: ", len(self.lc_file_list))
        # get all the labels
        self.labels_df = pd.DataFrame()
        for sector in SECTORS:
            self.labels_df = pd.concat([self.labels_df, pd.read_csv(f"{labels_root_path}/summary_file_sec{sector}.csv")], axis=0)
        print("num. labels: ", len(self.labels_df))
        # TODO add simulated data


        self.no_label_tics = []

    def __len__(self):
        return len(self.lc_file_list)


    def __getitem__(self, idx):
        start = time.time()
        # get lc file
        lc_file = self.lc_file_list[idx]

        # read lc file
        x, tic, sec = self._read_lc(lc_file)
        # if corrupt return None and skip c.f. collate_fn
        if x is None:
            return (x, tic, sec), None


        # TODO fill in nans more effectively in the light curve
        x = np.nan_to_num(x)

        # TODO make all the same length - each sector is slightly different length (could do this in the collate fn also)   
        # should take the start or the end off?
        x = x[:SHORTEST_LC]

        # get label for this lc file (if exists) match sector 
        # TODO check if label does not exist
        # filter by sector and TIC ID
        # print(self.labels_df.loc[(self.labels_df["TIC_ID"] == tic) & (self.labels_df["sector"] == sec), "maxdb"])
        y = self.labels_df.loc[(self.labels_df["TIC_ID"] == tic) & (self.labels_df["sector"] == sec), "maxdb"].values
        if len(y) == 1:
            y = torch.tensor(y[0])
        elif len(y) > 1:
            print(y, "more than one label for TIC: ", tic, " in sector: ", sec)
            y = None
            self.no_label_tics.append((tic, sec))
        else:
            print(y, "label not found for TIC: ", tic, " in sector: ", sec)
            y = None
            self.no_label_tics.append((tic, sec))

        end = time.time()
        # print("time to get data", end - start)
        # return tensors
        return (torch.from_numpy(x), torch.tensor(tic), torch.tensor(sec)), y

    def _read_lc(self, lc_file):
        """Read light curve file
        """
        # open the file in context manager - catching corrupt files
        try:
            with pf.open(lc_file) as hdul:
                d = hdul[1].data
                t = d["TIME"]   # currently not using time
                f2 = d["PDCSAP_FLUX"]  # the processed flux
                # print(f2)
                median = np.nanmedian(f2)
                f2 /= median
                # TODO normalisation 
                # median at 0
                f2 -= median


                f2 = f2.astype(np.float64)  # to fix numpy => torch byte error
                
                # not currently used
                t0 = t[0]  # make the time start at 0 (so that the timeline always runs from 0 to 27.8 days)
                t -= t0

                tic = int(hdul[0].header["TICID"])
                sec = int(hdul[0].header["SECTOR"])
                cam = int(hdul[0].header["CAMERA"])
                chi = int(hdul[0].header["CCD"])
                tessmag = hdul[0].header["TESSMAG"]
                teff = hdul[0].header["TEFF"]
                srad = hdul[0].header["RADIUS"]
        except OSError:
            print("OSError: ", lc_file)
            return None, None, None

        return f2, tic, sec

    # load file path lists and count number of curves (save in an array?)

    # split these into train, val, test by index. 

    # then on get item, load the file. (need to ensure loading is fast - preload batches)

# TODO function to get file from tic id - needed for specific lookup 


# TODO collate fn to return a good batch of simulated and real data
def collate_fn(batch):
    """Collate function for filtering out corrupted data in the dataset
    Assumes that missing data are NoneType
    """
    # print("len(batch): ", len(batch))
    batch = [x for x in batch if x[0][0] is not None]
    # print("len(batch): ", len(batch))
    batch = [x for x in batch if x[1] is not None]
    # print("len(batch): ", len(batch))
    # batch = list(filter(lambda x: x[0][0] is not None, batch))  # filter out corrupted .fits files
    # batch = list(filter(lambda x: x[1] is not None, batch))     # filter out missing labels
    return torch.utils.data.dataloader.default_collate(batch)
    # return batch


# TODO augmentations/transformations


# splitting of data into train, val, test use file path lists.


def get_data_loader(data_root_path, labels_root_path, batch_size, shuffle=True, num_workers=0, pin_memory=False):
    """Get data loader
    """
    dataset = LCData(data_root_path, labels_root_path)
    data_loader = torch.utils.data.DataLoader(dataset,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                num_workers=num_workers,
                                                pin_memory=pin_memory,
                                                collate_fn=collate_fn)
    return data_loader




if __name__ == "__main__":
    
    LC_ROOT_PATH = "/mnt/zfsusers/shreshth/pht_project/data/TESS/planethunters"
    LABELS_ROOT_PATH = "/mnt/zfsusers/shreshth/pht_project/data/pht_labels"

    # lc_data = LCData(LC_ROOT_PATH, LABELS_ROOT_PATH)
    # print(len(lc_data))
    data_loader = get_data_loader(LC_ROOT_PATH, LABELS_ROOT_PATH, batch_size=1024, shuffle=False, num_workers=0, pin_memory=False)
    with trange(len(data_loader)) as t:
        for i, (x, y) in enumerate(data_loader):
            # print(i, x, y)
            if i % 100 == 0:
                print(i, x, y)
                print(x[0].shape, y.shape)

                fig, ax = plt.subplots(figsize=(16, 5))
                plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.05)

                ## plot the binned and unbinned LC
                ax.plot(list(range(len(x[0][0]))), x[0][0],
                    color="royalblue",
                    marker="o",
                    markersize=1,
                    lw=0,
                    label="unbinned",
                )
                ## save the image
                im_name = "./test_lc_dataloader_" + str(i) + ".png"
                path = "/mnt/zfsusers/shreshth/pht_project/data/examples"
                plt.savefig("%s/%s" % (path, im_name), format="png")
            t.update()
    print("no label tics: ", data_loader.dataset.no_label_tics)

    # use pin_memory to speed up loading on GPU
