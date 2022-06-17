"""Light curve transformations.
"""

import warnings

from glob import glob

import numpy as np

import pandas as pd

import torch

from utils.utils import read_lc_csv, get_sectors

class ToFloatTensor(object):
    """Convert numpy array to float tensor.
    """
    def __init__(self):
        pass

    def __call__(self, x):
        return torch.tensor(x, dtype=torch.float)


class NormaliseFlux(object):
    """Normalise the flux so median = 1 (or -1)
    """
    def __init__(self):
        pass

    def __call__(self, x):
        median = np.nanmedian(x)
        # normalise
        x /= np.abs(np.nanmedian(x))
        # to fix numpy => torch byte error
        x = x.astype(np.float64)  
        return x


class MedianAtZero(object):
    """Median at zero
    """
    def __init__(self):
        pass

    def __call__(self, x):
        median = np.nanmedian(x)
        # median at 0
        if median < 0:
            x += 1
        else:
            x -= 1
        # to fix numpy => torch byte error
        x = x.astype(np.float64)  
        return x


class Cutoff(object):
    """Restrict LC data to a certain length. Cut start or end randomly
    TODO we could also pad to the median length
    """
    def __init__(self, length=2700):
        self.length = length
    
    def __call__(self, x):
        # choose a random start to cut
        if len(x) == self.length:
            return x
        elif len(x) > self.length:
            start = np.random.randint(0, len(x) - self.length)
            return x[start:start+self.length]
        else:
            raise ValueError(f"Length of light curve is {len(x)} but should be minimum {self.length}")


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
            x[np.arange(start, start + n)] = np.nan

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
            # check not overlapping sections
            overlapping = True
            while overlapping:
                start2 = np.random.randint(0, N - n)
                end2 = start2 + n
                if (start1 < start2 < end1) or (start1 < end2 < end1):
                    overlapping = True
                else:
                    overlapping = False
            # swap
            x[start1:end1], x[start2:end2] = x[start2:end2].copy(), x[start1:end1].copy()

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


class GaussianNoise(object):
    """Add Gaussian noise to the data
    """
    def __init__(self, prob, window=200, std=0.5):
        self.prob = prob
        self.window = window
        self.std = std
    
    def __call__(self, x):
        if np.random.rand() < self.prob:
            # suppress warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                # calculate rolling std
                rolling_std = np.zeros(x.shape)
                nrows = len(x) - self.window + 1
                n = x.strides[0]
                a2D = np.lib.stride_tricks.as_strided(x, shape=(nrows,self.window), strides=(n,n))
                rolling_std[self.window-1:] = np.nanstd(a2D, axis=1)
                rolling_std[:self.window-1] = rolling_std[self.window-1]

                # add noise (keeping the original nans as nans)
                x += np.random.normal(0, rolling_std*self.std)
        return x


class InjectLCNoise(object):
    """Add another LC to the data to simulate noise.
    NOTE: This is too inefficient (need to cache but will be repeated with training set cache)
    """
    def __init__(self, prob, bin_factor, data_root_path, data_split):
        self.prob = prob
        self.bin_factor = bin_factor
        self.data_root_path = data_root_path
        self.sectors = get_sectors(data_split)
        # get list of all lc files
        self.lc_file_list = []
        for sector in self.sectors:
            # print(f"sector: {sector}")
            new_files = glob(f"{self.data_root_path}/lc_csvs_cdpp/Sector{sector}/*binfac-{self.bin_factor}.csv", recursive=True)
            print("num. files found: ", len(new_files))
            self.lc_file_list += new_files
        print("total num. LC files found: ", len(self.lc_file_list))
        # get labels
        labels_df = pd.DataFrame()
        for sector in self.sectors:
            labels_df = pd.concat([labels_df, pd.read_csv(f"{self.data_root_path}/pht_labels/summary_file_sec{sector}.csv")], axis=0)
        print("num. total labels (including simulated data): ", len(labels_df))

        # removing non-zero labels and simulated data
        labels_df = labels_df[~labels_df["subject_type"]]
        labels_df = labels_df[labels_df["maxdb"] == 0]
        zero_tics = labels_df["TIC_ID"].to_list()
        print("num. zero LC labels: ", len(zero_tics))

        # remove non-zero LCs from list
        print("num. LC files before removing non-zero LCs: ", len(self.lc_file_list))
        self.lc_file_list = [x for x in self.lc_file_list if int(x.split("/")[-1].split("-")[1].split("_")[0]) in zero_tics]
        print("num. LC files after removing non-zero LCs: ", len(self.lc_file_list))


    def __call__(self, x):
        if np.random.rand() < self.prob:
            injected = False 
            while not injected:
                # choose a random lc
                lc_file = np.random.choice(self.lc_file_list)
                lc = read_lc_csv(lc_file)
                inj_flux = lc["flux"]
                if inj_flux is not None:
                    # normalise flux
                    median = np.nanmedian(inj_flux)
                    inj_flux /= np.abs(median)
                    # if median is negative, put back to 1
                    if median < 0:
                        inj_flux += 2
                    # make same length as x
                    if len(inj_flux) >= len(x):
                        inj_flux = inj_flux[:len(x)]
                    else:
                        inj_flux = np.pad(inj_flux, (0, len(x) - len(inj_flux)), "constant", constant_values=1)
                    # fill in nans
                    inj_flux = np.nan_to_num(inj_flux, nan=1.0)
                    # add noise
                    x = x * inj_flux
                    injected = True
        return x


class RemoveOutliersPercent(object):
    """Remove data which is more than percentage away from the median
    Params:
    - percent_change (float): remove data which is more than this fraction away from the median
    """
    def __init__(self, percent_change=None):
        self.percent_change = percent_change
    
    def __call__(self, x):
        # check if normalised already
        median = np.nanmedian(x)
        if np.isclose(median, 0):
            threshold = self.percent_change
        else:
            threshold = self.percent_change * median
        
        x[np.abs(x - median) > threshold] = np.nan

        return x


class RemoveOutliers(object):
    """Remove data which is more than x rolling standard deviations away from the median
    Params:
    - std_dev (float): remove data which is more than this many rolling stds away from the rolling median
    """
    def __init__(self, window=200, std_dev=3.0):
        self.std_dev = std_dev
        self.window = window
    
    def __call__(self, x):
        # suppress warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            # numpy version
            rolling_std = np.zeros(x.shape)
            rolling_median = np.zeros(x.shape)
            nrows = len(x) - self.window + 1
            n = x.strides[0]
            a2D = np.lib.stride_tricks.as_strided(x, shape=(nrows,self.window), strides=(n,n))
            rolling_std[self.window-1:] = np.nanstd(a2D, axis=1)
            rolling_std[:self.window-1] = rolling_std[self.window-1]
            rolling_median[self.window-1:] = np.nanmedian(a2D, axis=1)
            rolling_median[:self.window-1] = rolling_median[self.window-1]

            # remove outliers
            x[np.abs(x - rolling_median) > self.std_dev * rolling_std] = np.nan
        
        return x
