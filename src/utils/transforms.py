"""
"""

import numpy as np

import pandas as pd

import torch


class ToFloatTensor(object):
    """Convert numpy array to float tensor.
    """
    def __init__(self):
        pass

    def __call__(self, x):
        return torch.tensor(x, dtype=torch.float)


class NormaliseFlux(object):
    """Normalise the flux
    TODO check what normalisation is best, add optional args
    """
    def __init__(self):
        pass

    def __call__(self, x):
        # check if negative
        if np.any(x < 0):
            x = -x
        # normalise
        x /= np.nanmedian(x)
        # median at 0
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


class GaussianNoise(object):
    """Add Gaussian noise to the data
    """
    def __init__(self, prob, window=200, std=0.5):
        self.prob = prob
        self.window = window
        self.std = std
    
    def __call__(self, x):
        if np.random.rand() < self.prob:
            # calculate rolling std
            rolling_std = pd.Series(x).rolling(self.window).apply(lambda x : np.nanstd(x)).fillna(method='bfill').values
            # add noise (keeping the original nans as nans)
            x += np.random.normal(0, rolling_std*self.std)
            # x = np.nansum([x, np.random.normal(0, rolling_std*self.std)], axis=0)
            # add the nans back again
            # x[x == 0] = np.nan
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
        # compute rolling standard deviation and median
        rolling_std = pd.Series(x).rolling(self.window).apply(lambda x : np.nanstd(x)).fillna(method='bfill').values
        rolling_median = pd.Series(x).rolling(self.window).apply(lambda x : np.nanmedian(x)).fillna(method='bfill').values
        # remove outliers
        x[np.abs(x - rolling_median) > self.std_dev * rolling_std] = np.nan
        
        return x
