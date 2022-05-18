
import numpy as np

from torchvision import transforms

SHORTEST_LC = 17546

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
