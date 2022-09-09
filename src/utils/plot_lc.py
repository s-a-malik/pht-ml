"""plot_lc.py
Saves a plot of the light curve, both raw and rebinned. Saves another plot applying the training transform
command:
"""

import os
from glob import glob
from argparse import ArgumentParser
from copy import deepcopy 

import numpy as np
import pandas as pd

import torch
import torchvision

import astropy.io.fits as pf
from astropy.table import Table

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import AutoMinorLocator

from utils import transforms

matplotlib.use("Agg")


def rebin(arr, new_shape):
    """Function to bin the data to make it easier to visualise.
    Do not use this for ML purposes as will lose information.
    """
    shape = (
        new_shape[0],
        arr.shape[0] // new_shape[0],
        new_shape[1],
        arr.shape[1] // new_shape[1],
    )
    return np.nanmean(np.nanmean(arr.reshape(shape), axis=(-1)), axis=1)    # nan friendly
    # return arr.reshape(shape).mean(-1).mean(1)


def plot_from_csv(lcfile):
    """Plot from csv file
    """

    # load the curve
    df = pd.read_csv(lcfile)
    time = df["time"].values
    flux = df["flux"].values

    ## define the plotting area
    fig, ax = plt.subplots(figsize=(16, 5))
    plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.05)

    ax.plot(
        time,
        flux,
        color="darkorange",
        marker="o",
        markersize=1,
        lw=0,
        label="raw",
    )


    ## define that length on the x axis - I don't want it to display the 0 point
    delta_flux = np.nanmax(flux) - np.nanmin(flux)

    ## set the y lim.
    percent_change = delta_flux * 0.1
    ax.set_ylim(np.nanmin(flux) - percent_change, np.nanmax(flux) + percent_change)

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

    ax.set_xlabel("Time (days)", fontsize=10, color="white")

    # ax.set_axis_bgcolor("#03012d")  # depending on what version of Python you're using.
    ax.set_facecolor("#03012d")

    ## save the image
    path = "./data/examples"
    _, file_name = os.path.split(lcfile)

    plt.savefig("%s/%s.png" % (path, file_name), format="png")



def plot_lc_test(args):
    """
    Function to plot the TESS LCs.
    Uses the PDCSAP flux - processed by the TESS pipeline to have some of the systematics removed.

    Input
    ------
    sec: Sector number
    tic_id: TIC ID
    binfac : binning factor.

    Output
    ------
    Figure showing the TESS LC for a given TIC ID. Showing the binned and unbinned data.
    """
    binfac = args.binfac
    tic_id = args.tic_id
    sec = args.sec
    seed = args.seed
    path = args.plot_path
    # set seed
    np.random.seed(seed)

    lcfile = get_lc_file(sec, tic_id)

    # open the file in context manager
    with pf.open(lcfile) as hdul:
        # print(hdul.info())

        ## import all the header information,
        tic = int(hdul[0].header["TICID"])
        print("Reading data for TIC{}".format(tic))

        d = hdul[1].data
        t = d["TIME"]
        f1 = d["SAP_FLUX"]
        fbkg = d["SAP_BKG"]
        f1 /= np.nanmedian(f1)
        f01 = d["SAP_FLUX"]
        f2 = d["PDCSAP_FLUX"]  # the processed flux
        # f2 /= np.nanmedian(f2)
        f02 = d["PDCSAP_FLUX"]

        ## bin data
        N = len(t)
        n = int(np.floor(N / binfac) * binfac)
        X = np.zeros((2, n))
        X[0, :] = t[:n]
        X[1, :] = f02[:n]
        Xb = rebin(X, (2, int(n / binfac)))
        time_binned = Xb[0]
        flux_binned = Xb[1]

        time_binned -= time_binned[0]

        q = d[
            "QUALITY"
        ]  # I use these to plot the centroid positions to identify false positives but you probably dom't need them.
        x1 = d["MOM_CENTR1"]
        x1 -= np.nanmedian(x1)
        y1 = d["MOM_CENTR2"]
        y1 -= np.nanmedian(y1)
        x2 = d["POS_CORR1"]
        x2 -= np.nanmedian(x2)
        y2 = d["POS_CORR2"]
        y2 -= np.nanmedian(y2)

        # l = np.isfinite(time) * np.isfinite(flux) * (q == 0)  # from simulation script

        l = q > 0
        l2 = q <= 0  # can also plot the removed data points if you care about that
        
        t0 = t[0]  # make the time start at 0 (so that the timeline always runs from 0 to 27.8 days)
        t -= t0

        tic = int(hdul[0].header["TICID"])
        sec = int(hdul[0].header["SECTOR"])
        cam = int(hdul[0].header["CAMERA"])
        chi = int(hdul[0].header["CCD"])
        tessmag = hdul[0].header["TESSMAG"]
        teff = hdul[0].header["TEFF"]
        srad = hdul[0].header["RADIUS"]
        # can verify these with astroquery...

        scc = "%02d%1d%1d" % (sec, cam, chi)
        print(
            f"TIC {tic}, Sector {sec}, Camera {cam}, CCD {chi}, TESSmag {tessmag}, Teff {teff}, Radius {srad}, SCC {scc}"
        )
    # ------------------------------------------
    # plot the whole LC - binned and unbinned
    # ------------------------------------------

    ## define the plotting area
    fig, ax = plt.subplots(figsize=(16, 5))
    plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.05)

    ## plot the binned and unbinned LC
    ax.plot(
        t[l2],
        f2[l2],
        color="royalblue",
        marker="o",
        markersize=1,
        lw=0,
        label="unbinned",
    )
    # ax.plot(
    #     t,
    #     f2,
    #     color="darkorange",
    #     marker="o",
    #     markersize=1,
    #     lw=0,
    #     label="no quality filter",
    # )
    # ax.plot(t[l2], f1[l2], color = 'darkorange', marker = 'o', markersize=1, lw = 0, label = 'unprocessed')
    ax.plot(
        time_binned,
        flux_binned,
        color="white",
        marker="o",
        markersize=2,
        lw=0,
        label="binned",
    )


    ## define that length on the x axis - I don't want it to display the 0 point
    delta_flux = np.nanmax(f2[l2]) - np.nanmin(f2[l2])
    print(delta_flux, np.nanmin(f2[l2]), np.nanmax(f2[l2]))
    ## set the y lim.
    percent_change = delta_flux * 0.1
    print(percent_change)
    print(np.nanmin(f2[l2]) - percent_change, np.nanmax(f2[l2]) + percent_change)
    ax.set_ylim(np.nanmin(f2[l2]) - percent_change, np.nanmax(f2[l2]) + percent_change)
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
    ax.tick_params(axis="y", direction="in", pad=-50, color="white", labelcolor="white")
    ax.tick_params(axis="x", direction="in", pad=-17, color="white", labelcolor="white")
    ax.set_xlabel("Time (days)", fontsize=10, color="white")

    # ax.set_axis_bgcolor("#03012d")  # depending on what version of Python you're using.
    ax.set_facecolor("#03012d")

    ## save the images
    _, file_name = os.path.split(lcfile)
    save_name = file_name + f"_binfac-{binfac}.png"

    print(f"saving to {path}/{save_name}")
    plt.savefig("%s/%s" % (path, save_name), dpi=300, format="png")


    # ------------------------------------------
    # plot the transformed LC
    # ------------------------------------------

    training_transform = torchvision.transforms.Compose([
        transforms.NormaliseFlux(),
        transforms.MedianAtZero(),
        transforms.MirrorFlip(prob=1.0),
        transforms.RandomDelete(prob=0.1, delete_fraction=0.1),
        transforms.RandomShift(prob=0.1, permute_fraction=0.25),
        transforms.ImputeNans(method="zero"),
        transforms.Cutoff(length=int(17500/binfac)),
        transforms.ToFloatTensor()
    ])

    # test tranforms - do not randomly delete or permute
    val_transform = torchvision.transforms.Compose([
        transforms.NormaliseFlux(),
        transforms.MedianAtZero(),
        transforms.ImputeNans(method="zero"),
        transforms.Cutoff(length=int(17500/binfac)),
        transforms.ToFloatTensor()
    ])

    other_flux_binned = deepcopy(flux_binned)
    flux_binned_train_transformed = training_transform(flux_binned)
    flux_binned_val_transformed = val_transform(other_flux_binned)
    plt.clf()
    ## define the plotting area
    fig, ax = plt.subplots(figsize=(16, 5))
    plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.05)

    ax.plot(
        time_binned[:len(flux_binned_val_transformed)],
        flux_binned_val_transformed,
        color="white",
        marker="o",
        markersize=2,
        lw=0,
        label="val",
    )
    ## define that length on the x axis - I don't want it to display the 0 point
    delta_flux = np.nanmax(flux_binned_val_transformed) - np.nanmin(flux_binned_val_transformed)
    ## set the y lim.
    percent_change = delta_flux * 0.1
    # ax.set_ylim(np.nanmin(flux_binned_val_transformed) - percent_change, np.nanmax(flux_binned_val_transformed) + percent_change)

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
    ax.tick_params(axis="y", direction="in", pad=-50, color="white", labelcolor="white")
    ax.tick_params(axis="x", direction="in", pad=-17, color="white", labelcolor="white")
    ax.set_xlabel("Time (days)", fontsize=10, color="white")
    # ax.set_axis_bgcolor("#03012d")  # depending on what version of Python you're using.
    ax.set_facecolor("#03012d")

    ## save the images
    _, file_name = os.path.split(lcfile)
    save_name = file_name + f"_binfac-{binfac}_val_preprocessed.png"
    print(f"saving to {path}/{save_name}")
    plt.savefig("%s/%s" % (path, save_name), dpi=300, format="png")

    plt.clf()

    ## define the plotting area
    fig, ax = plt.subplots(figsize=(16, 5))
    plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.05)

    ax.plot(
        time_binned[:len(flux_binned_train_transformed)],
        flux_binned_train_transformed,
        color="white",
        marker="o",
        markersize=2,
        lw=0,
        label="train",
    )

    ## define that length on the x axis - I don't want it to display the 0 point
    delta_flux = np.nanmax(flux_binned_val_transformed) - np.nanmin(flux_binned_val_transformed)
    ## set the y lim.
    percent_change = delta_flux * 0.1
    # ax.set_ylim(np.nanmin(flux_binned_val_transformed) - percent_change, np.nanmax(flux_binned_val_transformed) + percent_change)

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
    ax.tick_params(axis="y", direction="in", pad=-50, color="white", labelcolor="white")
    ax.tick_params(axis="x", direction="in", pad=-17, color="white", labelcolor="white")
    ax.set_xlabel("Time (days)", fontsize=10, color="white")
    # ax.set_axis_bgcolor("#03012d")  # depending on what version of Python you're using.
    ax.set_facecolor("#03012d")

    # legend
    # ax.legend(loc="upper right", fontsize=10, facecolor="white", framealpha=0.5)

    ## save the images
    _, file_name = os.path.split(lcfile)
    save_name = file_name + f"_binfac-{binfac}_train_preprocessed.png"
    print(f"saving to {path}/{save_name}")
    plt.savefig("%s/%s" % (path, save_name), dpi=300, format="png")



def get_lc_file(sec, tic_id):
    """Get the file name from the TIC ID and sector
    """
    lc_root_path = "./data/TESS"

    # use sector as well to reduce the number of files to search
    lc_file = glob(os.path.join(lc_root_path, f"planethunters/Rel{sec}/Sector{sec}/**/*{tic_id}*.fit*"), recursive=True)
    print("found", lc_file, "for tic id", tic_id)
    return lc_file[0]


if __name__ == "__main__":

    ap = ArgumentParser(description="Script to plot tic ids")
    ap.add_argument("--binfac", type=int, help="Binning factor", default=7)
    ap.add_argument("--tic-id", type=int, help="TIC ID", default=461196191)
    ap.add_argument("--sec", type=int, help="Sector", default=10)
    ap.add_argument("--seed", type=int, help="Seed", default=0)
    ap.add_argument("--plot-path", type=str, help="Path to save the plots", default="./data/examples/lc_plots")

    args = ap.parse_args()

    plot_lc_test(args)

    # lc_csv = "./data/lc_csvs/test.csv"
    # plot_from_csv(lc_csv)
