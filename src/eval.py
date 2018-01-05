import numpy as np
from skimage.io import imread
from skimage import img_as_float
from skimage.transform import resize
import matplotlib.pyplot as plt

from stereoRec2 import loadImages, getDisparityMap


def disparity_error_statistics(our, ref):
    if np.shape(our) != np.shape(ref): # if dims don't fit, it's probably because of lost margins
        our = resize(our, np.shape(ref)) # correct this, by resizing to the same size
    err = our-ref
    # return mean and std of the error, and count large ones
    return [np.mean(err), np.std(err), len(np.where(err >= 3))]


if __name__ == "__main__":
    (L, R) = loadImages()
    ref = img_as_float(imread("../data/tsukuba/truedisp.row3.col3.pgm", as_grey=True))

    # test hyperparams
    for N in [1, 3, 5, 7, 9, 11]:
        for M in [1, 2, 3]:
            MapL, _ = getDisparityMap(L, R, N, M, 1)
            stats = disparity_error_statistics(MapL, ref)
            print(N, M, *stats)
