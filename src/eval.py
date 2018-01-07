import numpy as np
from skimage.io import imread
from skimage import img_as_float
from skimage.transform import resize

from stereoRec2 import loadImages, getDisparityMap, plotFig


def disparity_error_statistics(our, ref):
    if np.shape(our) != np.shape(ref): # if dims don't fit, it's probably because of lost margins
        our = resize(our, np.shape(ref)) # correct this, by resizing to the same size
    err = our-ref
    # return mean and std of the error, and count large ones
    return [np.mean(err), np.std(err), len(np.where(err >= 3)[0])]


if __name__ == "__main__":
    for name in ["map", "Tsukuba", "venus"]:
        print("loading " + name)
        L, R, ref = loadImages(name)
        # test hyperparams
        for N in [3, 5, 7, 9, 11]:
            for M in [1, 2, 3]:
                MapL, _ = getDisparityMap(L, R, N, M, 1)
                plotFig(MapL, name, N, M)
                stats = disparity_error_statistics(MapL, ref)
                print(N, M, *stats)
