import numpy as np
from skimage.io import imread
from skimage import img_as_float
from skimage.transform import resize

from stereoRec2 import loadImages, getDisparityMap, plotFig


def disparity_error_statistics(our, ref):
    if np.shape(our) != np.shape(ref): # if dims don't fit, it's probably because of lost margins
        our = resize(our, np.shape(ref)) # correct this, by resizing to the same size
    err = np.abs(our-ref) # ignore sign of error
    # return mean and std of the error, and count large ones
    return [np.mean(err), np.std(err), len(np.where(err >= 3)[0])]


if __name__ == "__main__":
    for name in ["Tsukuba", "venus", "map"]:
        print("loading " + name)
        L, R, ref = loadImages(name)
        imsize = np.size(L)
        # test hyperparams
        for N in [5,7,9,11]:
            for M in [2,3,4]:
                MapL, _ = getDisparityMap(L, R, N, M, 1)
                MapL = np.abs(MapL) # ignore sign of disparity
                plotFig(MapL, name, N, M)
                stats = disparity_error_statistics(MapL, ref)
                print(N, "&", M, "&", stats[0], "&", stats[1], "&", stats[2], "&", stats[2]/imsize, "\\\\")
            print("\\hline")
