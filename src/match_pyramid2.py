from skimage import img_as_float
from skimage.io import imread
from skimage.transform import rescale
import numpy as np
import matplotlib.pyplot as plt

def match_layer(L, R, N):
    Rows, Cols = [s // N for s in np.shape(L)]
    partitions = [k*N for k in range(1, Cols+1)]
    disparities = np.zeros((Rows, Cols))
    for row in range(Rows):
        L_row = L[row*N:(row+1)*N]
        R_row = R[row*N:(row+1)*N]
        L_patches = np.split(L_row, partitions, axis=1)[:-1]
        R_patches = np.split(R_row, partitions, axis=1)[:-1]
        matches = twoway_match_cross_corr(L_patches, R_patches)
        for col, m in enumerate(matches):
            if m != -1:
                disparities[row, col] = m-col # positive for right shift
    print(disparities)


def match_cross_corr(l_patches, r_patches, threshold=0.5):
    # copied from Solem, 2012, p. 50
    N = np.size(l_patches[0])
    d = - np.ones((len(l_patches), len(r_patches))) # pair-wise distances
    for i in range(len(l_patches)):
        for j in range(len(r_patches)):
            d1 = (l_patches[i] - np.mean(l_patches[i])) / np.std(l_patches[i])
            d2 = (r_patches[j] - np.mean(r_patches[j])) / np.std(r_patches[j])
            ncc_value = np.sum(d1 * d2) / N
            if ncc_value > threshold:
                d[i,j] = ncc_value
    ndx = np.argsort(-d)
    matchscores = ndx[:,0]
    return matchscores


def twoway_match_cross_corr(l_patches, r_patches):
    matches_lr = match_cross_corr(l_patches, r_patches)
    matches_rl = match_cross_corr(r_patches, l_patches)
    ndx_lr = np.where(matches_lr >= 0)[0]
    # remove matches that are not symmetric
    for n in ndx_lr:
        if matches_rl[matches_lr[n]] != n:
            matches_lr[n] = -1
    return matches_lr



if __name__ == "__main__":
    L = img_as_float(imread("../data/TsukubaL.png", as_grey=True))
    R = img_as_float(imread("../data/TsukubaR.png", as_grey=True))
    L = L[0:60, 0:60]
    R = R[0:60, 0:60]

    match_layer(L, R, 3)
