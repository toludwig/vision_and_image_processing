from skimage import img_as_float
from skimage.io import imread
from skimage.transform import rescale
from skimage.feature import corner_harris, corner_peaks
import numpy as np
import matplotlib.pyplot as plt


def match_layer(L, R, N):
    '''
    L and R are the left and right images
    N is the patch size (side length of the square patch)
    '''
    l = rescale(L, 0.5)
    r = rescale(R, 0.5)
    Rows, Cols = (s//N for s in np.shape(l))
    disparities = np.zeros((Rows, Cols))

    for row in range(Rows):
        l_row = l[row*N : (row+1)*N]
        r_row = r[row*N : (row+1)*N]
        partitions = [k*N for k in range(1, Cols+1)]
        l_patches = np.split(l_row, partitions, axis=1)[:-1]
        r_patches = np.split(r_row, partitions, axis=1)[:-1]
        row_mapping = match_patches(l_patches, r_patches)
        for i, j in row_mapping.items():
            disparities[row, i] = j-i # how much is patch in r
            # shifted to the right relative to corr. patch in l
    return disparities


def match_patches(a_patches, b_patches):
    # build up a similarity matrix
    n = len(a_patches) # = len(b_patches)
    similarities = np.zeros((n,n))
    for i, a in enumerate(a_patches):
        for j, b in enumerate(b_patches):
            similarities[i,j] = np.linalg.norm(a-b)**2
    # match most similar patches
    mapping = {} # {a_patch : b_patch}
    for _ in range(n):
        i, j = np.unravel_index(np.argmin(similarities), (n,n)) # coords of min
        mapping.update({i: j})
        # cross out (= set infinity) rows/cols of already mapped patches
        similarities[i,:] = [np.inf for r in range(n)] # rows
        similarities[:,j] = [np.inf for c in range(n)] # cols
    return mapping


if __name__ == "__main__":
    # L = img_as_float(imread("../data/TsukubaL.png", as_grey=True))
    # R = img_as_float(imread("../data/TsukubaR.png", as_grey=True))

    # L = img_as_float(imread("../data/map/im0.pgm", as_grey=True))
    # R = img_as_float(imread("../data/map/im1.pgm", as_grey=True))

    L = np.arange(100).reshape(10, 10)
    R = L+1

    disp = match_layer(L, R, 1)
    print(disp)

    plt.figure()
    plt.imshow(disp)
    plt.show()
