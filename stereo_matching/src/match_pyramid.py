from skimage import img_as_float
from skimage.io import imread
from skimage.transform import rescale
import numpy as np
import matplotlib.pyplot as plt



def match_pyramid(L, R, N, M,
                  layers, factor):
    '''
    We start with the topmost (=corsest, small-scale) level and compute disparities of patches.
    When scaling up again the patch size stays constant, so a patch on the new
    level corresponds to factor**2 patches in the previous level.
    Since we know the disparities at the lower level we can now match
    the corresponding patches (+/- M) on this level by considering them as new images.
    We do so recursively and stop at the top-level, the original image.
    Arguments:
    L and R are the left and right images
    N is the patch size (side length of the square patch)
    M is the tolerance radius for exploring the neighborhood of the prior from the higher level (+/- M)
    layers is the number of layers excluding the original, layers > 0
    factor is the downsampling factor between layers, 1 < factor (e.g. 2 means half size)
    '''
    if layers == 0:
        return match_layer(L, R, N)
    else:
        # rescale the images
        l = rescale(L, 1/factor)
        r = rescale(R, 1/factor)

        disparities = match_pyramid(l, r, N, M,
                                    layers-1, factor)

        new_disp = np.zeros((np.shape(disparities)[0]*factor,
                             np.shape(disparities)[1]*factor))
        Rows, Cols = np.shape(disparities)
        for row in range(Rows):
            for col in range(Cols):
                L_patch = L[row*N*factor : (row+1)*N*factor,
                            max(col-M, 0)*N*factor : min(col+1+M, Cols-1)*N*factor]
                disp = int(disparities[row, col])
                R_patch = R[row*N*factor : (row+1)*N*factor,
                            max(col+disp-M, 0)*N*factor : min(col+disp+1+M, Cols-1)*N*factor]
                patch_disp = match_layer(L_patch, R_patch, N)
                new_disp[row*factor : (row+1)*factor,
                         max(col-M, 0)*factor : min(col+1+M, Cols-1)*factor] = disparities[row, col] * factor + patch_disp
        return new_disp


def match_layer(L, R, N):
    '''
    L and R are the left and right images
    N is the patch size (side length of the square patch)
    The right image may be wider than the left.
    '''
    L_Rows, L_Cols = (s//N for s in np.shape(L))
    R_Rows, R_Cols = (s//N for s in np.shape(R))
    disparities = np.zeros((L_Rows, L_Cols))

    for row in range(L_Rows):
        L_row = L[row*N : (row+1)*N]
        R_row = R[row*N : (row+1)*N]
        L_partitions = [k*N for k in range(1, L_Cols+1)]
        R_partitions = [k*N for k in range(1, R_Cols+1)]
        L_patches = np.split(L_row, L_partitions, axis=1)[:-1]
        R_patches = np.split(R_row, R_partitions, axis=1)[:-1]

        row_mapping = match_patches(L_patches, R_patches)
        for i, j in row_mapping.items():
            disparities[row, i] = j-i # how much is patch in r
            # shifted to the right relative to corr. patch in l
    return disparities


def match_patches(l_patches, r_patches):
    '''
    Left and right patches are matched in global greedy matching.
    There may be more right than left patches because matching is done from left to right.
    Returns a mapping of indices {l : r}.
    '''
    # build up a similarity matrix
    l_n = len(l_patches)
    r_n = len(r_patches)
    if r_n == 0: # sanity check
        return {}
    similarities = np.zeros((l_n,r_n))
    for i, l in enumerate(l_patches):
        for j, r in enumerate(r_patches):
            similarities[i,j] = np.linalg.norm(l-r)**2
    # match most similar patches
    mapping = {} # {l_patch : r_patch}
    for _ in range(l_n):
        i, j = np.unravel_index(np.argmin(similarities), (l_n,r_n)) # coords of min
        mapping.update({i: j})
        # cross out (= set infinity) rows/cols of already mapped patches
        similarities[i,:] = [np.inf for cell in range(r_n)] # rows
        similarities[:,j] = [np.inf for cell in range(l_n)] # cols
    return mapping


def match_cross_correllation(l_patches, r_patches, threshold=0.5):
    # copied from Solem, 2012, p. 50
    N = np.size(l_patches[0])
    d = - np.ones((len(l_patches), len(r_patches))) # pair-wise distances
    for i in range(len(l_patches)):
        for j in range(len(r_patches)):
            d1 = (l_patches[i] - np.mean(l_patches[i])) / np.std(l_patches[i])
            d2 = (r_patches[j] - np.mean(r_patches[j])) / np.std(r_patches[j])
            ncc_value = np.sum(d1 * d2) / (N-1)
            if ncc_value > threshold:
                d[i,j] = ncc_value
    ndx = np.argsort(-d)
    return ndx



if __name__ == "__main__":
    L = img_as_float(imread("../data/TsukubaL.png", as_grey=True))
    R = img_as_float(imread("../data/TsukubaR.png", as_grey=True))

    #L = img_as_float(imread("../data/map/im0.pgm", as_grey=True))
    #R = img_as_float(imread("../data/map/im1.pgm", as_grey=True))

    #L = np.arange(100).reshape(10, 10)
    #R = L+1

    disp = match_pyramid(L, R, 3, 1,
                         2, 2)
    print(disp)

    plt.figure()
    plt.imshow(disp, cmap="gray")
    plt.show()
