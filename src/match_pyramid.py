import numpy as np
from skimage import img_as_float
from skimage.io import imread
from skimage.transform import rescale
from skimage.feature import corner_harris, corner_peaks


def pyramid_match(L, R):
    l = rescale(L, 0.5)
    r = rescale(R, 0.5)

    N = 5
    Rows = len(l)//N
    Columns = l.shape()[1]//N
    mapping = numpy.zeros((len(l)//N + 1, len(r)//N + 1))

    for row in range(Rows):
        l_row = l[row*N : (row+1)*N]
        r_row = r[row*N : (row+1)*N]
        partitions = [k*N for k in range(len(l_row)//N)]
        row_mapping = match_patches(np.split(l_row, partitions, axis=1),
                                    np.split(r_row, partitions, axis=1))
        match

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
    L = img_as_float("data/TsukubaL.png")
    R = img_as_float("data/TsukubaR.png")

    pyramid_match(L, R)
