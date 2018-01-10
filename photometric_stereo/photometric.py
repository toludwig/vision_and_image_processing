import numpy as np
import skimage as si
import matplotlib.pyplot as plt

import ps_utils as ps

def beethoven():
    (I, mask, S) = ps.read_data_file("Beethoven.mat")
    print(I, mask, S)
    mask_coords = np.where(mask != 0)
    nz = len(mask_coords[0])
    J = np.zeros((3, nz))
    for i, (x,y) in enumerate(zip(*mask_coords)):
        J[:,i] = I[x,y]
    M = np.linalg.inv(S) @ J # albedo modulated normal field
    albedo = np.sum(np.abs(M), axis=0)
    alb_Im = np.zeros(np.shape(mask))
    alb_Im[mask_coords] = albedo
    plt.imshow(alb_Im, cmap="gray")
    plt.show()

    N = M / albedo # normal field by normalizing M by albedo
    n1 = np.zeros(np.shape(mask))
    n1[mask_coords] = N[0]
    n2 = np.zeros(np.shape(mask))
    n2[mask_coords] = N[1]
    n3 = np.zeros(np.shape(mask))
    n3[mask_coords] = N[2]
    print(np.shape(n1))
    Z = ps.simchony_integrate(n1, n2, n3, mask)
    ps.display_depth_matplotlib(Z)


def buddha():
    print(ps.read_data_file("Buddha.mat"))


if __name__ == "__main__":
    beethoven()
