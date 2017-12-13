import cv2
import numpy as np




def loadImages():
    L = np.float32(cv2.imread('../data/TsukubaL.png',0))
    R = np.float32(cv2.imread('../data/TsukubaR.png',0))
    return (L, R)

# Scales image suche that scaledIm.shape[0] % N = 0
def scaleIm(I, scale, N):
    (rows, cols)  = I.shape[:2]
    xs = rows*scale
    ys = cols*scale
    xs = np.int((xs//N)*N)
    ys =np.int((ys//N)*N)
    resized = cv2.resize(I, (xs,ys))
    return resized

# Returns a vector Map where Map[i] is the disparity of NbyN patch nr i
# Perhaps use suggested matching method?
def getMovements(lrow, rrow, cols, N):
    dim = cols//N
    movements = np.zeros((dim))
    sim = np.zeros((dim, dim))
    for lind in range(0, cols, N):
        patchL = lrow[:][lind:lind + N]
        for rind in range(0, cols, N): ## Want to do NCC. This does not work
            patchR = rrow[:][rind:rind + N]
            NCC = np.vdot(patchL, patchR)/(np.sum(patchL)*np.sum(patchR))
            sim[rind//N][lind//N] = NCC
    return movements

# Map is the disparity of one scale-level down
# Must find a way to use it to narrow search
def computeMap(L, R, N, scale, Map):
    return Map

# The lowest scaling level does not have a Map to guide it
def computeFirstMap(L, R, N, scale):
    L = scaleIm(L, scale, N)
    R = scaleIm(R, scale, N)
    (rows, cols) = L.shape
    Map = np.zeros((rows//N, cols//N))
    for rowInd in range(0, rows, N):
        lrow = L[rowInd:rowInd+N]
        rrow = R[rowInd:rowInd+N]
        Map[rowInd] = getMovements(lrow, rrow, cols, N)
    return Map

# Recusively call itself for lower lever Map
def getDisparityMap(L, R, N, scale):
    if scale == 1/8:
        return computeFirstMap(L, R, N, scale)
    else:
        Map = getDisparityMap(L, R, N, scale/2)
        return computeMap(L, R, N, scale, Map)

def main():
    N = 5 #Patch size
    (L, R) = loadImages()
    Map = getDisparityMap(L, R, N, 1)

if __name__ == "__main__":
    main()


