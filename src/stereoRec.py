import cv2
import numpy as np
from skimage.feature import match_template
from matplotlib import pyplot as plt


def loadImages():
    L = np.float32(cv2.imread('../data/TsukubaL.png',0))
    R = np.float32(cv2.imread('../data/TsukubaR.png',0))
    return (L, R)

# Scales image suche that scaledIm.shape[0] % N = 0
def scaleIm(I, scale, N, withBorder):
    withBorder=True
    halfwidth = np.int((N-1)/2)
    I = cv2.resize(I, (0,0), fx=scale, fy=scale)
    if withBorder:
        I = cv2.copyMakeBorder(np.array(I,dtype = np.uint8),halfwidth,halfwidth,halfwidth,halfwidth,cv2.BORDER_CONSTANT)

    return I

# Returns a vector Map where Map[i] is the disparity of NbyN patch nr i
# Perhaps use suggested matching method?


# Map is the disparity of one scale-level down
# Must find a way to use it to narrow search
def computeMap(L, R, N, M, scale, OldMap, isFirst):
    
    OldMap = OldMap.astype(int)
    (rowss, colss) = L.shape
    L = scaleIm(L, scale, N, isFirst)
    R = scaleIm(R, scale, N, isFirst)
  
    halfwidth = int((N-1)/2)
    (rows, cols) = L.shape
    rows = rows
    cols = cols 
    NewMap = np.zeros((rows, cols))
    for rowInd in range(halfwidth, rows - halfwidth):
        for colInd in range(halfwidth, cols - halfwidth):
            offset = OldMap[rowInd, colInd] + colInd
            lrow = L[rowInd-halfwidth:rowInd+halfwidth+1,colInd-halfwidth:colInd+halfwidth+1]
            rrow = R[rowInd-halfwidth:rowInd+halfwidth+1,max(0,offset - M - halfwidth):min(offset + M + halfwidth+1, cols)]
            startInd = max(0,-(offset - M - halfwidth))
            signal = match_template(rrow, lrow)
            NewMap[rowInd, colInd] = np.argmax(signal) + startInd -M

            
    return NewMap


# Recusively call itself for lower lever Map
def getDisparityMap(L, R, N, M):

    (sx,sy) = scaleIm(L, 1/8, N, True).shape
    Map = np.zeros((sx,sy)).astype(int)
    MapL = computeMap(L, R, N, M, 1/8, Map, True)
    MapR = computeMap(R, L, N, M, 1/8, Map, True)

    for i in range(sx):
        for j in range(sy):
            if MapR[i,j + int(MapL[i,j])]+MapL[i,j] != 0:
                MapL[i,j] = 0
                
    for i in range(sx):
        for j in range(sy):
            if MapL[i,j + int(MapR[i,j])]+MapR[i,j] != 0:
                MapR[i,j] = 0
    
    plt.subplot(421), plt.imshow(MapR)
    
    (sx,sy) = scaleIm(L, 1/4, N, False).shape
    MapL = cv2.resize(MapL, (sy,sx))
    MapL = MapL*2
    MapR = cv2.resize(MapR, (sy,sx))
    MapR = MapR*2
    MapL = computeMap(L, R, N, 2*M, 1/4, MapL, False)
    MapR = computeMap(R, L, N, 2*M, 1/4, MapR, False)
    
    plt.subplot(422), plt.imshow(MapR)

    for i in range(sx):
        for j in range(sy):
            if MapR[i,j + int(MapL[i,j])]+MapL[i,j] != 0:
                MapL[i,j] = 0
                
    for i in range(sx):
        for j in range(sy):
            if MapL[i,j + int(MapR[i,j])]+MapR[i,j] != 0:
                MapR[i,j] = 0
    
    plt.subplot(423), plt.imshow(MapR)

    (sx,sy) = scaleIm(L, 1/2, N, False).shape
    MapL = cv2.resize(MapL, (sy,sx))
    MapL = MapL*2
    MapR = cv2.resize(MapR, (sy,sx))
    MapR = MapR*2
    MapL = computeMap(L, R, N, 4*M, 1/2, MapL, False)
    MapR = computeMap(R, L, N, 4*M, 1/2, MapR, False)
  
    plt.subplot(424), plt.imshow(MapR)

    
    for i in range(sx):
        for j in range(sy):
            if MapR[i,j + int(MapL[i,j])]+MapL[i,j] != 0:
                MapL[i,j] = 0
                
    for i in range(sx):
        for j in range(sy):
            if MapL[i,j + int(MapR[i,j])]+MapR[i,j] != 0:
                MapR[i,j] = 0
                
    plt.subplot(425), plt.imshow(MapR)
    
    #MapL = np.zeros(MapL.shape)
    (sx,sy) = scaleIm(L, 1, N, False).shape
    MapL = cv2.resize(MapL, (sy,sx))
    MapL = MapL*2
    MapR = cv2.resize(MapR, (sy,sx))
    MapR = MapR*2
    MapL = computeMap(L, R, N, 8*M, 1, MapL, False)
    MapR = computeMap(L, R, N, 8*M, 1, MapR, False)

    plt.subplot(426), plt.imshow(MapR)
    
    for i in range(sx):
        for j in range(sy):
            if MapR[i,j + int(MapL[i,j])]+MapL[i,j] != 0:
                MapL[i,j] = 0
                
    for i in range(sx):
        for j in range(sy):
            if MapL[i,j + int(MapR[i,j])]+MapR[i,j] != 0:
                MapR[i,j] = 0

    return (MapL, MapR)
    
    plt.subplot(427), plt.imshow(MapR)

    

def main():
    N = 11 #Patch size
    M = 2 #Serach readius
    (L, R) = loadImages()
    (MapL, MapR) = getDisparityMap(R, L, N, M)
    plt.subplot(221), plt.imshow(L, 'gray')
    plt.subplot(222), plt.imshow(R, 'gray')
    plt.subplot(223), plt.imshow(MapL, 'gray')
    plt.subplot(224), plt.imshow(MapR, 'gray')

  

if __name__ == "__main__":
    main()


