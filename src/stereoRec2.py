import cv2
import numpy as np
from skimage.feature import match_template
from matplotlib import pyplot as plt


def loadImages():
    L = np.float32(cv2.imread('../data/TsukubaL.png',0))
    R = np.float32(cv2.imread('../data/TsukubaR.png',0))
   # L = np.float32(cv2.imread('../data/map/im0.pgm',0))
   # R = np.float32(cv2.imread('../data/map/im1.pgm',0))
    return (L, R)

# Scales image suche that scaledIm.shape[0] % N = 0
def scaleIm(I, scale, N):
    halfwidth = np.int((N-1)/2)
    I = cv2.resize(I, (0,0), fx=scale, fy=scale)
    I = cv2.copyMakeBorder(np.array(I,dtype = np.uint8),halfwidth,halfwidth,halfwidth,halfwidth,cv2.BORDER_REFLECT)

    return I

def scaleMap(I, sx, sy, N):
    halfwidth = np.int((N-1)/2)
    minn = I.min()
    I = np.uint8(I-minn)
    I = cv2.resize(I, (sy,sx))
    I = I + minn
    B = cv2.copyMakeBorder(I,halfwidth,halfwidth,halfwidth,halfwidth,cv2.BORDER_CONSTANT)
    print(I.min(), I.max())
    return B


def computeMap(L, R, N, M, scale, OldMap):
    
    OldMap = OldMap.astype(int)
    (rowss, colss) = L.shape
    L = scaleIm(L, scale, N)
    R = scaleIm(R, scale, N)
  
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
            signal = match_template(rrow, lrow)
            signInd = int(np.argmax(signal))
            
            if signal[0,signInd] < 0.5:
                NewMap[rowInd, colInd] = 0
            else:
                NewMap[rowInd, colInd]=range(max(halfwidth,offset - M),min(offset + M+1, cols-halfwidth))[signInd] - colInd
                
    return NewMap

def pyramidLevel(L, R, MapL, MapR, N, M, scale, isFirst):
    HW = int((N-1)/2)
    (sx,sy) = scaleIm(L, scale, N).shape
    (mx,my) = MapL.shape
    
    if isFirst == False:
        MapL = scaleMap(MapL[HW:mx-HW,HW:my-HW], sx-2*HW, sy-2*HW, N)
        MapL = MapL*2
        MapR = scaleMap(MapR[HW:mx-HW,HW:my-HW], sx-2*HW, sy-2*HW, N)
        MapR = MapR*2

        
    MapL = computeMap(L, R, N, M, scale, MapL)
    MapR = computeMap(R, L, N, M, scale, MapR)

#    for i in range(sx):
#        for j in range(sy):
#            if MapR[i,j + int(MapL[i,j])]+MapL[i,j] !=0:
#                MapL[i,j] = 0
#                
#    for i in range(sx):
#        for j in range(sy):
#            if MapL[i,j + int(MapR[i,j])]+MapR[i,j] !=0:
#                MapR[i,j] = 0
                
    return (MapL, MapR)
    

def plotFig(Map, N):
    HW = int((N-1)/2)
    plt.figure()
    (sx,sy) = Map.shape
    plt.imshow(Map[HW:sx-HW,HW:sy-HW],cmap = 'gray')
    plt.colorbar()
    plt.show()


# Recusively call itself for lower lever Map
def getDisparityMap(L, R, N, M, scale):
    

    (sx,sy) = scaleIm(L, 1/8, N).shape
    MapL = np.zeros((sx,sy)).astype(int)
    MapR = MapL

    (MapL, MapR) = pyramidLevel(L,R, MapL, MapR, N, M, 1/8, True)
    minL = MapL.min()
    minR = MapR.min()
    MapL = cv2.medianBlur(np.uint8(MapL-minL),3)+minL
    MapR = cv2.medianBlur(np.uint8(MapR-minR),3)+minR
    plotFig(MapL,N)
    plotFig(MapR,N)
    
    (MapL, MapR) = pyramidLevel(L,R, MapL, MapR, N, M, 1/4, False)
    minL = MapL.min()
    minR = MapR.min()
    MapL = cv2.medianBlur(np.uint8(MapL-minL),3)+minL
    MapR = cv2.medianBlur(np.uint8(MapR-minR),3)+minR
    plotFig(MapL,N)
    plotFig(MapR,N)
    
    (MapL, MapR) = pyramidLevel(L,R, MapL, MapR, N, M, 1/2, False)
    minL = MapL.min()
    minR = MapR.min()
    MapL = cv2.medianBlur(np.uint8(MapL-minL),5)+minL
    MapR = cv2.medianBlur(np.uint8(MapR-minR),5)+minR
    plotFig(MapL,N)
    plotFig(MapR,N)
    
    (MapL, MapR) = pyramidLevel(L,R, MapL, MapR, N, M, 1, False)
    minL = MapL.min()
    minR = MapR.min()
    MapL = cv2.medianBlur(np.uint8(MapL-minL),5)+minL
    MapR = cv2.medianBlur(np.uint8(MapR-minR),5)+minR
    plotFig(MapL,N)
    plotFig(MapR,N)

    return (MapL, MapR)
    

def main():
    N = 7 #Patch size
    M = 3 #Serach readius
    (L, R) = loadImages()
    (MapL, MapR) = getDisparityMap(R, L, N, M, 1)



if __name__ == "__main__":
    main()


