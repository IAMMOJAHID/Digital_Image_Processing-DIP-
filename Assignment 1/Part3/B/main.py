import numpy as np
from math import exp, sqrt, pi, log
from PIL import Image
import cv2

def Kernel1D(Sigma, KernelSize):
    kernel = np.zeros(KernelSize)
    center = KernelSize // 2
    SumVal = 0.0
    for i in range(0,KernelSize,1):
        x = i - center
        kernel[i] = exp(-0.5 * (x**2) / (Sigma**2))
        SumVal += kernel[i]
    kernel /= SumVal
    return kernel

def Apply1DFilter(InputImage, Kernel):

    h, w = InputImage.shape[:2]
    padding = len(Kernel) // 2
    Answer = np.zeros_like(InputImage)

    for i in range(0,h,1):
        print("i:",i)
        for j in range(0,w,1):
            SumVal = 0.0
            for k in range(-padding, padding + 1):
                if 0 <= j + k < w:
                    SumVal += InputImage[i, j + k]*Kernel[k + padding]
            Answer[i, j] = SumVal

    return Answer

def HistogramEqualisation(InputImage):
    h, w = InputImage.shape[:2]

    IntensityCount = [0] * 256 
    for i in range(h):
        for j in range(w):
            IntensityCount[int(InputImage[i, j])] += 1

    for i in range(1,256,1):
        IntensityCount[i]+=IntensityCount[i-1]
    
    FinalImage = np.zeros((h, w), dtype=InputImage.dtype)

    for i in range(h):
        for j in range(w):
            FinalImage[i,j]= int((255)*( IntensityCount[int(InputImage[i,j])]/(h*w) ))


    return FinalImage

def LowPassAndHighPassFilter(ImagePath, sigma):

    InputImage = cv2.imread(ImagePath, cv2.IMREAD_ANYDEPTH)
    h, w = InputImage.shape[:2]
    Image2D = np.zeros((h, w), dtype=InputImage.dtype)

    for i in range(h):
        for j in range(w):
            Image2D[i,j] = np.mean(InputImage[i,j])

    Image.fromarray(Image2D.astype(np.uint8)).save('BaseImage.png')

    for i in range(h):
        for j in range(w):
            Image2D[i,j] = log(Image2D[i,j])
    
    Image.fromarray(Image2D.astype(np.uint8)).save('BaseLogImage.png')

    KernelSize = int(2 * round(3 * sigma) + 1)
    GaussianKernel = Kernel1D(sigma, KernelSize)

    LowContrast = Apply1DFilter(Image2D, GaussianKernel)
    LowContrast = Apply1DFilter(LowContrast.T, GaussianKernel).T
    Image.fromarray(LowContrast.astype(np.uint8)).save('LowContrast.png')

    # HighContrast = np.clip(Image2D-LowContrast, 0, 255).astype(np.uint8)
    HighContrast = Image2D-LowContrast
    Image.fromarray(HighContrast.astype(np.uint8)).save('HighContrast.png')

    # MaxInt=MaxIntensity(LowContrast)
    ContrastReduction = HistogramEqualisation(LowContrast)
    Image.fromarray(ContrastReduction.astype(np.uint8)).save('ContrastReduction.png')
    print("Constrast Reduction done")

    CombineLowHighContrast= ContrastReduction+HighContrast
    Image.fromarray(CombineLowHighContrast.astype(np.uint8)).save('CombineLowHighContrast.png')
    print("Combination of Low and High Contrast done")

    # Undo Log
    UndoLog = np.zeros((h, w), dtype=InputImage.dtype)
    for i in range(h):
        for j in range(w):
            UndoLog[i,j] = exp(CombineLowHighContrast[i,j])

    Image.fromarray(UndoLog.astype(np.uint8)).save('UndoLog.png')
    print("UndoLog Saved")


LowPassAndHighPassFilter(ImagePath = "memorial.hdr", sigma=3.0)

