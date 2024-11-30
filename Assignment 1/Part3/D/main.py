import numpy as np
from math import exp, sqrt, pi, log
from PIL import Image
import cv2
import math

def ApplyPadding(image_array, a):
    h, w = image_array.shape
    new_h = h + 2 * a
    new_w = w + 2 * a
    padded_image = np.zeros((new_h, new_w), dtype=image_array.dtype)
    
    # Place the original image in the center of the padded image
    padded_image[a:h+a, a:w+a] = image_array
    
    # Reflect the top and bottom edges
    padded_image[:a, a:w+a] = np.flip(image_array[:a, :], axis=0)
    padded_image[h+a:, a:w+a] = np.flip(image_array[-a:, :], axis=0)
    
    # Reflect the left and right edges
    padded_image[a:h+a, :a] = np.flip(image_array[:, :a], axis=1)
    padded_image[a:h+a, w+a:] = np.flip(image_array[:, -a:], axis=1)
    
    # Reflect the corners
    padded_image[:a, :a] = np.flip(np.flip(image_array[:a, :a], axis=0), axis=1)
    padded_image[:a, w+a:] = np.flip(np.flip(image_array[:a, -a:], axis=0), axis=1)
    padded_image[h+a:, :a] = np.flip(np.flip(image_array[-a:, :a], axis=0), axis=1)
    padded_image[h+a:, w+a:] = np.flip(np.flip(image_array[-a:, -a:], axis=0), axis=1)
    
    return padded_image

def BilateralFilter(image, sigma_s, sigma_r): 
    a=math.ceil(3*sigma_s)
    PaddedImage = ApplyPadding(image,a)
    KernalSize = (2*a+1)
    height, width = image.shape
    FilteredImage = np.zeros_like(image)
    
    # G_s
    G_s = np.zeros((KernalSize, KernalSize))
    for i in range(KernalSize):
        for j in range(KernalSize):
            x = i - KernalSize // 2
            y = j - KernalSize // 2
            G_s[i, j] = math.exp(-(x**2 + y**2) / (2 * sigma_s**2))

    HalfKernal=KernalSize // 2
    for i in range(height):
        print("i:", i)
        for j in range(width):

            SubMatric = PaddedImage[a+i-HalfKernal:a+i+HalfKernal+1, a+j-HalfKernal:a+j+HalfKernal+1] - image[i, j]
            G_r = np.exp(-(SubMatric**2) / (2 * sigma_r**2))
            weight = G_s * G_r
            SumWeight= np.sum(weight)
            FilteredImage[i, j] = np.sum(weight * image[i, j])/SumWeight

    return FilteredImage.astype(np.uint8)

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


def DecomposeWithBF(ImagePath, SigmaSpace, SigmaColor):

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

    LowContrast = BilateralFilter(Image2D, SigmaSpace, SigmaColor)
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


DecomposeWithBF(ImagePath = "memorial.hdr", SigmaColor=5, SigmaSpace=3)

