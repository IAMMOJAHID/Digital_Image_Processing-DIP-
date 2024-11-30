import numpy as np
from math import exp, sqrt, pi
from PIL import Image

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

def GaussianFilterWithSeparableKernal(ImagePath, sigma):

    image = np.array(Image.open(ImagePath))
    h,w = image.shape[:2]
    Image2D = np.zeros((h, w), dtype=image.dtype)

    for i in range(h):
        for j in range(w):
            Image2D[i,j] = np.mean(image[i,j])

    KernelSize = int(2 * round(3 * sigma) + 1)
    GaussianKernel = Kernel1D(sigma, KernelSize)

    FilteredImage = Apply1DFilter(Image2D, GaussianKernel)
    FilteredImage = Apply1DFilter(FilteredImage.T, GaussianKernel).T

    FilteredImage = np.clip(FilteredImage, 0, 255).astype(np.uint8)
    FilteredImage = Image.fromarray(FilteredImage)
    FilteredImage.save('Sigma5.png')
    print("Filtered image saved")


GaussianFilterWithSeparableKernal('flower.png', 5.0)

