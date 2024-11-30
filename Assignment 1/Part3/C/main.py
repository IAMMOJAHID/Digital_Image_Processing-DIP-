import numpy as np
from PIL import Image
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

    # Apply the filter
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

def main():
    InputImage = np.array(Image.open('flower.png').convert('L'))
    SigmaR = 20
    SigmaS = 20

    Output= BilateralFilter(InputImage, SigmaR, SigmaS)
    Image.fromarray(Output.astype(np.uint8)).save('BilateralFiltering2.png')

if __name__ == "__main__":
    main()
