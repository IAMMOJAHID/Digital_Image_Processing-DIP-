import numpy as np
from PIL import Image
import math
import cv2

def RGB_To_HSI(R, G, B):

    R, G, B = R / 255.0, G / 255.0, B / 255.0
    
    I = (R + G + B) / 3.0
    
    min_rgb = np.minimum(np.minimum(R, G), B)
    S = 1 - min_rgb / I
    
    numerator = 0.5 * ((R - G) + (R - B))
    denominator = np.sqrt((R - G) ** 2 + (R - B) * (G - B))
    theta = np.arccos(numerator / (denominator + 1e-7))
    
    H = np.where(B > G, 2 * np.pi - theta, theta)
    H = H / (2 * np.pi) 
    
    return H, S, I

def SingleHSI_To_RGB(H, S, I):
    H = H * 2 * np.pi
    if H < 2 * np.pi / 3:
        B = I * (1 - S)
        R = I * (1 + S * np.cos(H) / np.cos(np.pi / 3 - H))
        G = 3 * I - (R + B)
    elif H < 4 * np.pi / 3:
        H = H - 2 * np.pi / 3
        R = I * (1 - S)
        G = I * (1 + S * np.cos(H) / np.cos(np.pi / 3 - H))
        B = 3 * I - (R + G)
    else:
        H = H - 4 * np.pi / 3
        G = I * (1 - S)
        B = I * (1 + S * np.cos(H) / np.cos(np.pi / 3 - H))
        R = 3 * I - (G + B)
    
    return np.clip(R, 0, 1), np.clip(G, 0, 1), np.clip(B, 0, 1)

def HSI_To_RGB(H, S, I):

    R, G, B = np.zeros(H.shape), np.zeros(H.shape), np.zeros(H.shape)

    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            R[i,j], G[i,j], B[i,j] = SingleHSI_To_RGB(H[i,j], S[i,j], I[i,j])

    R, G, B = np.clip(R, 0, 1), np.clip(G, 0, 1), np.clip(B, 0, 1)
    return (R * 255).astype(np.uint8), (G * 255).astype(np.uint8), (B * 255).astype(np.uint8)

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

def ProcessHDR(ImagePath):
    image = cv2.imread(ImagePath, cv2.IMREAD_ANYDEPTH)
    # image = Image.open(ImagePath)
    image_np = np.array(image)
    
    R, G, B = image_np[:, :, 0], image_np[:, :, 1], image_np[:, :, 2]
    H, S, I = RGB_To_HSI(R, G, B)

    IComponent=BilateralFilter(I, sigma_s=5, sigma_r=3)
    RNow, GNow, BNow = HSI_To_RGB(H, S, IComponent)
    rgb_image = np.stack((RNow, GNow, BNow), axis=-1)
    Image.fromarray(rgb_image).save('CR_I_Component.png')

    RComponent=BilateralFilter(R, sigma_s=5, sigma_r=3)
    rgb_image = np.stack((RComponent, G, B), axis=-1)
    Image.fromarray(rgb_image.astype(np.uint8)).save('CR_R_Component.png')

    GComponent=BilateralFilter(G, sigma_s=5, sigma_r=3)
    rgb_image = np.stack((R, GComponent, B), axis=-1)
    Image.fromarray(rgb_image.astype(np.uint8)).save('CR_G_Component.png')

    BComponent=BilateralFilter(B, sigma_s=5, sigma_r=3)
    rgb_image = np.stack((R, G, BComponent), axis=-1)
    Image.fromarray(rgb_image.astype(np.uint8)).save('CR_B_Component.png')

ProcessHDR(ImagePath='memorial.hdr')