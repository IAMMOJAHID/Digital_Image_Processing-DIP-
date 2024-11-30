import numpy as np
from PIL import Image

def HSI_To_RGB(H, S, I):

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

def GenerateHSIImage(H_fixed=None, S_fixed=None, I_fixed=None, size=500):

    image = np.zeros((size, size, 3))
    for i in range(0, size, 1):
        for j in range(0, size, 1):
            if H_fixed is not None:
                H = H_fixed
                S = j / (size - 1)
                I = i / (size - 1)
            elif S_fixed is not None:
                H = j / (size - 1)
                S = S_fixed
                I = i / (size - 1)
            elif I_fixed is not None:
                H = j / (size - 1)
                S = i / (size - 1)
                I = I_fixed
            
            R, G, B = HSI_To_RGB(H, S, I)
            
            # Is color out of the RGB gamut?
            if np.any([R < 0, R > 1, G < 0, G > 1, B < 0, B > 1]):
                # Red: Error color
                image[i, j] = [1, 0, 0]
            else:
                image[i, j] = [R, G, B]
    
    return (image * 255).astype(np.uint8)

def main():

    H_const_image = GenerateHSIImage(H_fixed=0.5)
    S_const_image = GenerateHSIImage(S_fixed=0.5)
    I_const_image = GenerateHSIImage(I_fixed=0.5)
    
    Image.fromarray(H_const_image).save('H_const_image.png')
    Image.fromarray(S_const_image).save('S_const_image.png')
    Image.fromarray(I_const_image).save('I_const_image.png')

if __name__ == "__main__":
    main()
