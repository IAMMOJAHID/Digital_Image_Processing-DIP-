import numpy as np
from PIL import Image

def RGB_To_HSI(R, G, B):

    R, G, B = R / 255.0, G / 255.0, B / 255.0
    
    I = (R + G + B) / 3.0
    min_rgb = np.minimum(np.minimum(R, G), B)
    S = 1 - min_rgb / I
    numerator = 0.5 * ((R - G) + (R - B))
    denominator = np.sqrt((R - G) ** 2 + (R - B) * (G - B))
    theta = np.arccos(numerator / (denominator + 1e-7))
    
    H = np.where(B > G, 2 * np.pi - theta, theta)
    H = H / (2 * np.pi)  # Normalize to [0, 1]
    
    return H, S, I

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


def main():
    image_path = 'flower.png'
    K = 1.5
    image = np.array(Image.open(image_path))

    # RGB scale up by K.
    ScaledImageK = np.clip(image * K, 0, 255).astype(np.uint8)
    Image.fromarray(ScaledImageK).save(f'rgb_{K}_scaled_image.png')

    # HSI scaled image
    R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    H, S, I = RGB_To_HSI(R, G, B)
    I_ScaledUp = np.clip(I * K, 0, 1)
    
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            R[i,j], G[i,j], B[i,j] = HSI_To_RGB(H[i,j], S[i,j], I_ScaledUp[i,j])

    HSI_ScaleUp = np.stack((R, G, B), axis=-1)
    Image.fromarray((255*HSI_ScaleUp).astype(np.uint8)).save(f'HSI_{K}_ScaleUp.png')


if __name__ == "__main__":
    main()
