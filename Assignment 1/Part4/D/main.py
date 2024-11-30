from PIL import Image
import numpy as np

def GammaTransform(image, gamma):
    image = image / 255.0
    image = image ** gamma
    return (image * 255).astype(np.uint8)

def ApplyOnALL(R,G,B,I):

    imageR = np.array(Image.open(R))
    imageG = np.array(Image.open(G))
    imageB = np.array(Image.open(B))
    imageI = np.array(Image.open(I))
    gamma=1/2.2

    Image.fromarray(GammaTransform(imageR, gamma)).save('CR_R_Component.png')
    Image.fromarray(GammaTransform(imageG, gamma)).save('CR_G_Component.png')
    Image.fromarray(GammaTransform(imageB, gamma)).save('CR_B_Component.png')
    Image.fromarray(GammaTransform(imageI, gamma)).save('CR_I_Component.png')


ApplyOnALL("../C/CR_R_Component.png", "../C/CR_G_Component.png", "../C/CR_B_Component.png", "../C/CR_I_Component.png")