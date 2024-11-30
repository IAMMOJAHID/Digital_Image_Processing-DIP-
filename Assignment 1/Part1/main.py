from PIL import Image
import numpy as np
import cv2

# Part A

def FindCommonIntensity(Path):
    image = cv2.imread(Path, cv2.IMREAD_GRAYSCALE)
    h, w = image.shape

    IntensityCount = [0] * 256 
    for i in range(h):
        for j in range(w):
            IntensityCount[image[i, j]] += 1

    MostCommonIntensity = 0
    MaxCount = 0
    for i in range(256):
        if IntensityCount[i] > MaxCount:
            MaxCount = IntensityCount[i]
            MostCommonIntensity = i
    
    # print("MostCommon Intensity",MostCommonIntensity)
    return MostCommonIntensity

def BackgroundImage(image_path, save_path='Alpha.png'):

    Img= np.array(Image.open(image_path))
    Height, Width = Img.shape

    Answer = np.zeros((Height, Width), dtype=Img.dtype)
    r=FindCommonIntensity(image_path)
    t=10

    for i in range(Height):
        for j in range(Width):
            if( abs(Img[i, j]-r)<t):
                Answer[i,j] = 1
            else:
                Answer[i,j] = 0
    
    FinalImage = Image.fromarray(Answer)
    FinalImage.save(save_path)
    print(f'Background Image saved as {save_path}')

BackgroundImage('cse-logo.png')


# Part B
def NNInterpolation(Img, factor):
    OriginalHeight, OriginalWidth = Img.shape[:2]
    Height = OriginalHeight * factor
    Width = OriginalWidth * factor

    Answer = np.zeros((Height, Width), dtype=Img.dtype)

    for i in range(Height):
        for j in range(Width):
            Answer[i, j] = Img[int(i / factor), int(j / factor)]
    
    return Answer

def BLInterpolation(Img, factor):
    OriginalHeight, OriginalWidth = Img.shape[:2]
    Height = OriginalHeight * factor
    Width = OriginalWidth * factor

    Answer = np.zeros((Height, Width), dtype=Img.dtype)

    for i in range(Height):
        for j in range(Width):
            x = i / factor
            y = j / factor

            # Find the coordinates of the 4 surrounding pixels
            x1 = int(x)
            y1 = int(y)
            x2 = min(x1 + 1, OriginalHeight - 1)
            y2 = min(y1 + 1, OriginalWidth - 1)

            # Interpolation weights
            a = x - x1
            b = y - y1

            # Perform bilinear interpolation
            Answer[i, j] = ( (1 - a) * (1 - b) * Img[x1, y1] + a * (1 - b) * Img[x2, y1] + (1 - a) * b * Img[x1, y2] + a * b * Img[x2, y2] )

    return Answer

def ResizeImage(image_path, factor=3):

    Img = np.array(Image.open(image_path))

    NNImage = Image.fromarray(NNInterpolation(Img, factor))
    BIamge = Image.fromarray(BLInterpolation(Img, factor))

    NNImage.save('NNLogo.png')
    BIamge.save("BLLogo.png")

ResizeImage('cse-logo.png')


# Part C

def DrawLogo(BaseImage, Logo, AlphaLogo):

    BaseImg = np.array(Image.open(BaseImage).convert("L"))
    height, width= BaseImg.shape[:2]

    LogoImg = np.array(Image.open(Logo))
    x, y = LogoImg.shape[:2]

    AlphaImage= np.array(Image.open(AlphaLogo))

    Save_array = np.zeros((height, width), dtype=BaseImg.dtype)

    for i in range(height):
        for j in range(width):
            n,m = (height-i), (width-j)
            if(n<=x and m<=y):
                Save_array[i,j]= (1-AlphaImage[x-n, y-m])*LogoImg[x-n, y-m] + (AlphaImage[x-n, y-m])*np.mean(BaseImg[i,j])
            else:
                Save_array[i,j] = np.mean(BaseImg[i,j])

    
    FinalImg = Image.fromarray(Save_array)
    FinalImg.save('MyPhotoWithNNLogoWithLConvert.png')
    print("Image Saved")


DrawLogo('BaseImage.png', 'NNLogo.png', 'NNAlpha.png')
