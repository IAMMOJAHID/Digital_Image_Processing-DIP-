from PIL import Image
import numpy as np
import cv2
import math

# PartA Answer
def MaxIntensity(Image):
    h, w = Image.shape[:2]

    MaxIntensity = -2**63
    for i in range(h):
        for j in range(w):
            if(Image[i, j]>MaxIntensity):
                MaxIntensity = Image[i, j]

    return MaxIntensity

def MinIntensity(Image):
    h, w = Image.shape[:2]

    MinIntensity = 2**63
    for i in range(h):
        for j in range(w):
            if(Image[i, j]<MinIntensity):
                MinIntensity = Image[i, j]

    return MinIntensity

def MapMaxAndMinIntensity(Path):
    InputImage = cv2.imread(Path, cv2.IMREAD_ANYDEPTH)
    h, w = InputImage.shape[:2]

    Image2D = np.zeros((h, w), dtype=InputImage.dtype)

    for i in range(h):
        for j in range(w):
            Image2D[i,j] = np.mean(InputImage[i,j])

    MaxInt=MaxIntensity(Image2D)
    MinInt=MinIntensity(Image2D)
    print(f'Max intensity is {MaxInt} & Min is {MinInt}')


    # Map Max and Min intensity [0,255]
    MaxMap = np.zeros((h, w), dtype=InputImage.dtype)
    MinMap = np.zeros((h, w), dtype=InputImage.dtype)

    for i in range(h):
        for j in range(w):
            MaxMap[i,j]= int(max((255.0/MaxInt)*(Image2D[i,j]),0))
            MinMap[i,j]= int(min((1.0/MinInt)*(Image2D[i,j]), 255))

    MaxImg = Image.fromarray(MaxMap).convert("L")
    MaxImg.save('MaxMapImage.png')
    MinImg = Image.fromarray(MinMap).convert("L")
    MinImg.save('MinMapImage.png')

MapMaxAndMinIntensity("memorial.hdr")


# PartB Answer

def LogTransformation(Path):
    InputImage = cv2.imread(Path, cv2.IMREAD_ANYDEPTH)
    h, w = InputImage.shape[:2]

    Image2D = np.zeros((h, w), dtype=InputImage.dtype)

    for i in range(h):
        for j in range(w):
            Image2D[i,j] = np.mean(InputImage[i,j])

    MaxInt=MaxIntensity(Image2D)
    MinInt=MinIntensity(Image2D)
    # print(f'Max intensity is {MaxInt} & Min is {MinInt}')
    a = ( 255.0/(math.log(MaxInt/MinInt)))
    b = -a*math.log(MinInt)

    # Map Max and Min intensity [0,255]
    FinalImage = np.zeros((h, w), dtype=InputImage.dtype)


    for i in range(h):
        for j in range(w):
            FinalImage[i,j]= int(a*math.log(Image2D[i,j])+b)

    SaveImage = Image.fromarray(FinalImage).convert("L")
    SaveImage.save('PartBAnswer.png')


LogTransformation("memorial.hdr")


# PartC Answer

def UndoLogTransformation(Path):
    InputImage = cv2.imread(Path, cv2.IMREAD_ANYDEPTH)
    h, w = InputImage.shape[:2]

    Image2D = np.zeros((h, w), dtype=InputImage.dtype)

    for i in range(h):
        for j in range(w):
            Image2D[i,j] = np.mean(InputImage[i,j])

    MaxInt=MaxIntensity(Image2D)
    MinInt=MinIntensity(Image2D)
    # print(f'Max intensity is {MaxInt} & Min is {MinInt}')
    a = ( math.log(255)/(math.log(MaxInt/MinInt)))
    b = -a*math.log(MinInt)

    # Map Max and Min intensity [0,255]
    FinalImage = np.zeros((h, w), dtype=InputImage.dtype)


    for i in range(h):
        for j in range(w):
            FinalImage[i,j]= int( math.exp(a*math.log(Image2D[i,j])+b))

    SaveImage = Image.fromarray(FinalImage).convert("L")
    SaveImage.save('PartCAnswer.png')


UndoLogTransformation("memorial.hdr")

# PartD Answer

def HistogramEqualisation(Path):
    InputImage = cv2.imread(Path, cv2.IMREAD_ANYDEPTH)
    h, w = InputImage.shape[:2]

    Image2D = np.zeros((h, w), dtype=InputImage.dtype)

    for i in range(h):
        for j in range(w):
            Image2D[i,j] = math.log(np.mean(InputImage[i,j]))

    IntensityCount = [0] * 256 
    for i in range(h):
        for j in range(w):
            IntensityCount[int(Image2D[i, j])] += 1

    for i in range(1,256,1):
        IntensityCount[i]+=IntensityCount[i-1]
    
    FinalImage = np.zeros((h, w), dtype=InputImage.dtype)

    for i in range(h):
        for j in range(w):
            FinalImage[i,j]= int((255)*( IntensityCount[int(Image2D[i,j])]/(h*w) ))

    SaveImage = Image.fromarray(FinalImage).convert("L")
    SaveImage.save('PartDAnswerLogTransform.png')

HistogramEqualisation("memorial.hdr")

# Part E Answer

def HistogramMatching(SourcePath, referencePath):
    InputImage = cv2.imread(SourcePath, cv2.IMREAD_ANYDEPTH)
    h, w = InputImage.shape[:2]
    InputImageRef = cv2.imread(referencePath, cv2.IMREAD_ANYDEPTH)
    hr, wr = InputImageRef.shape[:2]

    Image2D = np.zeros((h, w), dtype=InputImage.dtype)
    ImageRef = np.zeros((hr, wr), dtype=InputImageRef.dtype)

    for i in range(h):
        for j in range(w):
            Image2D[i,j] = np.mean(InputImage[i,j])

    for i in range(hr):
        for j in range(wr):
            ImageRef[i,j] = np.mean(InputImageRef[i,j])

    IntensityCount = [0] * 256 
    for i in range(h):
        for j in range(w):
            IntensityCount[int(Image2D[i, j])] += 1
    
    IntensityCountRef = [0] * 256 
    for i in range(hr):
        for j in range(wr):
            IntensityCountRef[int(ImageRef[i, j])] += 1

    for i in range(1,256,1):
        IntensityCount[i]+=IntensityCount[i-1]

    for i in range(1,256,1):
        IntensityCountRef[i]+=IntensityCountRef[i-1]

    
    FinalImage = np.zeros((h, w), dtype=InputImage.dtype)

    for i in range(h):
        for j in range(w):
            y = (255)*( IntensityCount[int(Image2D[i,j])]/(h*w) )
            y = (hr*wr)*(y/255)
            FinalImage[i,j]=0
            for k in range(0,256,1):
                if(IntensityCountRef[k]<=y):
                    FinalImage[i,j]=k
                else: 
                    break


    SaveImage = Image.fromarray(FinalImage).convert("L")
    SaveImage.save('PartEAnswerHistogramMatching.png')

HistogramMatching(SourcePath="memorial.hdr", referencePath='children-7934514.jpg')

