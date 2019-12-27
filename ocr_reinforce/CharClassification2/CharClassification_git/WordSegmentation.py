import math
import cv2
import numpy as np

def wordSegmentation(img, kernelSize, sigma, theta, minArea):
    kernel = createKernel(kernelSize,sigma, theta)
    imgFiltered = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE).astype(np.uint8)
    (_, imgThres) = cv2.threshold(imgFiltered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    imgThres = 255 - imgThres

    (_, components, _) = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    res = []
    for c in components:
        # skip small word
        if cv2.contourArea(c) < minArea:
            continue
        currBox = cv2.boundingRect(c)
        (x,y,w,h) = currBox
        currImg = img[y:y+h, x:x+w]
        res.append((currBox, currImg))

    return sorted(res, key = lambda entry:entry[0][0])

def prepareImg(img, height):
    assert img.ndim in (2,3)
    if img.ndim ==3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h = img.shape[0]
    factor = height / h
    zoom2 = cv2.resize(img, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_CUBIC)
    return zoom2

def createKernel(kernelSize, sigma, theta):
    assert kernelSize % 2
    halfsize = kernelSize //2

    kernel = np.zeros([kernelSize, kernelSize])
    sigmaX = sigma
    sigmaY = sigma * theta

    for i in range(kernelSize):
        for j in range(kernelSize):
            x = i - halfsize
            y = j - halfsize

            expTerm = np.exp(-x**2 / (2 * sigmaX) - y**2 / (2 * sigmaY))
            xTerm = (x ** 2 - sigmaX ** 2) / (2 * math.pi * sigmaX ** 5 * sigmaY)
            yTerm = (y ** 2 - sigmaY ** 2) / (2 * math.pi * sigmaY ** 5 * sigmaX)

            kernel[i,j] = (xTerm + yTerm) * expTerm
        kernel = kernel / np.sum(kernel)
        return kernel