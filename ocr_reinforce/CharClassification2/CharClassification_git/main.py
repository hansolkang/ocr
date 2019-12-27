import os
import cv2
from WordSegmentation import wordSegmentation, prepareImg

def rectangle():
    imgFiles = os.listdir('../data/')
    for (i,f) in enumerate(imgFiles):
        print('Segmentin : %s'%f)
        img = prepareImg(cv2.imread('../data/%s' % f), 2000)
        # img = prepareImg(cv2.imread('../data/%s' % f, cv2.IMREAD_GRAYSCALE), 2000)
        print(img.shape[1])

        res = wordSegmentation(img, kernelSize=15, sigma=11, theta=7, minArea=10)

        if not os.path.exists('../out/%s' % f):
            os.mkdir('../out/%s' % f)

        print('Segmented into %d words' % len(res))
        for (j, w) in enumerate(res):
            (wordBox, wordImg) = w
            (x, y, w, h) = wordBox
            y = y-10
            h = h+2
            cv2.imwrite('../out/%s/%d.png' % (f, j), wordImg)
            cv2.rectangle(img, (x, y), (x + w, y + h), 0, thickness=2)

        cv2.imwrite('../out/%s/summary.png'%f,img)

if __name__ == '__main__':
    rectangle()