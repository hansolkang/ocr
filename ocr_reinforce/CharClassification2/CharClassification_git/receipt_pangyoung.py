import numpy as np
import cv2
# import mp
# from mp import *
import math

img = cv2.imread('./receipt_pangyoung/75.jpg',0)
# img = cv2.imread('./icdar2019_task/task1_train/X00016469612.jpg',0)

# img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 3)
f = open('./receipt_pangyoung/75.txt')
# f = open('./icdar2019_task/task1_train/X00016469612.txt')
lines = f.readlines()
def preprocessing(img, minArea):
    (_, imgThres) = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    imgThres = 255 - imgThres

    (_, components, _) = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    res = []
    for c in components:
        # skip small word
        if cv2.contourArea(c) < minArea:
            continue
        currBox = cv2.boundingRect(c)
        (x, y, w, h) = currBox
        currImg = img[y:y + h, x:x + w]
        res.append((currBox, currImg))

    return sorted(res, key=lambda entry: entry[0][0])

for line in lines:
    split_line = line.split(',')
    row1 = int(split_line[6])
    row2 = int(split_line[2])
    # row1 = int(split_line[1])
    # row2 = int(split_line[7])
    row_width = row2-row1

    col1 = int(split_line[5])
    col2 = int(split_line[1])
    # col1 = int(split_line[0])
    # col2 = int(split_line[2])
    col_width = col2-col1
    subimg = img[row1:row2, col1:col2]
    cv2.imwrite("test2.jpg",subimg)
    imread_img = cv2.imread("test2.jpg",0)
    res = preprocessing(imread_img, minArea=10)
    for (j, w) in enumerate(res):
        (wordBox, wordImg) = w
        (x, y, w, h) = wordBox
        cv2.imwrite('test2.jpg', wordImg)

    char_plus_width = int(round(row_width * 0.45))
    second_start = 8
    count = 0
    for char_width in range(0,col_width,char_plus_width):
        character_img = subimg[0:row_width,second_start:char_width+char_plus_width]
        # cv2.resize(character_img,(32,32),cv2.INTER_LINEAR)
        cv2.imwrite("test.jpg", character_img)
        # imread_img = cv2.imread("test.jpg",0)

        # mp.predict("trained_weights_4epoch.h5", "test.jpg")
        # if (mp.max<=0.75):
        #     print("this didn't high than 0.7 : " , mp.max)
        #     while mp.max <=0.75:
        #         char_plus_width +=1
        #         count +=1
        #         character_img = subimg[0:row_width, second_start:char_width + char_plus_width]
        #         # gray = cv2.cvtColor(character_img,cv2.COLOR_RGB2GRAY)
        #         # ret, binary = cv2.threshold(character_img,127,255,cv2.THRESH_BINARY)
        #         # binary = cv2.bitwise_not(binary)
        #         # cv2.resize(character_img, (32, 32), cv2.INTER_LINEAR)
        #         cv2.imwrite("test.jpg", character_img)
        #         mp.predict("trained_weights_4epoch.h5", "test.jpg")
        #         if (count >= 40):
        #             break;
        count = 0
        second_start = char_width + char_plus_width
        cv2.imshow('modified2', character_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # cv2.imshow('modified', subimg)
    # cv2.imshow('modified', character_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
f.close()

