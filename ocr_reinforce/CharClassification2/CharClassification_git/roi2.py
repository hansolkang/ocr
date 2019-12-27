import numpy as np
import cv2
import mp
from mp import *
import math
import os

path = "./icdar2019_task/task2_test"
path_text = "./icdar2019_task/task2_test_text_hyeckchan"
file_lists=  os.listdir(path)
file_lists_text = os.listdir(path_text)
file_lists.sort()
file_lists_text.sort()
global img
global file

# img = cv2.imread('./task1_train/X00016469619.jpg',0)
# img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 3)
# file = open('./out_text/text.txt', 'w')
# f = open('./task1_train/X00016469619.txt')
# lines = f.readlines()
def preprocessing(img, minArea):
    # (_, imgThres) = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # imgThres = 255 - imgThres
    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 3)
    imgThres = 255 - img


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
for root,dirs,files in os.walk(path):
    for file in files:
        index = 0
        filepath = os.path.join(root,file_lists[index])
        img = cv2.imread(filepath,0)
        write_file = open('./out_text/' + file_lists_text[index], 'w')
        f = open(path_text + '/' + file_lists_text[index])
        lines = f.readlines()

        for line in lines:
            split_line = line.split(',')
            row1 = int(split_line[1])
            row2 = int(split_line[7])
            row_width = row2-row1

            col1 = int(split_line[0])
            col2 = int(split_line[2])
            col_width = col2-col1
            subimg = img[row1:row2, col1:col2]
            cv2.imwrite("out/word.jpg",subimg)
            imread_img = cv2.imread("out/word.jpg",0)
            res = preprocessing(imread_img, minArea=10)

            char_plus_width = int(round(row_width * 0.45))
            first_start = 0
            second_start = 1
            count = 0
            word = []

        # for char_width in range(0,col_width,char_plus_width):
            for (j, w) in enumerate(res):
                (wordBox, wordImg) = w
                (x, y, w, h) = wordBox
                if wordImg is None:
                    break
                wordImg = cv2.resize(wordImg, (32, 32), cv2.INTER_LINEAR)
                cv2.imwrite('out/test%s.jpg'%count, wordImg)
                final_result = mp.predict("./check/trained_weights_3epoch.h5", "out/test%s.jpg"%count)
                res_plus = res[first_start][0][0] + res[first_start][0][2]

                word.append(final_result)
                if second_start == len(res):
                    write_file.write('%s' % final_result)
                    break

                if (res[second_start][0][0] - res_plus) > 5:
                    write_file.write('%s' % final_result)
                    write_file.write(' ')
                else:
                    write_file.write('%s' % final_result)

                second_start += 1
                first_start += 1
                count += 1

                print("test%s" %count)
            write_file.write('\n')
    index +=1
f.close()
file.close()
write_file.close()

