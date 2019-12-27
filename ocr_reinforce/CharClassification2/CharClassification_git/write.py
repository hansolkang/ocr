import os
import cv2
path = "./train"
save_path = "./train_binary/"
file_lists=  os.listdir(path)
i=-1
for root,dirs,files in os.walk(path):
    for file in files:
        filepath = os.path.join(root,file)
        img = cv2.imread(filepath,0)
        # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 3)
        imgThres = 255 - img
        mdir = save_path + file_lists[i]
        if not os.path.isdir(mdir):
            os.mkdir(mdir)
        cv2.imwrite(str(mdir) + '/' + file,img)
    i +=1
# for fname in file_lists:
#     full_name = os.path.join(path,fname)
#
#     print(full_name)