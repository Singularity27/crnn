import cv2
import numpy as np
import os
from skimage import transform
import random

# 输入视频文件目录
mv_file_mv = 'H:/深度学习/新建文件夹 (5)/6/'
# 输出numpy文件目录
mv_file_np = 'H:/深度学习/crnn/traindata2/'

mv_name_list = os.listdir(mv_file_mv)

for m in range(len(mv_name_list)):
    mv_name = mv_file_mv + mv_name_list[m]
    input_movie = cv2.VideoCapture(mv_name)
    np_file = []
    for n in range(10):
        ret, frame = input_movie.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        res = transform.resize(image, (224, 224))
        np_file.append(res)
    np_name = mv_file_np + mv_name_list[m].replace('.mp4', "")
    np.save(np_name, np_file)
    print(mv_name_list[m])


# 文件检测
# import matplotlib.pyplot as plt
# npy = np.load(mv_file_np + '1.npy')
# for i in range(10):
#     plt.imshow(npy[i])
#     plt.show()
