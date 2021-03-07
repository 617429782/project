import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import torch

from src import model
from src import util
from src.body import Body
from src.hand import Hand

body_estimation = Body('model/body_pose_model.pth')
# hand_estimation = Hand('model/hand_pose_model.pth')

#print(f"Torch device: {torch.cuda.get_device_name()}")

cap = cv2.VideoCapture('/home/jlm/pytorch-openpose/data/ut-interaction_dataset/44_8_4.avi')    # cv2.VideoCapture()的参数为0：打开笔记本的内置摄像头；参数为路径：即打开指定视频
cap.set(3, 640)     # 第一个参数表示所要设置的视频的参数，例如：编号3表示帧的宽度、编号4表示帧的高度
cap.set(4, 480)     # 所以这两句表示将视频流的帧的分辨率(宽，高)设置成(640，480)
count = 0
while True:
    count = count + 1
    ret, oriImg = cap.read()    # ret,frame = cap.read()按帧读取视频。如果读取帧正确ret为true，若文件读取到结尾，ret为false。frame是每一帧的图像，是个三维矩阵。
    if isinstance(oriImg, torch.Tensor):
        print("Yes")
    else:
        print("No")
    candidate, subset = body_estimation(oriImg)
    canvas = copy.deepcopy(oriImg)
    canvas = util.draw_bodypose(canvas, candidate, subset)
    print('canvas.shape')
    print(canvas.shape)
    '''
    # detect hand
    hands_list = util.handDetect(candidate, subset, oriImg)

    all_hand_peaks = []
    for x, y, w, is_left in hands_list:
        peaks = hand_estimation(oriImg[y:y+w, x:x+w, :])
        peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
        peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
        all_hand_peaks.append(peaks)

    canvas = util.draw_handpose(canvas, all_hand_peaks)
    '''
    cv2.imshow('demo', canvas)    # 一个窗口用以显示原视频。第一个参数：窗口名称； 第二个参数：要展示的图像
    plt.savefig('result%d.jpg' % count)
    if cv2.waitKey(1) & 0xFF == ord('q'):    # waitKey()表示等待键盘键入任意值。参数为1：延时1ms切换到下一帧图像；参数为0：只显示当前帧，相当于视频暂停
        break

cap.release()
cv2.destroyAllWindows()    # 销毁我们创建的所有窗口



