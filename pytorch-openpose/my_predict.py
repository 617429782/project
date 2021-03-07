import torch
from torchvision.transforms import transforms
from torch.autograd import Variable
import os
import cv2
import numpy as np
from src.my_body import my_Body
from src import my_model as mymodel
from src import my_global as mg

"""### 全局变量 ###"""
# 类名映射
class_map = {0:"hi", 1:"lei", 2:"ming", 3:"he", 4:"ji", 5:"ya"}

# 构造网络并初始化权值
body_estimation = my_Body('model/body_pose_model.pth')
state_path = "/home/jlm/pytorch-openpose/model/myRnnModel_26.model"    # 模型权值文件路径
checkpoint = torch.load(state_path)
rnn = mymodel.my_RNN(input_size=15, hidden_size=256, output_size=6)
rnn.load_state_dict(checkpoint)
rnn.to(mg.device)
rnn.eval()


def predict_video(video_path):
    print("Prediction in progress")
    cap = cv2.VideoCapture(video_path)  # cv2.VideoCapture()的参数为路径：即打开指定视频；参数为0：打开笔记本的内置摄像头
    video = []
    while (True):
        ret, frame = cap.read()
        if ret:
            video.append(frame)
            cv2.waitKey(0)  # waitKey()表示等待键盘键入任意值。参数为1：延时1ms切换到下一帧图像；参数为0：只显示当前帧，相当于视频暂停
        else:
            break
    cap.release()
    frame_count = len(video)  # 得到视频总帧数
    target_frame_count = 15  # 每个视频等间隔提取15帧
    gap = int(frame_count / target_frame_count)
    count = frame_count - frame_count % target_frame_count
    new_video = []
    idx = 0
    while idx < count:
        # 这里就可以做一些操作了：显示截取的帧图片、保存截取帧到本地等
        frame = video[idx]
        frame = cv2.resize(frame, (240, 180), interpolation=cv2.INTER_LINEAR)
        new_video.append(frame)
        idx += gap
    input = np.array(new_video)
    batch_size = 1
    length = len(input)
    h = len(input[0])
    w = len(input[0][0])
    c = len(input[0][0][0])
    intermediate_output = []  # 存放通过openpose得到的特征图。len:(batch_size*length); 元素size：n*h'*w'*c'
    for idx in range(len(input)):
        interout = body_estimation(input[idx])
        interout = np.array(torch.squeeze(interout).cpu())
        intermediate_output.append(interout)
    input = torch.from_numpy(np.array(intermediate_output)).to(mg.device)
    output = rnn(input)

    index = output.data.cpu().numpy().argmax()
    return index


if __name__ == "__main__":

    videopath = "/home/jlm/pytorch-openpose/data/ut-interaction_dataset/9_2_2.avi"
    if not os.path.exists(videopath):
        print("no video")
    else:
        # run prediction function and obtain prediccted class index
        index = predict_video(videopath)
        prediction = class_map[int(index)]
        print("Predicted Class : {}".format(prediction))