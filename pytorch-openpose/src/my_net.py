import cv2
import numpy as np
import math
import time
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib
import torch
from torchvision import transforms

from src import util
from src.model import bodypose_model

class Body(object):
    def __init__(self, model_path):
        self.model = bodypose_model()                                    # 构造网络模型
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print("cuda")
        model_dict = util.transfer(self.model, torch.load(model_path))    # 得到pytorch格式的model_weight
        self.model.load_state_dict(model_dict)
        self.model.eval()

    def __call__(self, oriImg):
        # scale_search = [0.5, 1.0, 1.5, 2.0]
        # oriImg.shape[0]为高;oriImg.shape[1]为宽
        print("oriImg.shape: ")
        print(oriImg.shape)
        scale_search = [0.5]
        boxsize = 368
        stride = 8
        padValue = 128
        thre1 = 0.1
        thre2 = 0.05
        multiplier = [x * boxsize / oriImg.shape[0] for x in scale_search]  # multiplier：乘数，倍数。看清楚哦，这个multiplier变量是个列表哦
        heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))   # 猜测这个变量是存放confidence maps
        paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))       # 猜测这个变量是存放part affinity fields

        for m in range(len(multiplier)):
            scale = multiplier[m]
            imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)   # 调整图像大小
            imageToTest_padded, pad = util.padRightDownCorner(imageToTest, stride, padValue)    # 将调整大小后的图像进行pad，并返回pad后图像和相应pad
            im = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 256 - 0.5    # 改变索引顺序
            im = np.ascontiguousarray(im)    # 将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快

            data = torch.from_numpy(im).float()
            if torch.cuda.is_available():
                data = data.cuda()
            # data = data.permute([2, 0, 1]).unsqueeze(0).float()
            with torch.no_grad():
                Mconv7_stage6_L1, Mconv7_stage6_L2 = self.model(data)    # model(data)，这里就是在调用网络的forward的啦
                print("Mconv7_stage6_L1 : ")
                print(Mconv7_stage6_L1.shape)
                print("Mconv7_stage6_L2 : ")
                print(Mconv7_stage6_L2.shape)
            Mconv7_stage6_L1 = Mconv7_stage6_L1.cpu().numpy()
            Mconv7_stage6_L2 = Mconv7_stage6_L2.cpu().numpy()
        return candidate, subset

