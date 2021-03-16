import numbers
import random
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps


# 效果测试
def drawImage(img1, img2):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.savefig("test.jpg")

def getSize(size):
    # 若输入单个数，则把图像裁剪成crop_size*crop_size的正方形
    # 否则，就按照相应的输入裁剪
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        size = size
    return size

# 缩放
def GroupScale(img_group, size=224):
    img1 = img_group[0]
    size = getSize(size)
    for img in img_group:
        cv2.resize(img, size, img, interpolation=cv2.INTER_AREA)
    img2 = img_group[0]
    drawImage(img1, img2)
    return img_group

# 目前只用上水平翻转、随机平移
# 水平翻转
def GroupHorizontalFlip(img_group, size=224):
    size = getSize(size)
    img1 = img_group[0]
    result = []
    for img in img_group:
        _img = cv2.flip(img, 1)
        #_img = cv2.resize(_img, size, interpolation=cv2.INTER_LINEAR)
        result.append(_img)
    img2 = result[0]
    drawImage(img1, img2)
    return result

# 随机平移
def GroupRandomTranslate(img_group, size=224):
    img1 = img_group[0]
    size = getSize(size)
    h, w = (img_group[0].shape)[:2]
    # 平移矩阵M：[[1,0,x],[0,1,y]]  :  x轴上平移x，y轴上平移y
    x = random.randint(-50, 50)
    y = random.randint(-30, 30)
    M = np.float32([[1, 0, x], [0, 1, y]])
    result = []
    for img in img_group:
        _img = cv2.warpAffine(img, M, (w, h))
        #_img = cv2.resize(_img, size, interpolation=cv2.INTER_LINEAR)
        result.append(_img)
    img2 = result[0]
    drawImage(img1, img2)
    return result

# 绘图看了一下，裁剪会导致丢掉很多信息，以后再考虑裁剪的事吧
# 随机裁剪
def GroupRandomCrop(img_group, crop_size=224):
    img1 = img_group[0]
    print(img1.shape)
    crop_size = getSize(crop_size)

    result = []

    h, w = (img_group[0].shape)[:2]
    print("h:{},w:{},cs0:{},cs1:{}".format(h, w, crop_size[0], crop_size[1]))
    x = random.randint(0, h - crop_size[0])
    y = random.randint(0, w - crop_size[1])

    for img in img_group:
        result.append(img[x:x+crop_size[0], y:y+crop_size[1]])
    img2 = result[0]
    drawImage(img1, img2)
    #return result

# 尺度抖动
def GroupScaleJittering(img_group, crop_size=224):
    img1 = img_group[0]
    h, w = img_group[0].shape
    dsth = random.randint(256, 480)
    scale = dsth / h
    dstw = scale * w
    for img in img_group:
        cv2.resize(img, (dsth,dstw), img, interpolation=cv2.INTER_AREA)
    img_group = GroupRandomCrop(img_group, crop_size)
    img2 = img_group[0]
    drawImage(img1, img2)
    return img_group

def PCA_Jittering(img_path):
    img_num = 1

    for i in range(img_num):
        img = Image.open(img_path)
        img1 = img

        img = np.asanyarray(img, dtype = 'float32')

        img = img / 255.0
        img_size = img.size // 3    #转换为单通道
        img1 = img.reshape(img_size, 3)

        img1 = np.transpose(img1)   #转置
        img_cov = np.cov([img1[0], img1[1], img1[2]])    #协方差矩阵
        lamda, p = np.linalg.eig(img_cov)     #得到上述协方差矩阵的特征向量和特征值

        #p是协方差矩阵的特征向量
        p = np.transpose(p)    #转置回去

        #生成高斯随机数********可以修改
        alpha1 = random.gauss(0,3)
        alpha2 = random.gauss(0,3)
        alpha3 = random.gauss(0,3)

        #lamda是协方差矩阵的特征值
        v = np.transpose((alpha1*lamda[0], alpha2*lamda[1], alpha3*lamda[2]))     #转置

        #得到主成分
        add_num = np.dot(p,v)

        #在原图像的基础上加上主成分
        img2 = np.array([img[:,:,0]+add_num[0], img[:,:,1]+add_num[1], img[:,:,2]+add_num[2]])

        #现在是BGR，要转成RBG再进行保存
        img2 = np.swapaxes(img2,0,2)
        img2 = np.swapaxes(img2,0,1)

        drawImage(img1, img2)


if __name__ == "__main__":
    video_path = "/home/jlm/pytorch-openpose/data/test/0_11_4.avi"
    cap = cv2.VideoCapture(video_path)
    video = []
    for i in range(3):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (320, 256), interpolation=cv2.INTER_LINEAR)
        video.append(frame)
    GroupRandomCrop(video)


