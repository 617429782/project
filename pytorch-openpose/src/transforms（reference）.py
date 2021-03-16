import torchvision
import random
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import numpy as np
import numbers
import cv2

"""
crop（裁剪）、scale（缩放）：填补、截掉原始图片
resize：改变图片分辨率

后期可以扩充一下，比如PCA
"""

# 随机裁剪，定义时输入size。
# 输入图片集合
class GroupRandomCrop(object):
    def __init__(self, size=224):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):

        w, h = img_group[0].size
        th, tw = self.size

        out_images = list()

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for img in img_group:
            # assert(img.size[0] == w and img.size[1] == h)
            if w == tw and h == th:
                out_images.append(img)
            else:
                out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return out_images

# 随机水平翻转，概率是0.5
class GroupRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, img_group, is_flow=False):
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            if self.is_flow:
                for i in range(0, len(ret), 2):
                    ret[i] = ImageOps.invert(ret[i])  # invert（颠倒、倒置） flow pixel values when flipping
            return ret
        else:
            return img_group

# 角裁剪（corner cropping）
# 仅从图片的边角或中心提取区域，来避免默认关注图片的中心。
class GroupCornerCrop(object):
    def __init__(self, crop_size=224):
        self.worker = torchvision.transforms.FiveCrop(crop_size)

    # idx取值范围是0~4，表示四角和中心。一次只返回一个（四角之一或中心）
    def __call__(self, img_group, idx):
        result = list()
        for img in img_group:
            img = self.worker(img)
            result.append(img[idx])
        return result

# 尺度抖动（scale jittering）
class GroupScaleJittering(object):
    """
    crop size是固定的。如224×224
    image size是随机可变的。如image的短边随机从[480,320,256,224](我瞎填的)选择，长边按比例缩放
    然后随机偏移裁剪个224×224的图像区域。
    """
    def __init__(self, crop_size=224):
        self.scales = [334,292,256,224]
        self.groupRandomCrop = GroupRandomCrop(size=crop_size)
        self.idx = random.randint(0, 3)
        print("idx:{}".format(self.idx))

    def __call__(self, img_group):
        img_group = self.scale_jittering(img_group, self.scales[self.idx])
        return self.groupRandomCrop(img_group)

    def scale_jittering(self, img_group, scale):
        worker = torchvision.transforms.Scale(scale)
        img_group = [worker(img) for img in img_group]
        return img_group



if __name__ == "__main__":
    img1 = Image.open("/home/jlm/pytorch-openpose/images/ski.jpg")
    print("img1.size:{}".format(img1.size))

    worker2 = torchvision.transforms.Resize((256,340), interpolation=2)
    img2 = worker2(img1)
    print("img2.size:{}".format(img2.size))

    worker3 = torchvision.transforms.RandomCrop(224)
    img3 = worker3(img1)
    print("img3.size:{}".format(img3.size))

    img4 = worker3(img2)
    print("img4.size:{}".format(img4.size))

    img3 = [img] * 3
    randomcrop = GroupScaleJittering()
    traimg = randomcrop(img3)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.imshow(traimg[0])
    plt.show()

