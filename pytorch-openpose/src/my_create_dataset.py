from torch.utils.data import Dataset, DataLoader, random_split
import cv2
import numpy as np

"""### 自定义数据集 ###"""
class MyDataset(Dataset):
    """
    dataPath: txt文件夹路径
    """
    def __init__(self, dataPath, transform=None, target_transform=None):

        super(Dataset, self).__init__()
        videosPath = open(dataPath, 'r')   # dataPath是存放“视频路径 类别”形式信息的文件夹的路径
        videos = []
        for line in videosPath:
            line = line.rstrip()
            words = line.split()
            videos.append((words[0], int(words[1])))    # 路径 类别

        self.videos = videos
        self.transform = transform
        self.target_transform = target_transform
        self._video_num = len(self.videos)
        videosPath.close()
        # self.load_size = 128

    def __getitem__(self, index):
        video_path, label = self.videos[index]
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
        new_video = np.array(new_video)
        return new_video, label

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self._video_num