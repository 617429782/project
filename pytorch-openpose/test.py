import cv2

if __name__ == '__main__':
    video_path = "/home/jlm/pytorch-openpose/data/ut-interaction_dataset/0_1_4.avi"
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
    target_frame_count = 15  # 每个视频等间隔提取n帧
    gap = int(frame_count / target_frame_count)
    count = frame_count - frame_count % target_frame_count
    new_video = []
    idx = 0
    while idx < count:
        # 这里就可以做一些操作了：显示截取的帧图片、保存截取帧到本地等
        # (240, 180)、(480, 256)、(224, 224)
        # cv2对图像的处理是用x、y轴表示的，坐标原点在图片右上角，x轴方向向左，y轴方向向下
        frame = video[idx]
        cv2.imwrite("{}.jpg".format(idx), frame)
        idx += gap
