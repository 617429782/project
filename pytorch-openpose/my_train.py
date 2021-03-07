from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import numpy as np
import os
import torch
import torch.optim as optim
import torch.nn as nn
import src.my_create_dataset as dataset
from src.my_body import my_Body
from src import my_model as mymodel
from src import my_global as mg
from src.util import draw_accuracy_loss as draw_accuracy_loss

"""### 全局变量 ###"""
# 数据集
"""
transform_train = transforms.Compose([transforms.Resize(256),  # 重置图像分辨率
                                      transforms.RandomResizedCrop(224),  # 随机裁剪
                                      transforms.RandomHorizontalFlip(),  # 以概率p水平翻转
                                      transforms.RandomVerticalFlip(),  # 以概率p垂直翻转
                                      transforms.ToTensor(),])
transformation = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
transform = transforms.Compose([
    transforms.Resize(256),  # 重置图像分辨率
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
"""
dataPath = "/home/jlm/pytorch-openpose/data/ut-interaction_dataset/dataset.txt"
all_dataset = dataset.MyDataset(dataPath)

# 将数据集划分为trian、val、test三类（0.8，0.1，0.1） ps：测试能成后记得把util里自制的数据集划分删掉
train_size = int(0.8 * len(all_dataset))
val_size = int(0.1 * len(all_dataset))
test_size = len(all_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(all_dataset, [train_size, val_size, test_size])

# dataloader
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=1)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=1)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=1)

# 网络
body_estimation = my_Body('model/body_pose_model.pth')
rnn = mymodel.my_RNN(input_size=15, hidden_size=256, output_size=6)
rnn.to(mg.device)

# 损失函数、优化器（待定)
criterion = nn.CrossEntropyLoss().to(mg.device)
optimizer = optim.SGD(rnn.parameters(), lr=1e-3, momentum=0.7)


def save_models(epoch):
    modelPath = "/home/jlm/pytorch-openpose/model/myRnnModel_{}.model".format(epoch)
    torch.save(rnn.state_dict(), modelPath)
    print("Chekcpoint saved")

def my_text(dataloader):
    val_acc = 0.0
    val_loss = 0.0
    rnn.eval()
    for i, data in enumerate(dataloader):
        """ (1) 运用openpose """
        inputs, labels = data  # 返回的是tensor类型
        batch_size = len(inputs)
        length = len(inputs[0])
        h = len(inputs[0][0])
        w = len(inputs[0][0][0])
        c = len(inputs[0][0][0][0])
        inputs = inputs.view(-1, h, w, c)
        inputs = np.array(inputs)
        intermediate_outputs = []  # 存放通过openpose得到的特征图。len:(batch_size*length); 元素size：n*h'*w'*c'
        for idx in range(len(inputs)):
            interout = body_estimation(inputs[idx])
            interout = np.array(torch.squeeze(interout).cpu())
            intermediate_outputs.append(interout)

        """ (2) lstm """
        inputs = torch.from_numpy(np.array(intermediate_outputs)).to(mg.device)
        labels = torch.squeeze(labels).to(mg.device)
        outputs = rnn(inputs)
        loss = criterion(outputs, labels)

        val_loss += loss.item() * batch_size
        _, prediction = torch.max(outputs.data, 1)
        val_acc += torch.sum(prediction == labels.data)
        print("val: {}/{}, prediction:{} , lables:{}, equal:{}".format(i, len(dataloader), prediction, labels.data, torch.sum(prediction == labels.data)))
        print("val_acc:{}".format(val_acc))

    # Compute the average acc and loss over all 10000 test images
    val_acc = val_acc / len(dataloader)
    val_loss = val_loss / len(dataloader)
    print("val_acc:{}， val_loss：{}".format(val_acc, val_loss))
    return val_acc, val_loss

def my_train(epo_num=50):
    """### 记录训练过程每个epoch相关指标 ###"""
    # all_train_iter_loss = []    # 记录所有loss值。即每个batch的loss都会记录下来，所有epoch都记录在一起
    # 每个epoch后都要val一下，测试当前网络的性能，并记录每个epoch得到的train和val的loss、acc。 best_acc用于记录最佳val_acc值
    train_loss_list = []
    train_accuracy_list = []
    val_loss_list = []
    val_accuracy_list = []
    best_acc = 0.0

    """### 训练 ###"""
    """ 第一部分相当于将图像输入openpose得到特征图，再将得到的特征图作为rnn网络的输入，训练rnn网络的权值 """
    for epoch in range(epo_num):
        train_loss = 0.0
        train_acc = 0.0
        rnn.train()
        print("epoch:{}".format(epoch))
        for i, data in enumerate(train_dataloader):
            """ (1) 运用openpose """
            inputs, labels = data  # 返回tensor类型
            batch_size = len(inputs)
            length = len(inputs[0])
            h = len(inputs[0][0])
            w = len(inputs[0][0][0])
            c = len(inputs[0][0][0][0])
            inputs = inputs.view(-1, h, w, c)
            inputs = np.array(inputs)
            intermediate_outputs = []      # 存放通过openpose得到的特征图。len:(batch_size*length); 元素size：n*h'*w'*c'
            for idx in range(len(inputs)):
                interout = body_estimation(inputs[idx])
                interout = np.array(torch.squeeze(interout).cpu())
                intermediate_outputs.append(interout)

            """ (2) 训练lstm网络 """
            inputs = torch.from_numpy(np.array(intermediate_outputs)).to(mg.device)
            labels = torch.squeeze(labels).to(mg.device)
            optimizer.zero_grad()
            outputs = rnn(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  # 需要计算导数，则调用backward
            optimizer.step()

            iter_loss = loss.item()  # .item()返回一个具体的值，一般用于loss和acc
            # all_train_iter_loss.append(iter_loss)

            # 统计train_loss、train_acc
            train_loss += iter_loss * batch_size
            _, prediction = torch.max(outputs.data, 1)
            train_acc += torch.sum(prediction == labels.data)
            print("train: {}/{}, prediction:{}, labels:{}, equal:{}".format(i, len(train_dataloader), prediction, labels.data, torch.sum(prediction == labels.data)))
            print("train_acc:{}".format(train_acc))

            # 每5个bacth，输出一次训练过程的数据
            #if np.mod(i, 5) == 0:
            #    print('epoch {}, {}/{},train loss is {}'.format(epoch, i, len(train_dataloader), iter_loss))

        train_acc = train_acc / len(train_dataloader)
        train_loss = train_loss / len(train_dataloader)
        train_accuracy_list.append(train_acc)
        train_loss_list.append(train_loss)
        print("train_acc:{}, train_loss:{}".format(train_acc, train_loss))

        # 每个epoch都要验证（val）一下
        val_acc, val_loss = my_text(val_dataloader)
        val_accuracy_list.append(val_acc)
        val_loss_list.append(val_loss)
        # 若测试准确率高于当前最高准确率，则保存模型
        if val_acc > best_acc:
            save_models(epoch)
            best_acc = val_acc

        # 打印度量。每个epoch结束后都输出一下这个epoch得到的准确率、损失值、预测准确率
        print("Epoch {}, Train Accuracy: {} , TrainLoss: {} , Val Accuracy: {}".format(epoch, train_acc, train_loss, val_acc))

    # 训练结束后，绘制训练（train）、验证（val）的accuracy_loss曲线
    draw_accuracy_loss(train_loss_list, train_accuracy_list, epo_num, "train", 1)
    draw_accuracy_loss(val_loss_list, val_accuracy_list, epo_num, "val", 2)

if __name__ == '__main__':

    """### 训练模型 ###"""
    my_train(epo_num=50)

    """### 测试模型 ###"""
    # my_test(test_dataloader, epo_num=50)