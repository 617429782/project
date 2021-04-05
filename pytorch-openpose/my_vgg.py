from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchsnooper
import time
import src.my_create_dataset as dataset
from src import my_global as mg
from src.util import draw_accuracy_loss as draw_accuracy_loss

"""### 全局变量 ###"""
time = str(time.time())
# 数据集
dataPath = "/home/jlm/pytorch-openpose/data/ut-interaction_dataset/dataset.txt"
all_dataset = dataset.MyDataset(dataPath)

# 将数据集划分为trian、val、test三类（0.8，0.1，0.1） ps：测试能成后记得把util里自制的数据集划分删掉
train_size = int(0.8 * len(all_dataset))
val_size = int(0.1 * len(all_dataset))
test_size = len(all_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(all_dataset, [train_size, val_size, test_size])

# dataloader
train_dataloader = DataLoader(train_dataset, batch_size=mg.batch_size, shuffle=True, num_workers=1)
val_dataloader = DataLoader(val_dataset, batch_size=mg.batch_size, shuffle=True, num_workers=1)
test_dataloader = DataLoader(test_dataset, batch_size=mg.batch_size, shuffle=True, num_workers=1)

# 网络
vgg = models.vgg16(pretrained=True).to(mg.device)
vgg_last_layer = (vgg.classifier)[-1]
in_features = vgg_last_layer.in_features
(vgg.classifier)[-1] = nn.Linear(in_features, 6).to(mg.device)
for i,p in enumerate(vgg.parameters()):
    if i < 30:
        p.requires_grad = False

# 损失函数、优化器（待定)
criterion = nn.CrossEntropyLoss().to(mg.device)   # CrossEntropyLoss()函数的主要是将softmax-log-NLLLoss(交叉熵)合并到一块得到的结果
optimizer = optim.Adam(vgg.parameters())

def save_models(epoch):
    modelPath = "/home/jlm/pytorch-openpose/model/myVggModel_{}.model".format(epoch)
    torch.save(vgg.state_dict(), modelPath)
    print("Chekcpoint saved")
    return modelPath

def get_outputs(inputs):
    batch_size = len(inputs)
    length = len(inputs[0])
    h = len(inputs[0][0])
    w = len(inputs[0][0][0])
    c = len(inputs[0][0][0][0])
    inputs = inputs.view(-1, h, w, c)
    inputs = inputs.permute(0, 3, 1, 2)  # x*h*w*c -> x*c*h*w
    inputs = inputs.type(torch.float32).to(mg.device)
    inputs = vgg(inputs)
    inputs = inputs.view(batch_size, length, -1)
    outputs = []
    for i in range(batch_size):
        print("i:{}".format(i))
        result = torch.zeros(6).to(mg.device)
        for j in range(length):
            print("result:{}".format(result))
            print("in:{}".format(inputs[i][j]))
            result += inputs[i][j]
            print("result:{}".format(result))
        outputs.append(result / float(length))
        print("ou:{}".format(outputs[i]))
    outputs = torch.stack(outputs)
    return outputs

def my_val(dataloader):
    val_acc = 0.0
    val_loss = 0.0
    vgg.eval()
    for i, data in enumerate(dataloader):
        inputs, labels = data  # 返回的是tensor类型
        labels = torch.squeeze(labels).to(mg.device)
        batch_size = len(inputs)

        outputs = get_outputs(inputs)
        loss = criterion(outputs, labels)

        val_loss += loss.item() * batch_size
        _, prediction = torch.max(outputs.data, 1)
        val_acc += torch.sum(prediction == labels.data)
        print("vgg: {}/{}, prediction:{}, lables:{}, equal:{}".format(i, len(dataloader), prediction, labels.data, torch.sum(prediction == labels.data)))
        print("val_acc:{}".format(val_acc))

    val_acc = float(val_acc) / float(val_size)
    val_loss = float(val_loss) / float(len(dataloader))
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
    best_weight_path = "None"

    """### 训练 ###"""
    """ 第一部分相当于将图像输入openpose得到特征图，再将得到的特征图作为rnn网络的输入，训练rnn网络的权值 """
    for epoch in range(epo_num):
        print("epoch:{}".format(epoch))
        train_loss = 0.0
        train_acc = 0.0
        vgg.train()
        for i, data in enumerate(train_dataloader):
            inputs, labels = data  # 返回的是tensor类型
            batch_size = len(inputs)
            labels = torch.squeeze(labels).to(mg.device)
            outputs = get_outputs(inputs)

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

        # train_acc = train_acc / len(train_dataloader)
        train_acc = float(train_acc) / float(train_size)
        train_loss = float(train_loss) / float(len(train_dataloader))
        train_accuracy_list.append(train_acc)
        train_loss_list.append(train_loss)
        print("train_acc:{}, train_loss:{}".format(train_acc, train_loss))

        # 每个epoch都要验证（val）一下
        val_acc, val_loss = my_val(val_dataloader)
        val_accuracy_list.append(val_acc)
        val_loss_list.append(val_loss)
        # 若测试准确率高于当前最高准确率，则保存模型
        if val_acc > best_acc:
            best_weight_path = save_models(epoch)
            best_acc = val_acc

        # 打印度量。每个epoch结束后都输出一下这个epoch得到的准确率、损失值、预测准确率
        print("Epoch {}, Train Accuracy: {} , TrainLoss: {} , Val Accuracy: {}".format(epoch, train_acc, train_loss, val_acc))

    # 训练结束后，绘制训练（train）、验证（val）的accuracy_loss曲线
    draw_accuracy_loss(train_loss_list, train_accuracy_list, epo_num, "train", time, 1)
    draw_accuracy_loss(val_loss_list, val_accuracy_list, epo_num, "val", time, 2)

    return best_weight_path

def my_test(best_weight_path, dataloader):
    if best_weight_path == "None":
        print("best_weight_path is None")
    else:
        checkpoint = torch.load(best_weight_path)
        vgg.load_state_dict(checkpoint)
        vgg.to(mg.device)
        vgg.eval()

    test_acc = 0.0
    for i, data in enumerate(dataloader):
        inputs, labels = data  # 返回的是tensor类型
        batch_size = len(inputs)
        labels = torch.squeeze(labels).to(mg.device)
        outputs = get_outputs(inputs)

        _, prediction = torch.max(outputs.data, 1)
        test_acc += torch.sum(prediction == labels.data)

    # Compute the average acc and loss over all 10000 test images
    val_acc = float(test_acc) / float(test_size)
    print("test_acc:{}".format(val_acc))

if __name__ == "__main__":

    """### 训练模型 ###"""
    best_weight_path = my_train(epo_num=50)

    """### 测试模型 ###"""
    my_test(best_weight_path, test_dataloader)