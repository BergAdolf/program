import os
import json
import pickle
import random
import gzip
import shutil
import sys

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm


def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    im_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    im_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(im_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('../data/class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    test_images_path = []  # 存储验证集的所有图片路径
    test_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in im_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        im_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                test_images_path.append(img_path)
                test_images_label.append(im_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(im_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_label)))
    print("{} images for validation.".format(len(test_images_label)))

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(im_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(im_class)), im_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, test_images_path, test_images_label


def read_valid_data(root: str):
    # 遍历文件夹，一个文件夹对应一个类别
    im_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    im_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(im_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('../data/class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    images_path = []  # 存储训练集的所有图片路径
    images_label = []  # 存储训练集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型

    for cla in im_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本

        for img_path in images:
            images_path.append(img_path)
            images_label.append(image_class)
    print("{} images were found in the dataset.".format(sum(every_class_num)))

    return images_path, images_label


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def data_zip(file_dir: str, zip_dir: str):
    with open(file_dir, 'rb') as f_in, gzip.open(zip_dir, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)


def data_unzip(file_name: str):
    with gzip.open(file_name, 'rb') as f:
        x_train, y_train, x_valid, y_valid = pickle.load(f, encoding="latin-1")
    return x_train, y_train, x_valid, y_valid


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.MSELoss()
    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data

        pred = model(images.to(device))

        loss = loss_function(pred.transpose(0, 1), labels.to(device).float())
        loss.backward()
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

        data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 3))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return mean_loss.item()


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()

    # 验证样本总个数
    total_num = len(data_loader.dataset)

    # 用于存储预测正确的样本个数
    sum_num = torch.zeros(1).to(device)
    pos_num = torch.zeros(1).to(device)
    neg_num = torch.zeros(1).to(device)

    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data
        pred = model(images.to(device))
        pred[pred > 0.5] = 1
        pred[pred < 0.5] = 0
        pred = pred.transpose(0, 1)
        labels = labels.to(device).float()
        sum_num += torch.eq(pred, labels).sum().item()
        pos_num += (torch.eq(pred, labels).float() * labels).sum().item()
        neg_num += (torch.eq(pred, labels).float() * (1 - labels)).sum().item()

    return sum_num.item() / total_num, pos_num, neg_num
