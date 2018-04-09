# -*- coding: UTF-8 -*-
import sys
import cv2
import numpy as np
import os
reload(sys)
sys.setdefaultencoding('utf-8')

"""

本文件在python 2.7 环境下运行
运行需要的库文件：openCV 3.0， numpy
运行方法，将模板文件以及测试图片放在py文件所属目录下的test文件夹和train文件夹中，运行python hw1.py即可

"""
test_file = './test/'
module_file = './train/'

# 读取测试文件以及模板文件
tests = os.listdir(test_file)
modules = os.listdir(module_file)


def read_module(modules):
    """
    函数功能：
    读入一个文件名列表，并且根据文件名来读取对应的模板，进行二值化和尺度归一之后再输出对应的标签和模板
    参数：
    modules： 读入的文件列表名
    imgs： 返回的图片序列
    labels：返回的图片标签
    """
    imgs = []
    labels = []
    for module in modules:
        img = cv2.imread(module_file+module,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (30, 30), interpolation=cv2.INTER_CUBIC)
        mean = img.mean()
        img[img<=mean] = 0
        img[img>mean] = 255
        imgs.append(np.array(img))
        label, _ = module.split('.')
        labels.append(label)
    return np.array(imgs), labels


def read_test(tests):
    """
    函数功能：
    读入测试文件列表名，根据文件名读取对应的文件，进行尺度归一化之后输出
    参数：
    tests： 读入的测试文件
    test_imgs： 输出的尺度归一化图片序列
    """
    test_imgs = []
    for img_name in tests:
        img = cv2.imread(test_file + img_name, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (270, 70), interpolation=cv2.INTER_CUBIC)
        test_imgs.append(np.array(img))
    return np.array(test_imgs)


def cut_picture(img, x, y, w, h):
    """
    函数功能：
    输入一个图片，以及切割参数，返回切割好后的二值化图片
    参数：
    img：图片输入
    x, y: 希望切割的图片的左上角坐标
    w, h: 图片的宽度和高度
    flag: flag = 1 表示切下的图片是外接的
    """
    cut_img = img[x:x+h, y:y+w]
    mean = cut_img.mean()
    cut_img[cut_img <= mean] = 0
    cut_img[cut_img > mean] = 255
    return np.array(cut_img)


def haar_extractor(img_1, img_2):
    """
    输入一个图片，计算该图的harr特征
    img_1 :白色图
    img_2 ：阴影图
    :return: 白色图与阴影图之差的loss
    size same
    """
    num1 = sum(sum(img_1))
    num2 = sum(sum(img_2))
    return num1 - num2


def feature_extractor(img):
    """
    输入一个图片，抽取该图片的特征向量
    img: 输入图像 尺寸大小为30 × 30 
    feature: 输出的特征向量 
    32*32  ——》   31*28
    """
    feature = []
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
    # 以2×2为一个方块抽取哈尔特征：
    for i in range(0, 32, 4):
        for j in range(0, 32, 2):
            split_img = cut_picture(img, i, j, 2, 4)
            split_img[split_img == 255] = 1
            num = haar_extractor(np.array(split_img[:2,:]), np.array(split_img[2:,:]))
            feature.append(num)
    # 以2×4为一个方块提取哈尔特征：
    for i in range(0, 32, 4):
        for j in range(0, 32, 4):
            split_img = cut_picture(img, i, j, 4, 4)
            split_img[split_img == 255] = 1
            num = haar_extractor(split_img[:2,:], split_img[2:,:])
            feature.append(num)
    # 以4×4为一个方块提取哈尔特征：
    for i in range(0, 32, 8):
        for j in range(0, 32, 4):
            split_img = cut_picture(img, i, j, 4, 8)
            split_img[split_img == 255] = 1
            num = haar_extractor(split_img[:4,:], split_img[4:,:])
            feature.append(num)
    # 以8×16为一个方块提取哈尔特征：
    for i in range(0, 32, 16):
        for j in range(0, 32, 16):
            split_img = cut_picture(img, i, j, 16, 16)
            split_img[split_img == 255] = 1
            num = haar_extractor(split_img[:8, :], split_img[8:, :])
            feature.append(num)
    # 以16×16为一个方块提取哈尔特征：
    for i in range(0, 32, 32):
        for j in range(0, 32, 16):
            split_img = cut_picture(img, i, j, 16, 32)
            split_img[split_img == 255] = 1
            num = haar_extractor(split_img[:16, :], split_img[16:, :])
            feature.append(num)
    return feature


def save_img(img, img_name):
    """
    函数功能:
    保存图像到当前目录
    参数：
    img：输入图片
    img_name: 图片名称
    """
    cv2.imwrite(img_name, img)


def rec_num(img, modules, labels, gate):
    """
    功能：将输入的图片和模板转换为对应的哈尔特征，在图片上进行选定框的选取
    以及加上一些抖动，最后输出预测值
    :param img: 输入图像
    :param modules: 输入模板
    :param labels: 输入模板对应的标签
    :param gate: 门限
    :return:rec：输出预测标签
    """
    tar_lis = [[0,0],[0,30],[0,70],[0,100],[0,135],[0,175],[0,200],[0,236],
               [40, 0], [40, 35], [40, 65], [40, 100], [40, 135], [40, 170], [40, 200], [40, 236]
               ]
    rec = ""
    gitters = []
    mod_haar = []
    for i in [0,3,6,9]:
        for j in [0, 1, 2, 3, 4]:
            gitters.append([i,j])
    for module in modules:
        module_ = feature_extractor(module)
        mod_haar.append(module_)
    mod_haar = np.array(mod_haar)
    for x,y in tar_lis:
        d = {}
        for gitter in gitters:
            if x == 40:
                x_ = x
            else:
                x_ = x + gitter[0]
            y_ = y + gitter[1]
            sample = cut_picture(img, x_, y_, 30, 30)
            sample_haar = np.array(feature_extractor(sample))
            index, loss = loss_func(sample_haar, mod_haar, labels, gate)
            d[loss] = index
        index = d[min(d.keys())]
        rec += index
    return rec


def loss_func (sample_haar, mod_haar, labels, gate):
    """
    计算样例和模板的均方误差
    :param sample_haar: 样本的哈尔特征向量
    :param mod_haar: 模板的哈尔特征向量
    :param labels: 标签
    :param gate: 门限
    :return: index
    """
    d = {}
    for i in range(len(mod_haar)):
        loss = sample_haar - mod_haar[i]
        loss = sum(loss ** 2)
        if loss < gate:
            d[loss] = labels[i]
    if d:
        return d[min(d.keys())], min(d.keys())
    else:
        return " ", 10000
# 主函数
if __name__ == '__main__':
    mod_img, labels = read_module(modules)
    test_img = read_test(tests)
    for img in test_img:
        rec = rec_num(img, mod_img, labels, gate = 6000)
        print(rec)
    print("20130129 181641")