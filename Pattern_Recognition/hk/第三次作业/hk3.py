import numpy as np
import glob
import cv2

def sim_distance(train,test):
    '''
    计算欧氏距离相似度
    :param train: 二维训练集
    :param test: 一维测试集
    :return: 该测试集到每一个训练集的欧氏距离
    '''
    return [np.linalg.norm(i - test) for i in train]

picture_path = './face/train/'
img_train = {}
array_list = []
i = 0
for name in glob.glob(picture_path+'*.jpg'):
    img_pre = cv2.imread(name,0)  
    img_train[i] = img_pre
    array_list.append(np.array(img_pre).reshape((1,19*19)))
    i += 1

mat = np.vstack((array_list)) # 将上述多个一维序列合并成矩阵 3*120000
P = np.dot(mat,mat.transpose()) # 计算P
v,d = np.linalg.eig(P) # 计算P的特征值和特征向量
d= np.dot(mat.transpose(),d) # 计算Sigma的特征向量 12000 * 3
train = np.dot(d.transpose(),mat.transpose()) # 计算训练集的主成分值 3*3

# 开始测试
test_pic = np.array(cv2.imread('./face/test/face.jpg',0)).reshape((1,19*19))
result = sim_distance(train.transpose(),np.dot(test_pic,d))
print result
test_pic = np.array(cv2.imread('./face/test/nonface.jpg',0)).reshape((1,19*19))
result = sim_distance(train.transpose(),np.dot(test_pic,d))
print result

# test_pic = np.array(Image.open('c.jpg').convert('L')).reshape((1,120000))
# result = sim_distance(train.transpose(),np.dot(test_pic,d))
# print result
# test_pic = np.array(Image.open('e.jpg').convert('L')).reshape((1,120000))
# result = sim_distance(train.transpose(),np.dot(test_pic,d))
# print result