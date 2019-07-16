#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# penalty表示L2的正则化
# solver : str, {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}, optional (default=’liblinear’)
# solver：优化算法选择参数liblinear：开源的liblinear库，使用了坐标轴下降法来迭代优化损失函数(默认),适用于小数据集
# solver：优化算法选择参数newton-cg：拟牛顿法的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数,可处理多分类问题
# solver：优化算法选择参数sag：随机平均梯度下降,使用于大一点的数据集

data = load_iris()
X = data['data']
y = data['target']
print(X[:10])
print(y[:10])

train_X,test_X,train_y,test_y = train_test_split(X,y,test_size=0.3,shuffle=1,random_state=42)

clf = LogisticRegression()
clf.fit(train_X,train_y)
print('模型详细情况: ', clf)
print('训练集分类精度: ', clf.score(train_X,train_y))
print('测试集分类精度: ', clf.score(test_X,test_y))
