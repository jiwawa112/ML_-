#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge,RidgeCV
from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV

# 创建样本数据集
data = list()
for i in range(30):
    data.append([i+np.random.rand()*3,3.5*i+np.random.rand()*3])

#生成矩阵
dataMat = np.array(data)
X = dataMat[:,0:1]
y = dataMat[:,1]

# 一般情况下的Ridge(alpha：正则化强度)
model_ridge = Ridge(alpha=0.5)
model_ridge.fit(X,y)
print('系数:\n',model_ridge.coef_)
print('线性回归模型详情:\n',model_ridge)
pred_1 = model_ridge.predict(X)

# 通过RidgeCV可以设置多个参数值，算法使用交叉验证获取最佳参数值
model = RidgeCV(alphas=[0.1, 1.0, 10.0])
model.fit(X, y)   # 线性回归建模
print('系数:\n',model.coef_)
print('线性回归模型详情:\n',model)
print('交叉验证最佳alpha值',model.alpha_)  # Ridge()无这个方法，只有RidgeCV算法有

pred_2 = model.predict(X)
# 绘制散点图
plt.scatter(X, y, marker='x')
plt.plot(X, pred_1,c='r')
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Lasso回归
model = Lasso(alpha=0.01)  # 调节alpha可以实现对拟合的程度
model = LassoCV()  # LassoCV自动调节alpha可以实现选择最佳的alpha。
model = LassoLarsCV()  # LassoLarsCV自动调节alpha可以实现选择最佳的alpha
model.fit(X, y)   # 线性回归建模
print('系数:\n',model.coef_)
print('线性回归模型详情:\n',model)
# print('最佳的alpha：',model.alpha_)  # 只有在使用LassoCV、LassoLarsCV时才有效

pred = model.predict(X)

# 绘制散点图
plt.scatter(X, y, marker='x')
plt.plot(X, pred,c='r')
plt.xlabel("x")
plt.ylabel("y")
plt.show()




