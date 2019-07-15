#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge,RidgeCV
from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV
from sklearn.metrics import mean_squared_error

data = load_boston()
data_X = data['data']
data_y = data['target']
print(data['data'][:10])
print(data['target'][:10])

train_X,test_X,train_y,test_y = train_test_split(data_X,data_y,test_size=0.3,random_state=42)
print(test_X.shape)
print(test_y.shape)

# Ridge回归
# 一般情况下的Ridge(alpha：正则化强度)
model_ridge = Ridge(alpha=0.5)
model_ridge.fit(train_X,train_y)
print('训练集预测的确定系数R ^ 2: ', model_ridge.score(train_X,train_y))
print('验证集预测的确定系数R ^ 2: ', model_ridge.score(test_X,test_y))
pred_1 = model_ridge.predict(test_X)
print('模型误差: ',mean_squared_error(test_y,pred_1))

# 通过RidgeCV可以设置多个参数值，算法使用交叉验证获取最佳参数
model = RidgeCV(alphas=[0.001,0.01,0.1,1.0])
model.fit(train_X,train_y)
print("模型参数:",model.get_params())
print("模型详情:",model)
print('最佳alpha',model.alpha_)       # Ridge()无这个方法，只有RidgeCV算法有
print('训练集预测的确定系数R ^ 2: ', model.score(train_X,train_y))
print('验证集预测的确定系数R ^ 2: ', model.score(test_X,test_y))

pred_2 = model.predict(test_X)
print('Ridge模型误差: ',mean_squared_error(test_y,pred_2))

# Lasso回归
model_lasso = Lasso(alpha=0.01)
model_lasso = LassoCV()
model_lasso = LassoLarsCV()
model_lasso.fit(train_X,train_y)
print("模型参数:",model_lasso.get_params())
print("模型详情:",model_lasso)
#print('最佳alpha',model_lasso.alpha_)
print('训练集预测的确定系数R ^ 2: ', model_lasso.score(train_X,train_y))
print('验证集预测的确定系数R ^ 2: ', model_lasso.score(test_X,test_y))

pred_3 = model_lasso.predict(test_X)
print('Lasso模型误差: ',mean_squared_error(test_y,pred_3))