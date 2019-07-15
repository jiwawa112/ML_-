#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate

# 生成2维正态分布，生成的数据按分位数分为两类，500个样本,2个样本特征，协方差系数为2
X1,y1 = make_gaussian_quantiles(cov=2.0,n_samples=500,n_features=2,n_classes=2,random_state=10)
# 生成2维正态分布，生成的数据按分位数分为两类，400个样本,2个样本特征均值都为3，协方差系数为2
X2,y2 = make_gaussian_quantiles(mean=(3,3),cov=1.5,n_samples=400,n_features=2,n_classes=2,random_state=10)

X = np.concatenate((X1,X2))
y = np.concatenate((y1,-y2+1))

plt.scatter(X[:,0],X[:,1],marker='o',c=y)
plt.show()

bdt1 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2,min_samples_split=20,min_samples_leaf=5),
                         algorithm='SAMME',n_estimators=200,learning_rate=0.8)
bdt1.fit(X,y)

x_min,x_max = X[:,0].min() - 1,X[:,0].max() + 1
y_min,y_max = X[:,1].min() - 1,X[:,1].max() + 1
xx,yy = np.meshgrid(np.arange(x_min,x_max,0.02),
                    np.arange(y_min,y_max,0.02))

Z = bdt1.predict(np.c_[xx.ravel(),yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx,yy,Z)
plt.scatter(X[:,0],X[:,1],marker='o',c=y)
plt.show()
print("Score:",bdt1.score(X,y))

"""
bdt2= AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
                         algorithm="SAMME",
                         n_estimators=300, learning_rate=0.8)
bdt2.fit(X, y)
print("Score:", bdt2.score(X,y))


bdt3 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
                         algorithm="SAMME",
                         n_estimators=300, learning_rate=0.5)
bdt3.fit(X, y)
print("Score:", bdt3.score(X,y))



bdt4 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
                         algorithm="SAMME",
                         n_estimators=600, learning_rate=0.7)
bdt4.fit(X, y)
print("Score:", bdt4.score(X,y))
"""

# 优化弱学习器个数
param_test1 = {"n_estimators":range(150,300,50)}
gsearch1 = GridSearchCV(estimator=AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),200),
                        param_grid=param_test1,scoring='roc_auc',cv=5)
gsearch1.fit(X,y)
n_estimators1 = gsearch1.best_params_
print(gsearch1.best_params_)
print(gsearch1.best_score_)
print(type(gsearch1.best_params_))


# 再次优化弱学习器个数
param_test2 = {"n_estimators":range(gsearch1.best_params_['n_estimators']-30,gsearch1.best_params_['n_estimators']+30,10)}
gsearch2 = GridSearchCV(estimator=AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),200),
                        param_grid=param_test2,scoring='roc_auc',cv=5)
gsearch2.fit(X,y)
print(gsearch2.best_params_)
print(gsearch2.best_score_)

# 弱学习器的参数择优
for i in range(1,3):
    print((i))
    for j in range(18,22):
        print(j)
        bdt5 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=i,min_samples_split=j),
                                                         n_estimators=gsearch2.best_params_['n_estimators'])
        cv_result = cross_validate(bdt5,X,y,return_train_score=False,cv=5)
        cv_value_vec = cv_result['test_score']
        cv_mean = np.mean(cv_value_vec)
        if cv_mean >= score:
            score = cv_mean
            tree_depth = i
            samples_split = j

# 使用最新参数构建模型
bdt6 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=tree_depth),n_estimators=gsearch2.best_params_['n_estimators'])
bdt6.fit(X,y)
print(bdt6.score(X,y))