#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score,recall_score,roc_auc_score,roc_curve
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('2015.csv').sample(100000,random_state=50)
print(df.head())

df = df.select_dtypes('number')

# label distribution
df['_RFHLTH'] = df['_RFHLTH'].replace({2:0})
df = df.loc[df['_RFHLTH'].isin([0,1])].copy()
df = df.rename(columns={'_RFHLTH':'label'})
df['label'].value_counts()

# Remove columns with missing values
df = df.drop(columns=['POORHLTH', 'PHYSHLTH', 'GENHLTH', 'PAINACT2',
                      'QLMENTL2', 'QLSTRES2', 'QLHLTH2', 'HLTHPLN1', 'MENTHLTH'])

# Split Data into Training and Testing Set

# Extract the labels
labels = np.array(df.pop('label'))
train_X,test_X,train_y,test_y = train_test_split(df,labels,test_size=0.3,random_state=50)

# imputation of Missing values
train_X = train_X.fillna(train_X.mean())
test_X = test_X.fillna(test_X.mean())

print(train_X.shape)
print(train_y.shape)
print(test_X.shape)
print(test_y.shape)

# 默认rf
rf0 = RandomForestClassifier(oob_score=True,random_state=10)
rf0.fit(train_X,train_y)
print(rf0.oob_score_)
y_hat = rf0.predict(test_X)
print("AUC Score {}".format(roc_auc_score(test_y,y_hat)))

# 对外层的bagging框架进行参数择优，即对n_estimators参数择优
# n_esimators参数择优的范围是：1~101，步长为10。十折交叉验证率选择最优n_estimators
# 优化决策树参数的最大特征数max_features，其他参数设置为常数，且n_estimators为上面跑出的结果
# max_features参数择优的范围：1~11，步长为1，十折交叉验证率选择最优max_features
# param_test1 = {'n_estimators':range(1,101,10),'max_features':range(1,11,1),'max_depth':range(10,100,1)}

"""
param_test1 = {'n_nestimators':range(1,101,10)}
gsearch1 = GridSearchCV(estimator=RandomForestClassifier(),param_grid=param_test1,scoring='roc_auc',cv=10)
gsearch1.fit(train_X,train_y)
print(gsearch1.best_params_)
print(gsearch1.best_estimator_)
print(gsearch1.best_score_)
print(gsearch1.best_index_)
print('best accuracy:%f' % gsearch1.best_score_)


param_test2 = {'max_features':range(1,11,1)}
gsearch2 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=91,random_state=10),
                        param_grid=param_test2,scoring='roc_auc',cv=10)
gsearch2.fit(train_X,train_y)
print(gsearch2.best_params_)
print(gsearch2.best_estimator_)
print(gsearch2.best_score_)
print(gsearch2.best_index_)
print('best accuracy:%f' % gsearch2.best_score_)


param_test3 = {'max_depth':range(10,20,1)}
gsearch3 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=91,max_features=9,random_state=10),
                        param_grid=param_test3,scoring='roc_auc',cv=10)
gsearch3.fit(train_X,train_y)
print(gsearch3.best_params_)
print(gsearch3.best_estimator_)
print(gsearch3.best_score_)
print(gsearch3.best_index_)
print('best accuracy:%f' % gsearch3.best_score_)


"""

# 用最优参数重新训练数据，计算泛化误差
rf0 = RandomForestClassifier(n_estimators=91,max_features=9,max_depth=16,oob_score=True,random_state=10)
rf0.fit(train_X,train_y)
print(rf0.oob_score_)
print(" auc accuracy: %f" % rf0.oob_score_)
print(rf0.score(test_X,test_y))


# 上面决策树参数中最重要的包括最大特征数max_features， 最大深度max_depth
# 内部节点再划分所需最小样本数min_samples_split和叶子节点最少样本数min_samples_leaf(样本数量不大，可以不用管这两个值)

