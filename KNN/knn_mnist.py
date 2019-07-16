#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# load mnist_data
data = load_digits()
print(data['data'][:10])
print(data['target'][:10])

# 划分训练集和测试集 75%数据作为训练集 25%数据作为测试集
(train_X, test_X, train_y, test_y) = train_test_split(np.array(data['data']),
                                                      data['target'], test_size=0.25, random_state=42)
# 从训练数据中再划分出10%的数据作为验证集
(train_X, val_X, train_y, val_y) = train_test_split(train_X, train_y,
                                                    test_size=0.1, random_state=84)

# 数据集大小
print("training data : {}".format(len(train_y)))
print("validation data : {}".format(len(val_y)))
print("testing data : {}".format(len(test_y)))

k_points = range(1,30,2)
accuracies = []

for k in range(1,30,2):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(train_X,train_y)

    score = model.score(val_X,val_y)
    print("k=%d,accuracy=%.2f%%" % (k,score*100))
    accuracies.append(score)

i = int(np.argmax(accuracies))
print("k=%d achieved highest accuracy of %.2f%% on validation data"
      % (k_points[i],accuracies[i] * 100))

# predicted
model = KNeighborsClassifier(n_neighbors=k_points[i])
model.fit(train_X,train_y)
pred = model.predict(test_X)

print(classification_report(test_y,pred))