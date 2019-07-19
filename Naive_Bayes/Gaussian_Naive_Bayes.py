#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.naive_bayes import GaussianNB

X = np.array([[-1, -1], [-2, -2], [-3, -3], [-4,-4], [-5,-5], [1, 1], [2,2], [3, 3], [4,4], [5,5]])
y = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2 ])
clf = GaussianNB()
clf.fit(X,y)

print(clf.predict([[-3,-3]]))