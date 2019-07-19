#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.naive_bayes import BernoulliNB
X = np.array([[1,2,3,4],[1,3,4,4],[2,4,5,5]])
y = np.array([1,1,2])
clf = BernoulliNB()
clf.fit(X,y)

print(clf.predict([[3,2,3,4]]))