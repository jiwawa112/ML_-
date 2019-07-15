#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# boston housing

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

train_path = 'F:/python/NLP_course/NLP_course/Course_11/data/boston-housing/train.csv'
test_path = 'F:/python/NLP_course/NLP_course/Course_11/data/boston-housing/test.csv'

train_data = pd.read_csv(train_path,encoding='utf-8')
test_data = pd.read_csv(test_path,encoding='utf-8')

X = train_data.drop(['ID','medv'],axis=1)
y = train_data.medv

train_X,test_X,train_y,test_y = train_test_split(X,y,test_size=0.3,random_state=42)

xg_reg = xgb.XGBRegressor(objective='reg:linear',colsample_bytree=0.3,learning_rate=0.1,
                          max_depth=8,alpha=8,n_estimators=500,reg_lambda=1)

y_train_hat = xg_reg.fit(train_X,train_y)
print(y_train_hat)
#error = mean_squared_error(y_train_hat,train_y)
#print(error)

test_X = test_data.drop(['ID'],axis=1)
predictions = xg_reg.predict(test_X)
ID = (test_data.ID).astype(int)
result = np.c_[ID,predictions]

np.savetxt('F:/python/NLP_course/NLP_course/Course_11/data/boston-housing/xgb_submission.csv',
           result,fmt="%d,%.4f" ,header='ID,medv', delimiter=',', comments='')


