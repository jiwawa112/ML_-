#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score,recall_score,roc_auc_score,roc_curve
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('2015.csv').sample(100000,random_state=50)
print(df.head())

df = df.select_dtypes('number')

# label distribution
df['_RFHLTH'] = df['_RFHLTH'].replace({2:0})
df = df.loc[df['_RFHLTH'].isin([0,1])].copy()
df = df.rename(columns = {'_RFHLTH':'label'})
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

# Decision Tree
clf = DecisionTreeClassifier(random_state=10)
clf.fit(train_X,train_y)
print(f'Decision tree has {clf.tree_.node_count} nodes with maximum depth {clf.tree_.max_depth}.')

# Assess Decision Tree Performance
train_probs = clf.predict_proba(train_X)[:,1]
probs = clf.predict_proba(test_X)[:,1]

print(f'Train ROC AUC Score: {roc_auc_score(train_y, train_probs)}')
print(f'Test ROC AUC  Score: {roc_auc_score(test_y, probs)}')

# fi = pd.DataFrame({'feature': features,
#                    'importance': clf.feature_importances_}).\
#                     sort_values('importance', ascending = False)
# print(fi.head())

# Random Forest
# Create the model with 100 trees
model = RandomForestClassifier(n_estimators=100,
                               random_state=50,
                               max_features='sqrt',
                               n_jobs=-1,verbose=1)

model.fit(train_X,train_y)

n_nodes = []
max_depths = []

for ind_tree in model.estimators_:
    n_nodes.append(ind_tree.tree_.node_count)
    max_depths.append(ind_tree.tree_.max_depth)

print(f'Average number of nodes {int(np.mean(n_nodes))}')
print(f'Average maximum depth {int(np.mean(max_depths))}')

train_rf_predict = model.predict(train_X)
train_rf_probs = model.predict_proba(train_X)[:,1]

rf_predict = model.predict(test_X)
rf_probs = model.predict_proba(test_X)[:,1]

print(model.score(test_X,test_y))