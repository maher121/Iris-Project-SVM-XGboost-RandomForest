# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 21:53:25 2021

@author: Maher
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# importing machine learning models for prediction
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
import xgboost as xgboost


# load iris dataset
colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
irisdata= pd.read_csv('iris.data',names=colnames)
irisdata.head(10)

#split the dataset into input features (X) and the feature we wish to predict (Y)
X = irisdata.drop('Class', axis=1)
y = irisdata['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


model_1 = SVC(kernel='poly', degree=8)

### XGboost #################
# convert label to number
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
y_train_xgb=le.fit_transform(y_train)
y_test_xgb=le.fit_transform(y_test)

model_2 = xgboost.XGBClassifier(use_label_encoder=False,eval_metric='mlogloss')
##### End XGboost ##########

model_3 = RandomForestClassifier()

######################################################
# training all the model on the training dataset
model_1.fit(X_train, y_train)
model_2.fit(X_train, y_train_xgb)
model_3.fit(X_train, y_train)

# # predicting the output on the validation dataset
pred_1 = model_1.predict(X_test)
pred_2 = model_2.predict(X_test)
pred_3 = model_3.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print("------------------  SVM Classfier----------------")
# print(confusion_matrix(y_test, pred_1))
print(classification_report(y_test, pred_1))

print("------------------  xgboost Classfier----------------")
print(confusion_matrix(y_test_xgb, pred_2))
print(classification_report(y_test_xgb, pred_2))

print("------------------  RandomForest Classfier----------------")
# print(confusion_matrix(y_test, pred_3))
print(classification_report(y_test, pred_3))

#------------------ Ensemble Classfier ------------------------
# Making the final model using voting classifier
final_model = VotingClassifier(
    estimators=[('lr', model_1), ('rf', model_3)], voting='hard')
 # training all the model on the train dataset
final_model.fit(X_train, y_train)
 # predicting the output on the test dataset
pred_final = final_model.predict(X_test)

print("------------------  Ensemble Classfier----------------")
print(confusion_matrix(y_test, pred_final))
print(classification_report(y_test, pred_final))
