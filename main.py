# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 13:06:17 2019

@author: Sajeevan
"""


from preprocessing import preprocessing
from train import feature_extractor, train
from predict import predict

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

##### Feature extraction #####

X, Y = preprocessing("./datasets/entreprise_1/")
X = feature_extractor(X)

##### Train step #####

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Save trained model and load it
train(X_train, y_train, RandomForestRegressor(n_estimators=10))

##### Prediction and results #####
print(predict(X_train, X_test, y_train, y_test))

