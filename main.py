# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 13:06:17 2019

@author: Sajeevan
"""


from preprocessing import X, y
from train import feature_extractor, train
from predict import predict

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

##### Feature extraction #####

X = feature_extractor(X)

##### Train step #####

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# Save trained model and load it
train(X_train, y_train, RandomForestRegressor())

##### Prediction and results #####
predict(X_train, X_test, y_train, y_test)

