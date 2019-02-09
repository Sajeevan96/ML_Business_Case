# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 12:43:10 2019

@author: Sajeevan
"""

import pickle
from math import sqrt

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics.scorer import make_scorer



def rmse(y_true, y_pred):
    '''
        Compute the RMSE (Root Mean Square Error) metric.
        y_true:  True Sales (y) values
        y_pred: Predicted Sales (y) values by the model
    '''
    return sqrt(mean_squared_error(y_true, y_pred))


def predict(X_train, X_test, y_train, y_test):
    '''
        Predict the labels on testing values.
        X_train, y_train: Training set
        X_test, y_test: Testing set
    '''
    with open('model.pkl', 'rb') as pickle_file:
        model = pickle.load(pickle_file)
    
    #my_scorer = make_scorer(rmse, greater_is_better=True)

    y_pred = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    #cv = get_cv(X_test, y_test)
    #results = cross_validate(pipe, X, y, scoring=['f1_weighted'], cv=cv,
                            verbose=1, return_train_score=True,
                            n_jobs=1)
    return("Training RMSE: %s \n Testing RMSE: %s" % (rmse(y_train, y_pred), 
                                                      rmse(y_test, y_pred_test)))
    

    