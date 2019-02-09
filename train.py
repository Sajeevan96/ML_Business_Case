# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 16:19:27 2019

@author: Clement_X240
"""

from sklearn.ensemble import RandomForestRegressor
import pickle
import pandas as pd


def feature_extractor(X):
    '''
        Extract some new features from the dataset. 
        X: Dataset (without labels)
    '''
    X['Month'] = X.Date.dt.month
    X['Year'] = X.Date.dt.year
    X['Day'] = X.Date.dt.day
    X['Week'] = X.Date.dt.weekofyear
    X = X.drop('Date', 1)
    
    X['AVG_Month'] = X[['Year','Month','Customers']].groupby(['Year','Month']).transform('mean')
    X['AVG_Week'] = X[['Year','Week','Customers']].groupby(['Year','Week']).transform('mean')
    X['AVG_Day'] = X[['Year','DayOfWeek','Customers']].groupby(['Year','DayOfWeek']).transform('mean')

    return X
    
    
def train(X_train, y_train, model):
    '''
        Train and save the model.
        X_train, y_train: Training set
        model: the chosen model
    '''
    model.fit(X_train, y_train)
    
    # Save the trained model
    output = open('model.pkl', 'wb')
    pickle.dump(model, output)
