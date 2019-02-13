# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 16:19:27 2019

@author: Clement_X240
"""

from sklearn.ensemble import RandomForestRegressor
import pickle
import pandas as pd
import matplotlib.pyplot as plt

def feature_extractor(X):
    '''
        Extract some new features from the dataset. 
        X: Dataset (without labels)
    '''
    # add some distincts columns for Month, Year and Day 
    X['Month'] = X.Date.dt.month
    X['Year'] = X.Date.dt.year
    X['Day'] = X.Date.dt.day
    
    # return a number indicating which week of the year the date falls in
    X['Week'] = X.Date.dt.weekofyear
    
    # remove the Date's column 
    X = X.drop('Date', 1)
    
    # calculate monthly/weekly/daily average customers
    X['AVG_Month'] = X[['Year','Month','Customers']].groupby(['Year','Month']).transform('mean')
    X['AVG_Week'] = X[['Year','Week','Customers']].groupby(['Year','Week']).transform('mean')
    X['AVG_Day'] = X[['Year','DayOfWeek','Customers']].groupby(['Year','DayOfWeek']).transform('mean')

    # calculate how long the nearest competitor has been opened
    X['Competition_Duration'] = 12 *(X.Year - X.CompetitionOpenSinceYear) *(X.CompetitionOpenSinceYear>0) + (X.Month - X.CompetitionOpenSinceMonth) *(X.CompetitionOpenSinceMonth>0)
    
    # calculate the duration of Promo2
    X['Promo_Duration'] = 12 *(X.Year - X.Promo2SinceYear) *(X.Promo2SinceYear>0) + (X.Week - X.Promo2SinceWeek) *(X.Promo2SinceWeek>0)
    
    # remove some columns 
    X = X.drop('CompetitionOpenSinceYear', 1)
    X = X.drop('CompetitionOpenSinceMonth', 1)
    X = X.drop('Promo2SinceYear', 1)
    X = X.drop('Promo2SinceWeek', 1)
    
    return X
    
    
def train(X_train, y_train, model):
    '''
        Train and save the model.
        X_train, y_train: Training set
        model: the chosen model
    '''
    for i in range(2):
        model.fit(X_train, y_train)
        model.n_estimators +=10
        print("{} trees\t oob_score {}".format(i*10+10, model.oob_score_))
        
    #plt.plot(model.oob_score_)
    # Save the trained model
    output = open('model.pkl', 'wb')
    pickle.dump(model, output)
