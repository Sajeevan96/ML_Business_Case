from sklearn.ensemble import RandomForestRegressor
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def feature_extractor(df):
    '''
        Extract some new features from the dataset. 
        X: Dataset (without labels)
    '''
    
    # Drop 0 labels
    df = df[df.Sales != 0]
    
    X = df.loc[:, df.columns != "Sales"]
    y = df["Sales"]
    # add some distincts columns for Month, Year and Day 
    X['Month'] = X.Date.dt.month
    X['Year'] = X.Date.dt.year
    X['Day'] = X.Date.dt.day
    
    # return a number indicating which week of the year the date falls in
    X['Week'] = X.Date.dt.weekofyear
    # remove the Date's column 
    X = X.drop('Date', 1)
    
    # Drop Customers
    X = X.drop('Customers', 1)
    
    # Drop Open
    X = X.drop('Open', 1)
    
    # calculate how long the nearest competitor has been opened
    X['Competition_Duration'] = 12 *(X.Year - X.CompetitionOpenSinceYear) *(X.CompetitionOpenSinceYear>0) + (X.Month - X.CompetitionOpenSinceMonth) *(X.CompetitionOpenSinceMonth>0)
    
    # calculate the duration of Promo2
    X['Promo_Duration'] = 12 *(X.Year - X.Promo2SinceYear) *(X.Promo2SinceYear>0) + (X.Week - X.Promo2SinceWeek) *(X.Promo2SinceWeek>0)
    
    X["CompetitionDistanceInt"] = X.CompetitionDistance.copy()
    intervalle_distance = np.array([45000, 30000, 18000, 10000, 4000, 1500, 500])
    X.CompetitionDistanceInt[X.CompetitionDistanceInt <= 500] = len(intervalle_distance)
    for i in range(len(intervalle_distance)):
        X.CompetitionDistanceInt[X.CompetitionDistanceInt > intervalle_distance[i]] = i
    
    # remove some columns 
    X = X.drop('CompetitionOpenSinceYear', 1)
    X = X.drop('CompetitionOpenSinceMonth', 1)
    X = X.drop('Promo2SinceYear', 1)
    X = X.drop('Promo2SinceWeek', 1)
    
    return X, y


def split_dataset(X, y, test_size=0.33):
    '''
        Split data into train and test set
        X: Dataset
        y: Labels
    '''
    first_year = np.min(X.Year)
    first_month =np.min(X[X.Year==first_year].Month)
    X["nb_month"] = (X.Year-first_year) * 12 + (X.Month - first_month)
    nb_months = len(np.unique(X.nb_month))
    l = np.random.randint(0, nb_months, int(test_size * nb_months))
    X_test = X[X.nb_month.isin(l)]
    y_test = y[X_test.index]

    X_train = X.drop(X_test.index)
    y_train = y[X_train.index]

    X_train = X_train.drop(["nb_month"], 1)
    X_test = X_test.drop(["nb_month"], 1)
    return X_train, X_test, y_train, y_test

def train(X_train, y_train, model):
    '''
        Train and save the model.
        X_train, y_train: Training set
        model: the chosen model
    '''
    for i in range(4):
        model.fit(X_train, y_train)
        print("{} trees\t oob_score {}".format(model.n_estimators, model.oob_score_))
        model.n_estimators +=10
        
    #plt.plot(model.oob_score_)
    # Save the trained model
    output = open('model.pkl', 'wb')
    pickle.dump(model, output)
