import pickle
from math import sqrt

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics.scorer import make_scorer


def rmspe(y_true, y_pred):
    '''
        Compute the RMSPE (Root Mean Square Percentage Error) metric.
        y_true:  True Sales (y) values
        y_pred: Predicted Sales (y) values by the model
    '''
    diff = y_pred - y_true
    diff_percentage = diff / y_true
    diff_percentage_squared = diff_percentage ** 2
    return sqrt(diff_percentage_squared.mean())


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
                            #verbose=1, return_train_score=True,
                            #n_jobs=1)
    return("Training RMSPE: %s \n Testing RMSPE: %s" % (rmspe(y_train, y_pred), 
                                                      rmspe(y_test, y_pred_test)))
    

    