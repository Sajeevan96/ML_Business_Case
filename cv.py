from preprocessing import preprocessing
from train import feature_extractor, train, split_dataset
from predict import predict

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xg
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

X, y = preprocessing("./datasets/entreprise_1/")
X, y = feature_extractor(X, y)

# ##### Train step #####

# Split data
X_train, X_test, y_train, y_test = split_dataset(X, y) #train_test_split(X, Y, test_size=0.3)

def rmspe_xg(yhat, y):
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return ("rmspe", rmspe(y,yhat))

def rmspe(y, yhat):
    return np.sqrt(np.mean((yhat/y-1) ** 2))

lr = 0.03
params = {"objective": "reg:linear", # for linear regression
          "booster" : "gbtree",   # use tree based models 
          "eta": lr,   # learning rate 0.03
          "max_depth": 10,    # maximum depth of a tree
          "silent": 1   # silent mode
          }
num_boost_round = 2500

dtrain = xg.DMatrix(X_train, y_train)
dvalid = xg.DMatrix(X_test, y_test)
watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
# train the xgboost model
model = xg.train(params, dtrain, num_boost_round, evals=watchlist, \
  early_stopping_rounds= 100000, feval=rmspe_xg, verbose_eval=True)
kfold = KFold(n_splits=10, random_state=7)
results = cross_val_score(model, X, Y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
