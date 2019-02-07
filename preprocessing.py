# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 16:11:25 2019

@author: Clement_X240
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder


train = pd.read_csv("./datasets/entreprise_1/train.csv")
store = pd.read_csv("./datasets/entreprise_1/store.csv")

train_store = pd.merge(train, store, how='inner', on="Store")
del(train)
del(store)

train_store.StateHoliday.replace(0, '0', inplace=True)
train_store['Date'] = pd.to_datetime(train_store.Date,
           format='%Y-%m-%d', errors='coerce')

train_store.CompetitionDistance.fillna(train_store.CompetitionDistance.median(), inplace=True)
train_store.PromoInterval.fillna('None', inplace=True)



train_store = train_store.drop(['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear','Promo2SinceWeek',
                     'Promo2SinceYear'], axis=1)


X = train_store.loc[:, train_store.columns != "Sales"]
Y = train_store.Sales



features_to_encode = ['StoreType', 'Assortment', 'PromoInterval', 'StateHoliday']

encoder = LabelEncoder()

for label in features_to_encode:
    X[label] = encoder.fit_transform(X[label])
