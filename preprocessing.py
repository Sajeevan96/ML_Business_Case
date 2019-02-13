import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Read data from csv files

def preprocessing(path):
    train = pd.read_csv(os.path.join(path, "train.csv"))
    store = pd.read_csv(os.path.join(path, "store.csv"))

    # Join the 2 csv files on the Store feature
    train_store = pd.merge(train, store, how='inner', on="Store")

    del(train)
    del(store)

    # Replace the 0 values by '0' as StateHoliday is a categorical feature (with string values).
    train_store.StateHoliday.replace(0, '0', inplace=True)

    # Convert Data feature from object to datetime type.
    train_store['Date'] = pd.to_datetime(train_store.Date,
               format='%Y-%m-%d', errors='coerce')
    
    # Drop 0 labels
    train_store = train_store[train_store.Open != 0]
    
    # Complete the CompetitionDistance column values by the median value 
    # (replacing NaN values by the median value).
    train_store.CompetitionDistance.fillna(train_store.CompetitionDistance.median(), inplace=True)

    # Replace the NaN values by 'None'. It will help us to do label encoding below.
    train_store.PromoInterval.fillna('None', inplace=True)
    
     # Replace the NaN values by 0, 0 will mean no competition yet.
    train_store.CompetitionOpenSinceYear.fillna(0, inplace=True)
    train_store.CompetitionOpenSinceMonth.fillna(0, inplace=True)
    train_store.Promo2SinceWeek.fillna(0, inplace=True)
    train_store.Promo2SinceYear.fillna(0, inplace=True)
     
    # Get Features and Labels
    #X = train_store.loc[:, train_store.columns != "Sales"]
    #Y = train_store.Sales


    ##### Label Encoding #####
    features_to_encode = ['StoreType', 'Assortment', 'PromoInterval', 'StateHoliday']
    
    encoder = LabelEncoder()

    for label in features_to_encode:
        train_store[label] = encoder.fit_transform(train_store[label])
        
    return train_store
