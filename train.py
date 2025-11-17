#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mutual_info_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier 
from IPython.display import display
import pickle


#Parameters

n_estimators = 100,
min_samples_leaf = 10, 
max_depth = 10, 


#Data Preparation



df = pd.read_csv('Dataset.csv', sep=';')

df.columns = df.columns.str.lower().str.replace(' ','_')


df = df.copy()
for i in range(0, df.shape[0]):
    
    if df['customer_rating'][i] == 5:
        df.loc[i, 'customer_rating'] = 'very_high'
    
    elif df['customer_rating'][i] == 4:
        df.loc[i, 'customer_rating'] = 'high'
    
    elif df['customer_rating'][i] == 3:
        df.loc[i, 'customer_rating'] = 'medium'
    
    elif df['customer_rating'][i] == 2:
        df.loc[i, 'customer_rating'] = 'low'
    
    elif df['customer_rating'][i] == 1:
        df.loc[i, 'customer_rating'] = 'very_low'


#Setting-up Validation Framework



categorical = (['warehouse_block', 'mode_of_shipment',
               'customer_rating', 'product_importance'])
numerical = (['customer_care_calls', 'cost_of_the_product',
              'prior_purchases', 'discount_offered', 'weight'])

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_full_train = df_full_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_full_train = df_full_train['on_time_delivery'].values
x_full_train = df_full_train[numerical + categorical]
df_train, df_val, y_train, y_val = train_test_split(x_full_train, y_full_train, test_size=0.25, random_state=42)
df_test_x = df_test[numerical + categorical]
y_test = df_test['on_time_delivery'].values


#Training the Model



def train(df, y):
    dict_x_full = df[categorical + numerical].to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    x_full = dv.fit_transform(dict_x_full)
    
    model = RandomForestClassifier(
                                    n_estimators,
                                    min_samples_leaf, 
                                    max_depth, 
                                    n_jobs=-1,
                                    random_state=42)
    
    model.fit(x_full, y)
    
    return dv, model


def predict(df, dv, model):
    dict_x = df[categorical + numerical].to_dict(orient='records')
    X = dv.transform(dict_x)

    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


#testing the Model



dv, model = train(x_full_train, y_full_train)

y_pred = predict(df_test_x, dv, model)

auc = roc_auc_score(y_test, y_pred)
print('auc = %.3f' % auc)


#Saving the Model



with open('delivery-model.bin', 'wb') as f_out:
    pickle.dump((dv, model), f_out)

