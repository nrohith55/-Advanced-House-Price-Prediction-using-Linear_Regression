# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 01:22:52 2020

@author:Rohith.N
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv("E:\\Data Science\\Data_Science_Projects\\Project_5\\train.csv")

#Now we are finding the missing values or null values for categorical value

features_nan=[feature for feature in data.columns if data[feature].isnull().sum()>1 and data[feature].dtypes=='O']

for feature in features_nan:
    print(feature,np.round(data[feature].isnull().mean(),4))


#Now replacing the Null values with new label('Missing')
    
def replace_cat_feature(data,features_nan):
    data=data.copy()
    data[features_nan]=data[features_nan].fillna('Missing')
    return data

data=replace_cat_feature(data, features_nan)

data[features_nan].isnull().sum()    
    

data.head()

#Similarly we are fining the numerical nan values

numerical_with_nan=[feature for feature in data.columns if data[feature].isnull().sum()>1 and data[feature].dtypes!='O']

for feature in numerical_with_nan:
    print(feature,np.round(data[feature].isnull().mean(),4))

#Replacing the numeric missing values:
    
    for feature in numerical_with_nan:
        data=data.copy()
        median_value=data[feature].median()
        data[feature].fillna(median_value,inplace=True)
    
    data[numerical_with_nan].isnull().sum()


data.head()

data.isnull().sum()    

#We need to conver Temporal Variables(Date Time variables)

for feature in ['YearBuilt','YearRemodAdd','GarageYrBlt']:
    data[feature]=data['YrSold']-data[feature]


data[['YearBuilt','YearRemodAdd','GarageYrBlt']].head()

#Since the numerical variables are skewed we will perform log normal distribution


num_features=['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']

for feature in num_features:
    
    data[feature]=np.log(data[feature])

data.head()


#Now handling Categorical features

categorical_feature=[feature for feature in data.columns if data[feature].dtypes =='O']

print(categorical_feature)

for feature in categorical_feature:
    temp=data.groupby(feature)['SalePrice'].count()/len(data)
    temp_df=temp[temp>0.01].index
    data[feature]=np.where(data[feature].isin(temp_df),data[feature],'Rare_var')


data.head()


for feature in categorical_feature:
    labels_ordered=data.groupby([feature])['SalePrice'].mean().sort_values().index
    labels_ordered={k:i for i,k in enumerate(labels_ordered,0)}
    data[feature]=data[feature].map(labels_ordered)


data.head()

#######################Feature Scaling##########################
    
    
new_data=[feature for feature in data.columns if feature not in ['Id','SalePrice']]

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(data[new_data])    
scaler.transform(data[new_data])    

data = pd.concat([data[['Id', 'SalePrice']].reset_index(drop=True),pd.DataFrame(scaler.transform(data[new_data]), columns=new_data)],axis=1)    

data.to_csv('data.csv',index=False)    




























    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    