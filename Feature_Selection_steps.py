# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 02:45:32 2020

@author: Rohith.N
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
## for feature slection
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

# to visualise al the columns in the dataframe
pd.pandas.set_option('display.max_columns', None)


data=pd.read_csv("E:\\Data Science\\Data_Science_Projects\\Project_5\\X_train.csv")
data.head()

#To capture dependent and independent variable

## drop dependent feature from dataset
X_train=data.drop(['Id','SalePrice'],axis=1)
y_train=data[['SalePrice']]


# Here first, we specify the Lasso Regression model, and I
# select a suitable alpha (equivalent of penalty).
# Bigger the alpha, less features that will be selected.

# selectFromModel object from sklearn, which will select the features which coefficients are non-zero

model=SelectFromModel(Lasso(alpha=0.005, random_state=0))
model.fit(X_train,y_train)

model.get_support()

#Note here true value indicates more important feature and should be considered while model building

final_selected_features= X_train.columns[(model.get_support())]

print(final_selected_features)

X_train=X_train[final_selected_features]

X_train.head()






















