# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 12:45:47 2020

@author: Rohith
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
## for feature slection
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
import seaborn as sns


# to visualise al the columns in the dataframe
pd.pandas.set_option('display.max_columns', None)


data=pd.read_csv("E:\\Data Science\\Data_Science_Projects\\Project_5\\data.csv")
data.head()

#To capture dependent and independent variable

## drop dependent feature from dataset
X_train=data.drop(['Id','SalePrice'],axis=1)
Y_train=data[['SalePrice']]

                                                                                           

# Here first, we specify the Lasso Regression model, and I
# select a suitable alpha (equivalent of penalty).
# Bigger the alpha, less features that will be selected.

# selectFromModel object from sklearn, which will select the features which coefficients are non-zero

model=SelectFromModel(Lasso(alpha=0.005, random_state=0))
model.fit(X_train,Y_train)

model.get_support()

#Note here true value indicates more important feature and should be considered while model building

final_selected_features= X_train.columns[(model.get_support())]

print(final_selected_features)

print(len(final_selected_features))

data_new=X_train[final_selected_features]

data_new.head()

data_new.shape

data_new['SalePrice']=data[['SalePrice']]

data_new.head()

#############################Linear Regression#################################
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,accuracy_score
 
x=data_new.iloc[:,0:21]
y=data_new.iloc[:,21]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=1)
LR= LinearRegression()
LR.fit(x_train, y_train)
 
predictions = LR.predict(x_test)

print(predictions)

LR.intercept_
LR.coef_

plt.scatter(y_test,predictions)

sns.distplot((y_test,predictions),bins=50)


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score(LinearRegression(), x, y, cv=cv)


#We can see that in 5 iterations we get a score above 80% all the time. This is pretty good


from sklearn.metrics import r2_score

print(r2_score(y_test,predictions))

#Since r2_score is near to 1 the model which we have used is good model

###############################################################################################



