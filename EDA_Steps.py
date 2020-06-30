# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 19:48:56 2020

@author: Rohith

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

###############################Exploratory Data Analysis#######################################

df=pd.read_csv("E:\\Data Science\\Data_Science_Projects\\Project_5\\house_price.csv")

pd.pandas.set_option('display.max_columns',None)

print(df.shape)#To chechk how many number of rows and columns

print(df.columns)#To display the name of all columns in the data set

print(df.head())#To display the top 5 rows of the data set

print(df.tail())#To display the last 5 rows of the data set

print(df.dtypes)#To find the data type in the data set


NaN=print(df.isnull().sum())

#Here we will check the percentage of NaN values in each fetures

features_with_na=[features for features in df.columns if df[features].isnull().sum()>1]

for feature in features_with_na:
    print(feature,np.round(df[feature].isnull().mean(),4),"%missing value")
    

#Since there are many missing values we have to find the relationship of missing values with dependent variable
    
for feature in features_with_na:
    
    data=df.copy()
# let's make a variable that indicates 1 if the observation was missing or zero otherwise
    data[feature]=np.where(df[feature].isnull(),1,0)
    
    data.groupby(feature)['SalePrice'].median().plt.bar()
    plt.title(feature)
    plt.show()
    
    
print(len(df.Id))#Length of Id column in the data set


#Now our task is to find the list of numeric variable in the data set


numeric_features=[features for features in df.columns if df[features].dtypes != 'O']

print('Number of numeric features:' ,len(numeric_features))
print(df[numeric_features].head())



#There are temporal variables such as (Date time variables)

year_feature=[feature for feature in numeric_features if 'Yr' in feature or 'Year' in feature]

print(year_feature)

#Now explore the content of Year 

for feature in year_feature:
    print(feature,df[feature].unique())
    
#Now we will chck is thr any relationship between Year the house is sold and Sales price
    
df.groupby('YrSold')['SalePrice'].median().plot()
plt.xlabel('Year Sold')
plt.ylabel('Median House Price')
plt.title("House Price vs YearSold")
 

year_feature

# To plot scatter plot

for feature in year_feature:
    
        data=df.copy()
        plt.scatter(df[feature],df['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.show()



#Numeric variables are of 2 types 1)Discrete type and 2)Continuos type
        
discrete_feature=[feature for feature in numeric_features if len(df[feature].unique())<25 and feature not in year_feature+['Id']]
        
print('discrete_value_count :',len(discrete_feature))

discrete_feature

df[discrete_feature].head()

#To plot the Bar plot for all the discrete values

for feature in discrete_feature:
    data=df.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()
    
    
    ###To find the continuous values in Numeric_Variables:
    #There is a relationship between number and Sales
  
    
 continuous_feature=[feature for feature in numeric_features if feature not in discrete_feature+year_feature+['Id']]
 print ('Len of Continuos feature :',len(continuous_feature))


for feature in continuous_feature:
    data=df.copy()
    data[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.title(feature)
    plt.show()

    

    ## We will be using logarithmic transformation


for feature in continuous_feature:
    data=df.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature]=np.log(data[feature])
        data['SalePrice']=np.log(data['SalePrice'])
        plt.scatter(data[feature],data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalesPrice')
        plt.title(feature)
        plt.show()
        
        
 for feature in continuous_feature:
     data=df.copy()
     data.boxplot(column=feature)
     plt.ylabel(feature)
     plt.title(feature)
     plt.show()    
        
        
        
 #To find categorical_variable    
     
     
categorical_features=[feature for feature in df.columns if df[feature].dtypes == 'O']

print("Number of categorical feature : " , len(categorical_features))        


df[categorical_features].head()

for feature in categorical_features:
    print("The feature is {} and number of categories are {}".format(features,len(df[feature].unique())))

for feature in categorical_features:
    data=df.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()
    
    





        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    








    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    































    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    