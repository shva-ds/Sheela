# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 12:00:03 2021

@author: Sheela Vatsala
"""


from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=load_boston()
df
dataset=pd.DataFrame(df.data)
dataset.columns= df.feature_names
dataset.head()
dir(df.target)
df.target.shape
dataset["price"]=df.target
dataset.info()
dataset.head()

#split the dataset into X, y 
X=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]

#perform CV and then perform L1 

#Perform Linear regression using cross validation as 5
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
model=cross_val_score(LR, X,y,scoring='neg_mean_squared_error',cv=5)
mse=np.mean(model)
print(mse) #-37.13180746769895

#perform Lasso regression -regularization with GridsearchCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import Lasso
from sklearn.model_selection import GridSearchCV
lasso=Lasso()
parameters={'alpha':[0.01,0.1,1,5,10,100]}
Lasso_reg=GridSearchCV(lasso, param_grid=parameters,scoring='neg_mean_squared_error',cv=5)
Lasso_reg.fit(X,y)
Lasso_reg.best_params_
Lasso_reg.best_score_

#Perform Ridge regression - regularization with GridsearchCV
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
ridge=Ridge()
parameters={'alpha':[0.01,0.1,1,5,10,100,105,150,200]}
ridge_reg=GridSearchCV(ridge, param_grid=parameters,scoring='neg_mean_squared_error',cv=5)
ridge_reg.fit(X,y)
ridge_reg.best_params_
ridge_reg.best_score_ #-29.75361723414266


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.25,random_state=11)

lasso_pred=Lasso_reg.predict(X_test)
ridge_pred=ridge_reg.predict(X_test)

rmse_lasso= np.sqrt(((y_test - lasso_pred)**2).mean())
rmse_ridge= np.sqrt(((y_test - ridge_pred)**2).mean())

import seaborn as sns
sns.distplot(y_test-lasso_pred)

sns.distplot(y_test-ridge_pred)
