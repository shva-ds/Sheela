# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 23:21:05 2020

@author: Sheela Vatsala
"""
#1.Prepare a prediction model for profit of 50_startups data.

import pandas as pd
import numpy as np

# loading the data
startups1 = pd.read_csv("C:\\Users\\Sheela Vatsala\\Desktop\\360digitmg\\Assignment\\MLR\\50_Startups.csv")

#EDA
startups1.describe()
list(startups1.columns) 
startups1.isna()
startups1.columns
startups1.head()
startups1.info()

# Import LabelEncoder
from sklearn import preprocessing
#creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
#Train data set
startups1["State"]=le.fit_transform(startups1["State"])
str(startups1)# note the categorical columns are not numeric
startups1.columns =['RDSpend', 'Administration', 'MarketingSpend', 'State', 'Profit'] 


#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

# profit
plt.bar(height = startups1.Profit, x = np.arange(1, 51, 1))
plt.hist(startups1.Profit) #histogram
plt.boxplot(startups1.Profit) #boxplot

# RnD
plt.bar(height = startups1.RDSpend, x = np.arange(1, 51, 1))
plt.hist(startups1.RDSpend) #histogram
plt.boxplot(startups1.RDSpend) #boxplot

# Jointplot
import seaborn as sns
sns.jointplot(x=startups1['MarketingSpend'], y=startups1['Profit'])

# Countplot
plt.figure(1, figsize=(16, 10))
sns.countplot(startups1['State'])

# Countplot
plt.figure(1, figsize=(16, 10))
sns.countplot(startups1['Administration'])

# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(startups1.Administration, dist = "norm", plot = pylab)
plt.show()

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(startups1.iloc[:, :])
                             
# Correlation matrix 
startups1.corr()

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
ml1 = smf.ols('Profit ~ RDSpend+Administration+MarketingSpend+State', data = startups1).fit() # regression model

# Summary
ml1.summary()


# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1)
# Studentized Residuals = Residual/standard deviation of residuals
# index 45,49 are showing high influence so we can exclude that entire row

startups1 = startups1.drop(startups1.index[[45,49]])

# Preparing model                  
ml_new = smf.ols('Profit ~ RDSpend+Administration+MarketingSpend+State', data = startups1).fit()    

# Summary
ml_new.summary()


# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
RDSpend = smf.ols('RDSpend ~ Administration + MarketingSpend + State', data = startups1).fit().rsquared  
vif_RDspend = 1/(1 - RDSpend) 

rsq_Administration = smf.ols('Administration ~ RDSpend + MarketingSpend + State', data = startups1).fit().rsquared  
vif_Administration = 1/(1 - rsq_Administration)

rsq_MarketingSpend = smf.ols('MarketingSpend ~ Administration + RDSpend + State', data = startups1).fit().rsquared  
vif_MarketingSpend = 1/(1 - rsq_MarketingSpend) 

rsq_State = smf.ols('State ~ MarketingSpend + Administration + RDSpend', data = startups1).fit().rsquared  
vif_state = 1/(1 - rsq_State) 

# Storing vif values in a data frame
d1 = {'Variables':['RDSpend', 'Administration', 'MarketingSpend', 'State'], 'VIF':[vif_RDspend, vif_Administration, vif_MarketingSpend, vif_state]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# no collinearity found

# Final model
final_ml = smf.ols('Profit ~ RDSpend+Administration+MarketingSpend+State', data = startups1).fit()
final_ml.summary() 

# Prediction
pred = final_ml.predict(startups1)

# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = startups1.Profit, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
start_train, start_test = train_test_split(startups1, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols('Profit ~ RDSpend+Administration+MarketingSpend+State', data = start_train).fit()

# prediction on test data set 
test_pred = model_train.predict(start_test)

# test residual values 
#test_resid = test_pred - start_test.Profit
test_resid=start_test.Profit-test_pred
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


# train_data prediction
train_pred = model_train.predict(start_train)

# train residual values 
#train_resid  = train_pred - start_train.Profit
train_resid  = start_train.Profit- train_pred

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse



###########################################################
#2.Computer data predict sales
# Multilinear Regression
import pandas as pd
import numpy as np

# loading the data
compdata1 = pd.read_csv("C:\\Users\\Sheela Vatsala\\Desktop\\360digitmg\\Assignment\\MLR\\Computer_Data.csv")

#EDA

#remove first column,sl.no or X
compdata1=compdata1.iloc[:,1:11]
compdata1.describe()
list(compdata1.columns) 
compdata1.isna()

compdata1.columns
compdata1.head()
compdata1.info()
# Import LabelEncoder
from sklearn import preprocessing
#creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
#Train data set
compdata1["cd"]=le.fit_transform(compdata1["cd"])
compdata1["multi"]=le.fit_transform(compdata1["multi"])
compdata1["premium"]=le.fit_transform(compdata1["premium"])

str(compdata1)# note the categorical columns are not numeric

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

# price
plt.bar(height = compdata1.price, x = np.arange(1, 6260, 1))
plt.hist(compdata1.price) #histogram
plt.boxplot(compdata1.price) #boxplot

# speed
plt.bar(height = compdata1.speed, x = np.arange(1, 6260, 1))
plt.hist(compdata1.speed) #histogram
plt.boxplot(compdata1.speed) #boxplot

# Jointplot
import seaborn as sns
sns.jointplot(x=compdata1['ram'], y=compdata1['price'])

# Countplot
plt.figure(1, figsize=(16, 10))
sns.countplot(compdata1['ram'])

# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(compdata1.screen, dist = "norm", plot = pylab)
plt.show()

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(compdata1.iloc[:, :])
                             
# Correlation matrix 
compdata1.corr()

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
ml1 = smf.ols('price ~ speed+hd+ram+screen+cd+multi+premium+ads', data = compdata1).fit() # regression model

# Summary
ml1.summary()


# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1)
# Studentized Residuals = Residual/standard deviation of residuals
# index 46,50 are showing high influence so we can exclude that entire row

compdata1 = compdata1.drop(compdata1.index[[1441,1701]])

# Preparing model                  
ml_new = smf.ols('price ~ speed+hd+ram+screen+cd+multi+premium+ads', data = compdata1).fit()    

# Summary
ml_new.summary()


# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
rsq_speed = smf.ols('speed ~ hd + ram + screen + ads', data = compdata1).fit().rsquared  
vif_speed = 1/(1 - rsq_speed) 

rsq_hd = smf.ols('hd ~ speed + ram + screen + ads', data = compdata1).fit().rsquared  
vif_hd = 1/(1 - rsq_hd)

rsq_ram = smf.ols('ram ~ hd + speed + screen + ads', data = compdata1).fit().rsquared  
vif_ram = 1/(1 - rsq_ram) 

rsq_screen = smf.ols('screen ~ ram + hd + speed + ads', data = compdata1).fit().rsquared  
vif_screen = 1/(1 - rsq_screen) 

# Storing vif values in a data frame
d1 = {'Variables':['speed', 'hd', 'ram', 'screen'], 'VIF':[vif_speed, vif_hd, vif_ram, vif_screen]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# no collinearity found

# Final model
final_ml = smf.ols('price ~ speed+hd+ram+screen+cd+multi+premium+ads', data = compdata1).fit()
final_ml.summary() 

# Prediction
pred = final_ml.predict(compdata1)

# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = compdata1.price, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
comp_train, comp_test = train_test_split(compdata1, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols('price ~ speed+hd+ram+screen+cd+multi+premium+ads+trend', data = comp_train).fit()

# prediction on test data set 
test_pred = model_train.predict(comp_test)

# test residual values 
test_resid = test_pred - comp_test.price
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


# train_data prediction
train_pred = model_train.predict(comp_train)

# train residual values 
train_resid  = train_pred - comp_train.price
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse

###################################################################



#3.prediction model for predicting Price of corolla

import pandas as pd
import numpy as np

# loading the data
corolla1 = pd.read_csv("C:\\Users\\Sheela Vatsala\\Desktop\\360digitmg\\Assignment\\MLR\\ToyotaCorolla.csv",encoding= 'unicode_escape')


#remove first column,sl.no or X
corolla=corolla1[['Price','Age_08_04','KM','HP','cc','Doors','Gears','Quarterly_Tax','Weight']]

#EDA

corolla.describe()
corolla.isna()
list(corolla.columns) 
#or 
corolla.columns
corolla.head()
corolla.info()


#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

# price
plt.bar(height = corolla.Price, x = np.arange(1, 1437, 1))
plt.hist(corolla.Price) #histogram
plt.boxplot(corolla.Price) #boxplot

# age
plt.bar(height = corolla.Age_08_04, x = np.arange(1, 1437, 1))
plt.hist(corolla.Age_08_04) #histogram
plt.boxplot(corolla.Age_08_04) #boxplot

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(corolla.iloc[:, :])
                             
# Correlation matrix 
corolla.corr()


# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
ml1 = smf.ols('Price ~ Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight', data = corolla).fit() # regression model

# Summary
ml1.summary()


# Prediction
pred = ml1.predict(corolla)

import statsmodels.api as sm

# Q-Q plot
res = ml1.resid
sm.qqplot(res)
plt.show()

from scipy import stats
import pylab

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = corolla.Price, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(ml1)


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
C_train, C_test = train_test_split(corolla, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols('Price ~ Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight', data = C_train).fit()

# prediction on test data set 
test_pred = model_train.predict(C_test)

# test residual values 
test_resid = test_pred - C_test.Price
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


# train_data prediction
train_pred = model_train.predict(C_train)

# train residual values 
train_resid  = train_pred - C_train.Price
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse
