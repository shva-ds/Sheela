# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 15:09:33 2021

@author: Sheela Vatsala
"""

## importing models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import  AdaBoostClassifier
import pandas as  pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# import os 
# os.listdir()

df=pd.read_csv('heart.csv')
rows=df.shape[0]
columns=df.shape[1]

print ( rows,columns )
## input Variables = x
## output variables = y 

y=df['target']

x=df.drop('target',axis=1)
x

df.head()

## input Variables = x
## output variables = y 

## test and Train Split

X_train,X_test,y_train, y_test = train_test_split( x,y,test_size=0.25,random_state=2)


log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()
nbg_clf = GaussianNB()

for clf in (log_clf, nbg_clf, rnd_clf, svm_clf):
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

from sklearn.svm import SVC
svc_clf= SVC( probability =True,kernel='linear')


### AdaBoostClassifier with SVM

boost = AdaBoostClassifier( base_estimator=svc_clf, n_estimators=500,algorithm='SAMME',learning_rate=.5)

boost.fit(X_train, y_train)

y_pred = boost.predict(X_test)

print("Accuracy ", accuracy_score(y_test, y_pred))


## Training Score

boost.score(X_train,y_train)

## Testing Score

boost.score(X_test,y_test)

### AdaBoostClassifier with Decision Tree


boost = AdaBoostClassifier( base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=10,algorithm='SAMME',learning_rate=0.5)

boost.fit(X_train, y_train)

y_pred = boost.predict(X_test)

print("Accuracy ", accuracy_score(y_test, y_pred))

## Training Score

boost.score(X_train,y_train)

## Testing Score

boost.score(X_test,y_test)

from sklearn.model_selection import GridSearchCV

param_grid={
   
    'learning_rate':[1,0.5,0.1,0.01,0.001,1.5,1.2],
    'algorithm':['SAMME','SAMME.R'],
    'n_estimators':[10,50,100,200]
    
}

grid= GridSearchCV(AdaBoostClassifier(),param_grid)


grid.fit(X_train,y_train)

# To  find the parameters givingmaximum accuracy
grid.best_params_

#RandomizedsearchCV
from sklearn.model_selection import RandomizedSearchCV

randomgrid= RandomizedSearchCV(AdaBoostClassifier(),param_grid)
randomgrid.fit(X_train,y_train)

# To  find the parameters givingmaximum accuracy
randomgrid.best_params_

