# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 15:56:42 2016

@author: ogier
"""

import gc as gc
from globalVar import *
from handling import *
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing.data import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.model_selection import StratifiedKFold

import sys

gc.collect()
#Prepare classifier
#clf1= XGBClassifier(learning_rate=0.1,min_child_weight=10,max_depth=10,gamma=1,subsample=0.5,colsample_bytree=0.5,reg_lambda=0,scale_pos_weight=10)
clfa=XGBClassifier()
clfb=RandomForestClassifier(class_weight="balanced")
scorer=make_scorer(matthews_corrcoef)
skf3 = StratifiedKFold(n_splits=2)

parametersa=[
           { "learning_rate" : [0.1] , "min_child_weight" : [5,10], "max_depth" : [3,6,10],
             "gamma" : [0] , "subsample":[0.5,1] , "colsample_bytree" : [0.7], "reg_lambda" : [0,1],
             "scale_pos_weight" : [1,20,100]
            }
            ]
#parametersb=[
#             {"max_features": [None, 20,45, 65],"n_estimators":[20], "min_samples_leaf":[1,35,60], "min_samples_split" : [2,5,10], "max_leaf_nodes":[None, 3,20]}    
#            ]
#           
#           
clf1=GridSearchCV(clfa,parametersa,scoring=scorer,cv=skf3,verbose=1)
#clf2=GridSearchCV(clfb,parametersb,scoring=scorer,cv=skf3,verbose=1)

#Import column filters
dateCols= read_csv("train_date_colnames")["0"].values
catCols= read_csv("train_categorical_colnames")["0"].values
#Import lines filter for Train
uselines= read_csv(linesTrainPath)["0"].values

#Numeric data
print("Importing numeric data")
dataTrain = read_csv(trainPath,dtype=np.float16)
print(dataTrain.shape)
#Extract responses
y=read_csv(trainPath,usecols=["Response"])["Response"]
dataTrain["Response"]=y

#Add date features
print("Importing features")
dataTrain= concat([dataTrain,read_csv(featTrainPath,usecols=["Min","Max","Diff","Evol"],dtype=np.float32)],axis=1)
print(dataTrain)
print(dataTrain.shape)
#Add date data
print("Importing date data")
dataTrain = concat([dataTrain,read_csv(dateTrainPath,dtype=np.float16, usecols= dateCols)],axis=1,copy=False)
gc.collect()
print(dataTrain)
print(dataTrain.shape)
#Add categorical data
print("Importing categorical data")
##dataTrain = concat([dataTrain,read_csv(catTrainPath,dtype=np.int16,usecols=catCols)],axis=1,copy=False)
##print(dataTrain)
##print(dataTrain.shape)
print("Removing useless rows")
dataTrain=dataTrain.ix[uselines].sample(frac=0.3, random_state=0)
gc.collect()
print(dataTrain.shape)

#Drop response and Id
y=dataTrain["Response"]
dataTrain=dataTrain.drop("Response",axis=1)
dataTrain=dataTrain.drop("Id",axis=1)

#Fit classifier than can handle NaN
print("Fitting")
clf1.fit(dataTrain,y)

# Scale data
print("Scaling")
#scaler=StandardScaler
#imputer=Imputer()
#dataTrain=imputer.fit_transform(dataTrain)
#dataTrain=scaler.fit_transform(dataTrain)

#Fit classifier that needs imputation
print("Fitting")
#clf2.fit(dataTrain,y)

print("Fit completed")
#Remove useless data to free RAM
del dataTrain
del y
gc.collect()
#Import numeric data
print("Importing numeric data")
dataTest = read_csv(testPath,dtype=np.float16)

#Isolate ids for end submission
ids=read_csv(testPath,usecols=["Id"])["Id"]
dataTest=dataTest.drop("Id",axis=1)

#Add date features
print("Importing features")
dataTest= concat([dataTest,read_csv(featTestPath,usecols=["Min","Max","Diff","Evol"],dtype=np.float16)],axis=1)
print(dataTest.shape)
#Add date data
print("Importing date data")
dataTest = concat([dataTest,read_csv(dateTestPath,dtype=np.float16, usecols= dateCols)],axis=1,copy=False)
print(dataTest.shape)
#Add categorical data
print("Importing categorical data")
##dataTest = concat([dataTest,read_csv(catTestPath,dtype=np.int16,usecols=catCols)],axis=1,copy=False)
#print(dataTrain.shape)
#Make prediction
print("Predicting")
bounds=[0,200000,400000,600000,800000,1000000,len(ids)]
preds=[]
for i in range(6):
    dtest=dataTest.ix[range(bounds[i],bounds[i+1])]
    y_pred=clf1.predict(dtest)
    del dtest
    preds.append(DataFrame(y_pred))
    gc.collect()
print(preds)
y=concat(preds,axis=0,copy=False)
del dataTest
del preds
print(y.shape)
print("Prediction done")
res=DataFrame(np.nan, index = range(len(ids)), columns = ["Id","Response"])
res["Id"]=ids
res["Response"]=y.values

res.to_csv("submission1.csv",index=False)

sys.exit("Fini")

#Scale
print("Scaling")
dataTest=imputer.transform(dataTest)
dataTest=scaler.transform(dataTest)

preds=[]
for i in range(6):
    dtest=dataTest.ix[range(bounds[i],bounds[i+1])]
    y_pred=clf2.predict(dtest)
    del dtest
    preds.append(DataFrame(y_pred))
    gc.collect()
print(preds)
y=concat(preds,axis=0,copy=False)
##y_pred2=clf2.predict(dataTest)
del dataTest
del preds
print(y.shape)
print("Prediction done")
res=DataFrame(np.nan, index = range(len(ids)), columns = ["Id","Response"])
res["Id"]=ids
res["Response"]=y.values

res.to_csv("submission2.csv",index=False)

##res["Response"]=y_pred2
##res.to_csv("submission2.csv")
