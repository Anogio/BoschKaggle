# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 15:56:42 2016

@author: ogier
"""

from globalVar import trainPath ,dateTrainPath,catTrainPath,testPath,dateTestPath,catTestPath,featTrainPath, linesTrainPath , featStationTrainPath , featTestPath , featStationTestPath, featNames,newFeaturesTrainPath,newFeaturesTestPath 
from handling import writeDict

import gc as gc
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing.data import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics.scorer import make_scorer
from sklearn.metrics.classification import matthews_corrcoef
from random import randint
from time import time, gmtime, strftime
from pandas import read_csv, concat, DataFrame

t0=time()

bestParamXGB1 = {'learning_rate': 0.05, 'max_depth': 6, 'colsample_bytree': 0.7, 'min_child_weight': 1, 'subsample': 0.5, 'gamma': 0.5, 'reg_lambda': 0, 'scale_pos_weight': 3}
bestParamXGB2 = {'learning_rate': 0.05, 'max_depth': 6, 'colsample_bytree': 0.7, 'min_child_weight': 1, 'subsample': 1, 'gamma': 0.5, 'reg_lambda': 0, 'scale_pos_weight': 3}
bestParamRF={}
gridSearchXGB1 = False
gridSearchXGB2 = False
gridSearchRF = False

frac=0.2
rst=randint(0,2000)

gc.collect()
#Prepare classifier
#clf1= XGBClassifier(learning_rate=0.1,min_child_weight=10,max_depth=10,gamma=1,subsample=0.5,colsample_bytree=0.5,reg_lambda=0,scale_pos_weight=10)
clfa=XGBClassifier()
clfb=RandomForestClassifier(class_weight="balanced")
scorer=make_scorer(matthews_corrcoef)
skf = StratifiedKFold(n_splits=3)
skf5= StratifiedKFold(n_splits=3)


parametersa=[
           { "learning_rate" : [0.05] , "min_child_weight" : [1,3], "max_depth" : [3,6,10],
             "gamma" : [0,0.5] , "subsample":[0.5,1] , "colsample_bytree" : [0.7], "reg_lambda" : [0,0.5],
             "scale_pos_weight" : [0.5,1,3]
             }
            ]
parametersb=[
             {"max_features": [None, 'sqrt',"log2"],"n_estimators":[20], "min_samples_leaf":[1,5,30], "max_leaf_nodes":[None, 3,20]}    
            ]
           
#parametersa= [{"learning_rate" : [0.1]} ]
#parametersb= [{"n_estimators":[10], "max_features":['log2']}]    
           
writeDict(str(parametersb[0]),"testDict.txt")

clf1=GridSearchCV(clfa,parametersa,scoring=scorer,cv=skf,verbose=2)
clf2=GridSearchCV(clfb,parametersb,scoring=scorer,cv=skf5,verbose=2,n_jobs=-1,pre_dispatch='n_jobs')

#Import column filters
dateCols= read_csv("train_date_colnames")["0"].values
catCols= read_csv("train_categorical_colnames")["0"].values
#Import lines filter for Train
uselines= read_csv(linesTrainPath)["0"].values

delta=gmtime(time() - t0)
tstr=strftime('%H:%M:%S',delta)
print("Time since beginning:%s" % tstr)

# Numeric data
print("Importing numeric data")
dataTrain = read_csv(trainPath,dtype=np.float16)
ids=read_csv(trainPath,usecols=["Id"])["Id"]
dataTrain["Id"]=ids
del ids
dataTrain=dataTrain.sample(frac=frac,random_state=rst)
print(dataTrain.shape)
#Extract responses
y=read_csv(trainPath,usecols=["Response"]).sample(frac=frac,random_state=rst)["Response"]


n_num_feat=dataTrain.shape[1]-2

#Add date features
print("Importing features")
dataTrain= concat([dataTrain,read_csv(featTrainPath,usecols=["Min","Max","Diff","Evol"]).sample(frac=frac,random_state=rst)],axis=1)
print(dataTrain.shape)
n_new_feat=dataTrain.shape[1]-2-n_num_feat

dataTrain= concat([dataTrain,read_csv(featStationTrainPath,usecols=["NbFeats", "Station1","StationLast","NbStations","PathDupe","PathNoDupe"]).sample(frac=frac,random_state=rst)],axis=1)
print(dataTrain.shape)
n_new_feat += 6

for name in featNames :
    dataTrain= concat([dataTrain,read_csv(newFeaturesTrainPath+name+".csv",usecols=[name])],axis=1)
    n_new_feat+=1

#Add date data
print("Importing date data")
dataTrain = concat([dataTrain,read_csv(dateTrainPath,dtype=np.float16, usecols= dateCols).sample(frac=frac,random_state=rst)],axis=1,copy=False)
gc.collect()
print(dataTrain.shape)
n_date_feat= dataTrain.shape[1]-2-n_num_feat - n_new_feat
#Add categorical data
print("Importing categorical data")
dataTrain = concat([dataTrain,read_csv(catTrainPath,dtype=np.float16,usecols=catCols).replace([np.inf, -np.inf], -1).sample(frac=frac,random_state=rst)],axis=1,copy=False)
n_cat_feat = dataTrain.shape[1]-2-n_num_feat - n_new_feat - n_date_feat


print("Number of columns for each category:")
print([n_num_feat,n_new_feat,n_date_feat,n_cat_feat])

print(dataTrain)
print(dataTrain.shape)

gc.collect()

#Drop response and Id
dataTrain=dataTrain.drop("Id",axis=1)
dataTrain=dataTrain.drop("Response",axis=1)

print("y size:")
print(y.shape)
print("Number of positives:")
print(len(np.where(y.values==1)[0]))

delta=gmtime(time() - t0)
tstr=strftime('%H:%M:%S',delta)
print("Time since beginning:%s" % tstr)

if(gridSearchXGB1):
    #Fit classifier to find best columns
    print("Fitting")
    clf1.fit(dataTrain,y)
    bestPar= clf1.best_params_
    writeDict(str(bestPar),"dict1XGB.txt")
    print("Best parameters:%s" % bestPar)
    
else:
    bestPar = bestParamXGB1
    print("Using those parameters for feature selection:")
    print(str(bestPar))

clf1= XGBClassifier(**bestPar)
clf1.fit(dataTrain,y)

delta=gmtime(time() - t0)
tstr=strftime('%H:%M:%S',delta)
print("Time since beginning:%s" % tstr)

important_indices = np.where(clf1.feature_importances_>0)[0]
scores = clf1.feature_importances_[important_indices]
order = np.argsort(scores)
print(dataTrain.columns.values[important_indices])
print("Scores:")
print(scores)
print("Sorted importances:")
print(dataTrain.columns.values[important_indices][order])
DataFrame(dataTrain.columns.values[important_indices]).to_csv("important_columns.csv")
print("Number of important columns:")
print(len(important_indices))

allCols= dataTrain.columns.values

del dataTrain
gc.collect()

numCols = allCols[important_indices[important_indices<n_num_feat]]
dateCols = allCols[important_indices[(important_indices<n_num_feat + n_new_feat + n_date_feat) & (important_indices>= n_num_feat + n_new_feat) ]]
catCols = allCols[important_indices[important_indices>= n_num_feat + n_new_feat + n_date_feat]]

print("Best columns selected. Now fitting for prediction")
lTot=len(dateCols)+len(numCols)+len(catCols) + 4 + 6
print("Selecting %d columns" % lTot)

####### Importing again just the important columns and refitting ######
#Numeric data
print("Importing numeric data")
dataTrain = read_csv(trainPath,dtype=np.float32,usecols=numCols)
print(dataTrain.shape)
#Extract responses
y=read_csv(trainPath,usecols=["Response"])["Response"]

n_num_feat=dataTrain.shape[1]-2

#Add date features
print("Importing features")
dataTrain= concat([dataTrain,read_csv(featTrainPath,usecols=["Min","Max","Diff","Evol"])],axis=1)
print(dataTrain.shape)
n_new_feat=dataTrain.shape[1]-2-n_num_feat

dataTrain= concat([dataTrain,read_csv(featStationTrainPath,usecols=["NbFeats", "Station1","StationLast","NbStations","PathDupe","PathNoDupe"])],axis=1)
print(dataTrain.shape)
n_new_feat +=6

for name in featNames :
    dataTrain= concat([dataTrain,read_csv(newFeaturesTrainPath+name+".csv",usecols=[name])],axis=1)
    n_new_feat+=1
        

#Add date data
print("Importing date data")
dataTrain = concat([dataTrain,read_csv(dateTrainPath,dtype=np.float32, usecols= dateCols)],axis=1,copy=False)
gc.collect()
print(dataTrain.shape)
n_date_feat= dataTrain.shape[1]-2-n_num_feat - n_new_feat
#Add categorical data
print("Importing categorical data")
dataTrain = concat([dataTrain,read_csv(catTrainPath,usecols=catCols).replace([np.inf, -np.inf], -1)],axis=1,copy=False)
n_cat_feat = dataTrain.shape[1]-2-n_num_feat - n_new_feat - n_date_feat

print(dataTrain.shape)
print("Number of columns for each category:")
print([n_num_feat,n_new_feat,n_date_feat,n_cat_feat])

print("Used columns")
print(dataTrain.columns.values)
DataFrame(dataTrain.columns.values).to_csv("used_columns_check.csv")

delta=gmtime(time() - t0)
tstr=strftime('%H:%M:%S',delta)
print("Time since beginning:%s" % tstr)

print("Fitting")
if(gridSearchXGB2):
    clf1=GridSearchCV(clfa,parametersa,scoring=scorer,cv=skf5,verbose=2)
    
    clf1.fit(dataTrain,y)
    bestPar2= clf1.best_params_
    writeDict(str(bestPar2),"dict2XGB.txt")
else : 
    clf1 = XGBClassifier(**bestParamXGB2)
    clf1.fit(dataTrain,y)



delta=gmtime(time() - t0)
tstr=strftime('%H:%M:%S',delta)
print("Time since beginning:%s" % tstr)

# Scale data
print("Scaling")
scaler=StandardScaler()
imputer=Imputer()
dataTrain=imputer.fit_transform(dataTrain)
dataTrain=scaler.fit_transform(dataTrain)

#Fit classifier that needs imputation
print("Fitting")
if(gridSearchRF):
    clf2.fit(dataTrain,y)
    bestPar2= clf2.best_params_
    writeDict(str(bestPar2),"dictRF.txt")
else:
    clf2 = XGBClassifier(**bestParamRF)
    clf2.fit(dataTrain,y)

print("Fit completed")

delta=gmtime(time() - t0)
tstr=strftime('%H:%M:%S',delta)
print("Time since beginning:%s" % tstr)

#Remove useless data to free up RAM
del dataTrain
del y
gc.collect()
#Import numeric data
print("Importing numeric data")
dataTest = read_csv(testPath,dtype=np.float16,usecols=numCols)
print(dataTest.shape)
#Isolate ids for end submission
ids=read_csv(testPath,usecols=["Id"])["Id"]

#Add date features
print("Importing features")
dataTest= concat([dataTest,read_csv(featTestPath,usecols=["Min","Max","Diff","Evol"])],axis=1)
print(dataTest.shape)

dataTest= concat([dataTest,read_csv(featStationTestPath,usecols=["NbFeats", "Station1","StationLast","NbStations","PathDupe","PathNoDupe"])],axis=1)
print(dataTest.shape)

for name in featNames :
    dataTest= concat([dataTest,read_csv(newFeaturesTestPath+name+".csv",usecols=[name])],axis=1)
    n_new_feat+=1
#Add date data
print("Importing date data")
dataTest = concat([dataTest,read_csv(dateTestPath,dtype=np.float16, usecols= dateCols)],axis=1,copy=False)
print(dataTest.shape)
#Add categorical data
print("Importing categorical data")
dataTest = concat([dataTest,read_csv(catTestPath,usecols=catCols).replace([np.inf, -np.inf], -1)],axis=1,copy=False)
print(dataTest.shape)

delta=gmtime(time() - t0)
tstr=strftime('%H:%M:%S',delta)
print("Time since beginning:%s" % tstr)

#Make prediction
print("Predicting")

"""
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
"""
y=DataFrame(clf1.predict(dataTest))
print("Prediction done")

res=DataFrame(np.nan, index = range(len(ids)), columns = ["Id","Response"])
res["Id"]=ids
res["Response"]=y.values

res.to_csv("submission1.csv",index=False)


#Scale
print("Scaling")
dataTest=imputer.transform(dataTest)
dataTest=scaler.transform(dataTest)

print("Predicting")
"""
preds=[]
for i in range(6):
    dtest=dataTest.ix[range(bounds[i],bounds[i+1])]
    y_pred=clf2.predict(dtest)
    del dtest
    preds.append(DataFrame(y_pred))
    gc.collect()
print(preds)
y=concat(preds,axis=0,copy=False)

del dataTest
del preds
print(y.shape)
"""
y=DataFrame(clf2.predict(dataTest))
print("Prediction done")

res=DataFrame(np.nan, index = range(len(ids)), columns = ["Id","Response"])
res["Id"]=ids
res["Response"]=y.values

res.to_csv("submission2.csv",index=False)

print("Finished")

delta=gmtime(time() - t0)
tstr=strftime('%H:%M:%S',delta)
print("Time since beginning:%s" % tstr)
