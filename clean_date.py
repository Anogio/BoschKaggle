# -*- coding: utf-8 -*-
"""
Created on Sat Nov 05 13:34:48 2016

@author: JLM
"""
#%% Importe les fichiers de base
import numpy as np
from pandas import *
from globalVar import *

filepath_date =dateTrainPath
filepath_numeric = trainPath

N = 100

data= read_csv(filepath_numeric,nrows=N)
y_numeric=data["Response"]
X_numeric=data.drop("Response",axis=1)
X_numeric_id=data["Id"]
X_numeric=data.drop("Id", axis=1)

data= read_csv(filepath_date,nrows=N)
X_date_id=data["Id"]
X_date=data.drop("Id",axis=1)

#%% Nettoie le tableau des dates en supprimant les dates doublons
date_feature_column = list(X_date.columns)
numeric_feature_column = list(X_numeric.columns)
numeric_feature_column.remove('Response')

i = 0
for name in date_feature_column:
    position = name.find('D') +1
    numero_date = int(name[position:])
    
    position = numeric_feature_column[i].find('F')+1
    numero_feature = int(numeric_feature_column[i][position:])
    
    if numero_date < numero_feature:
        del X_date[name]
    else:
        i = i +1
    
date_feature_column = list(X_date.columns)
 
date_feature_number = []
for name in date_feature_column:
    position = name.find('D') +1
    date_feature_number.append(int(name[position:])-1)
    
numeric_feature_number = []
for name in numeric_feature_column:
    position = name.find('F') +1
    numeric_feature_number.append(int(name[position:]))

 
numeric_feature_number = np.asarray(numeric_feature_number)
date_feature_number = np.asarray(date_feature_number) 

print(numeric_feature_number.shape)
print(date_feature_number.shape)
#Soit la date c'est +1 soit la date c'est +2
#On dirait qu'il y a deux catégories de mesure de dates...
print (numeric_feature_number-date_feature_number)
print (np.min(numeric_feature_number-date_feature_number))

names =DataFrame(X_date.columns.values)
print(type(names))
print(names)
names.to_csv("train_date_colnames")

names2=DataFrame(X_numeric.columns.values)
names2.to_csv("train_numeric_colnames")

