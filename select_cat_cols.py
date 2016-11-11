# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 18:12:30 2016

@author: ogier
"""

from pandas import * 
from globalVar import *

thresh=50


head = read_csv(catTrainPath,nrows=1)
colnames=list(head.drop("Id",axis=1).columns.values)
del head
newNames=[]
for name in colnames:
    print(name)
    column = read_csv(catTrainPath,usecols=[name])
    count=column.count()[0]
    print(count)
    if(count>thresh):
        newNames.append(name)
print("Nouveau nombre de colonnes:%d" % len(newNames))

DataFrame(newNames).to_csv("train_categorical_colnames")

colNames= read_csv("train_categorical_colnames")["0"].values
cat= read_csv(catTrainPath,dtype=np.float16,usecols=colNames)
        
max=cat.max()
newCols = cat.columns.values[np.where(max<200)]
