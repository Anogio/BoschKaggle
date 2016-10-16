'''
Created on Oct 9, 2016

@author: ogier
'''
from numpy import linspace
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from handling import multiGridSearch
from sklearn.model_selection import StratifiedKFold
from matplotlib import *
from sklearn.ensemble import RandomForestClassifier
Nlist=linspace(50000,50000,3                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    )
train_set_fraction=0.2    
    
skf= StratifiedKFold(n_splits=5)
    
classifiers=[SGDClassifier(class_weight="balanced"),RandomForestClassifier(class_weight="balanced")]
classNames=["SGD","Random Forest"]
parameters=[
            {"alpha":[0.01,1,10],"n_iter":[8,30]},
            {"n_estimators":[5,15]}
            ]
            
results=multiGridSearch(classifiers,classNames, parameters,skf,Nlist,train_set_fraction,plotResults=False, impute_scale=True, parallel=True)

for i in range(len(results)):
    plot(Nlist, results[i])

