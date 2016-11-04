'''
Created on Oct 9, 2016

@author: ogier
'''
from globalVar import trainPath, testPath, parall, disp

from numpy import linspace
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from handling import multiGridSearch, clfSearch
from sklearn.model_selection import StratifiedKFold
if disp:
    import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


Nfrac = 0.05
nTests=15
test_set_fraction=0.2 
    
skf= StratifiedKFold(n_splits=5)
skf3 = StratifiedKFold(n_splits=3)
#classifiers=[SGDClassifier(class_weight="balanced"),RandomForestClassifier(class_weight="balanced"),AdaBoostClassifier()]
#classNames=["SGD","Random Forest","Adaboost"]
#parameters=[
 #           {"alpha":[1],"n_iter":[30]},
 #          {"max_features": [None, 20,45, 65],"n_estimators":[20], "min_samples_leaf":[1,35,60], "min_samples_split" : [2,5,10], "max_leaf_nodes":[None, 3,20]},
   #         {"max_features": [None, 20,45, 65],"min_samples_leaf":[1,35,60],"min_samples_split" : [2,5,10],"max_leaf_nodes":[None, 3,20]}            
    #        ]

#results, bestEstim =multiGridSearch(trainPath,classifiers,classNames, parameters,skf3,N,1,test_set_fraction,plotResults=False, impute_scale=True, parallel= parall)
classifiers= [XGBClassifier()]
classNames=["XGB"]

#Version gourmande
prameters=[
           { "learning_rate" : [0.05] , "min_child_weight" : [1, 3,10], "max_depth" : [3,6,10],
             "gamma" : [0,1] , "subsample":[0.5,1] , "colsample_bytree" : [0.25,0.5], "reg_lambda" : [0,1],
             "scale_pos_weight" : [1,20,300]
            }
           ]
           
#Version diminuée
parameters=[
           { "learning_rate" : [0.01,0.1,0.2,0.5] , "min_child_weight" : [1, 3,10], "max_depth" : [3,6,10,15],
             "gamma" : [0,0.1,1,10] , "subsample":[0.5,1] , "colsample_bytree" : [0.1,0.25,0.5,1], "reg_lambda" : [0,1,10],
             "scale_pos_weight" : [1,10,100,1000]
            }
           ]
results, bestEstim =multiGridSearch(trainPath,classifiers,classNames, parameters,skf,Nfrac,nTests,test_set_fraction,plotResults=False, impute_scale=False, parallel= parall)
#results, bestClf = clfSearch(trainPath, classifiers, classNames, Nfrac, nTests, test_set_fraction,impute_scale = True)


