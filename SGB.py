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


N = 50000
nTests=15
test_set_fraction=0.4 
    
skf= StratifiedKFold(n_splits=5)
skf3 = StratifiedKFold(n_splits=3)
#classifiers=[SGDClassifier(class_weight="balanced"),RandomForestClassifier(class_weight="balanced"),AdaBoostClassifier()]
#classNames=["SGD","Random Forest","Adaboost"]
#parameters=[
 #           {"alpha":[1],"n_iter":[30]},
 #          {"max_features": [None, 20,45, 65],"n_estimators":[20], "min_samples_leaf":[1,35,60], "min_samples_split" : [2,5,10], "max_leaf_nodes":[None, 3,20]},
   #         {"max_features": [None, 20,45, 65],"min_samples_leaf":[1,35,60],"min_samples_split" : [2,5,10],"max_leaf_nodes":[None, 3,20]}            
    #        ]
classifiers= [DecisionTreeClassifier(max_depth=1, class_weight="balanced")]
classNames=("Decision Stump")
parameters=[
            {"min_samples_split" : [5,7,10,15,20], "min_samples_leaf" : [1,3,20,35,60,80], "criterion" :["gini","entropy"]}
        ]
#results, bestEstim =multiGridSearch(trainPath,classifiers,classNames, parameters,skf3,N,1,test_set_fraction,plotResults=False, impute_scale=True, parallel= parall)


#classifiers=[AdaBoostClassifier(base_estimator=bestEstim, n_estimators=50)]
classNames=["RandomForest"]
#results, Ada = clfSearch(trainPath, classifiers, classNames, N, nTests, test_set_fraction,impute_scale = True)

#classifiers=[AdaBoostClassifier(base_estimator=bestEstim, n_estimators=100)]
#results, Ada = clfSearch(trainPath, classifiers, classNames, N, nTests, test_set_fraction,impute_scale = True)

classifiers=[RandomForestClassifier(class_weight="balanced",n_estimators=20, max_features=None, min_samples_leaf=45, min_samples_split=7)]
results, Ada = clfSearch(trainPath, classifiers, classNames, N, nTests, test_set_fraction,impute_scale = True)


