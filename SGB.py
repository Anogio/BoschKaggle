'''
Created on Oct 9, 2016

@author: ogier
'''
from globalVar import trainPath, testPath, parall, disp

from numpy import linspace
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from handling import multiGridSearch
from sklearn.model_selection import StratifiedKFold
if disp:
    import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


Nlist= linspace(50000, 50000, 15 )
test_set_fraction=0.4  
    
skf= StratifiedKFold(n_splits=5)
    
classifiers=[SGDClassifier(class_weight="balanced"),RandomForestClassifier(class_weight="balanced")]
classNames=["SGD","Random Forest"]
parameters=[
            {"alpha":[1],"n_iter":[30]},
            {"max_features": [None, 20,45, 65],"n_estimators":[20], "min_samples_leaf":[1,35,60], "min_samples_split" : [2,5,10], "max_leaf_nodes":[None, 3,20], }
            ]
            
results=multiGridSearch(trainPath,classifiers,classNames, parameters,skf,Nlist,test_set_fraction,plotResults=False, impute_scale=True, parallel= parall)

for i in range(len(results)):
    print(results)
    plt.plot(Nlist, results[i])

