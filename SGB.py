'''
Created on Oct 9, 2016

@author: ogier
'''

from sklearn.metrics.scorer import make_scorer
from handling import multiGridSearch
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot

Nlist=linspace(5000,50000,51)
train_set_fraction=0.2    
    
skf= StratifiedKFold(n_splits=5)
    
classifiers=[SGDClassifier(class_weight="balanced"),RandomForestClassifier(class_weight="balanced")]
classNames=["SGD","Random Forest"]
parameters=[
            {"alpha":[0.0001, 0.01,1,10,100],"n_iter":[5,20,50]},
            {"n_estimators":[5,15,50]}
            ]
            
results=multiGridSearch(classifiers,classNames, parameters, Nlist,plotResults=False)

for i in range(len(results)):
    plot(Nlist, results[i])

