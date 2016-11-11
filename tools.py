# coding: utf-8
#Autorise les accents dans les commentaires
from pandas import *
from numpy import *
from scipy import *
from sklearn import *
from math import sqrt
import matplotlib.pyplot as plt

from sklearn.tests.test_cross_validation import train_test_split_pandas
from sklearn.preprocessing.data import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.model_selection import GridSearchCV , validation_curve
from sklearn.metrics.scorer import make_scorer
from sklearn.metrics.classification import matthews_corrcoef, confusion_matrix
from numpy import linspace
from matplotlib.pyplot import plot, show

def importData(filename,N):
    data= read_csv(filename,nrows=N)
    mid=int(N/2)
    train=data.iloc[0:mid]
    test=data.iloc[mid:N]   
    
    y_test=test["Response"].as_matrix()
    id_test=test["Id"].as_matrix()
    X_test=test.drop("Response",axis=1)
    X_test=X_test.drop("Id",axis=1)

    y_train=train["Response"].as_matrix()
    X_train=train.drop("Response",axis=1)
    X_train=X_train.drop("Id",axis=1)

    return X_train, y_train, X_test, y_test, id_test
    
def importDataDate(filename,N):
    data= read_csv(filename,nrows=N)
    mid=int(N/2)
    train=data.iloc[0:mid]
    test=data.iloc[mid:N]      
    
    id_train=train["Id"].as_matrix()
    id_test=test["Id"].as_matrix()
    
    X_test=test.drop("Id",axis=1)    
    X_train=train.drop("Id",axis=1)

    return X_train, X_test, id_train, id_test

#Compte des valeurs non nan    
def count(V):
    return sum(not np.isnan(x) for x in V)
    
#Calcule les features: moyenne/deviation    
def featureEngineering(X,nbr_feature,stats=[]):
    if(stats==[]):
        means=mean(X,axis=0)
        deviations=std(X,axis=0)
    else:
        means=stats[0]
        deviations=stats[1]
    X= X-means
    X=X/deviations   
    X=X.fillna(0)
    err=apply_along_axis(linalg.norm,axis=1,arr=X)
    err=err/nbr_feature
    return err, [means,deviations]

#Attention les tableaux d'entrée initiaux et new doivent avoir la même dimension...
def ajout_feature(X_train, X_test, X_train_new, X_test_new):
        return np.concatenate((X_train, X_train_new),axis=1), np.concatenate((X_test, X_test_new),axis=1)

