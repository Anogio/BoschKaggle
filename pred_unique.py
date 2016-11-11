#coding: utf-8
'''
Created on Oct 9, 2016

@author: ogier
'''

from tools import *
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

filepath_date ="C:/Users/JLM/Desktop/Kaggle/train_date.csv"
filepath_numeric = "C:/Users/JLM/Desktop/Kaggle/train_numeric.csv"


#%%
N = 20000
print("N=%s" % N)
data= read_csv(filepath_numeric,nrows=N)
y_numeric=data["Response"]
X_numeric=data.drop("Response",axis=1)
X_numeric_id=data["Id"]
X_numeric=data.drop("Id", axis=1)
#X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2)

data= read_csv(filepath_date,nrows=N)
X_date_id=data["Id"]
X_date=data.drop("Id",axis=1)

#%% Vérifie que les id correspondent bien 2 à 2
#print sum(X_date_id == X_numeric_id)

#%% Créé les features de dates
#Traitement des données Date
#Nombre de feature par objet
date_max_piece= np.nanmax(X_date,axis=1)
date_min_piece=np.nanmin(X_date,axis=1)

#J'ai du utiliser nanmin/nanmax car apparemment certaines valeurs du tableaux sont à Nan.
#Mais je comprends pas pourquoi... Ca veut dire que certaines pièces ne sont pas passées du tout dans la chaine?
#Ou qu'on a pas mesuré de date? Est-ce que ces objects sont plus souvent défectueux dans ce cas ?
#Par ailleurs, je vais remplacer par -1 valeurs min/max qui valent nan. L'écart sera alors 0 par ailleurs.

date_max_piece = date_max_piece[:,np.newaxis]
date_min_piece = date_min_piece[:,np.newaxis]

#Remplace les nan par -1 dans les deux tableaux
for i in linspace(0, N-1):
    if isnan(date_max_piece[i,0]):
        date_max_piece[i,0] = -1
        print i

#Ecart des dates entrées et sorties
ecart_date = date_max_piece - date_min_piece

#%% Realise la concathénation des tableaux numeric et feature avec les id qui correspondent
X = np.concatenate((X_numeric, ecart_date),axis=1)
X = np.concatenate((X, date_min_piece),axis=1)

#Split en test et train après avoir ajouté les features
X_train, X_test, y_train, y_test= train_test_split(X,y_numeric,test_size=0.2)

#%% Traite le tableau numeric sans les dates d'entrées
#In [20]: selector = [x for x in range(a.shape[1]) if x != 2]
#In [21]: a[:, selector]
#Out[21]: 
#array([[ 1,  2,  4],
#       [ 2,  4,  8],
#       [ 3,  6, 12]])

#Normalise tout sauf la dernière colonne dans train et test
imp= Imputer()
X_train[:,:-1]=imp.fit_transform(X_train[:,:-1])
X_test[:,:-1]=imp.transform(X_test[:,:-1])

scaler= StandardScaler().fit(X_train[:,:-1])
X_test[:,:-1]=scaler.transform(X_test[:,:-1])
X_train[:,:-1]= scaler.transform(X_train[:,:-1])


#%%
skf= StratifiedKFold(n_splits=5)
sgd=SGDClassifier(class_weight="balanced")
scorer=make_scorer(matthews_corrcoef)
parameters= {"alpha":[10],"n_iter":[10]}

#%%
if __name__ == "__main__" :
    clf=GridSearchCV(sgd,parameters,scoring=scorer,cv=skf, verbose=5, n_jobs=-1)
    clf.fit(X_train, y_train)   
    print("Best parameters:%s" % clf.best_params_)
    print("Cross Validation score: %s" % clf.best_score_)
    #validation_curve(X_test,
    y_pred=clf.predict(X_test)
    perf= matthews_corrcoef(y_test, y_pred)
    results.append(perf)
    print("Prediction score:%s" % perf)
    tn,fp,fn,tp=confusion_matrix(y_test, y_pred).ravel()
    print("True negatives:%s" % tn)
    print("True positives: %s" % tp)
    print("False negatives: %s" % fn)
    print("False positives: %s" % fp)
    del data
    del X_test
    del X_train
    del y_train
    del y_test
    del X_train_date
    del X_test_date
    del id_train
    del id_test
