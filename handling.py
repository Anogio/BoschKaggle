# coding: utf-8
#Autorise les accents dans les commentaires

from pandas import *
from numpy import *
from scipy import *
if disp:
    from matplotlib import *
    from matplotlib.pyplot import plot, show
from sklearn.preprocessing.data import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.metrics.scorer import make_scorer
from sklearn.model_selection import GridSearchCV ,train_test_split
from sklearn.metrics.classification import matthews_corrcoef, confusion_matrix
from numpy import *

def importData(filename,N,testSize):
    data= read_csv(filename,nrows=N)
    y=data["Response"]
    X=data.drop("Response",axis=1)
    X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=testSize)
    
    test_id= X_test["Id"]
    X_test= X_test.drop("Id", axis=1)
    X_train= X_train.drop("Id",axis=1)

    return X_train, y_train, X_test, y_test, test_id

def imputeAndScale(X_train,X_test):
    imp= Imputer()
    X_train=imp.fit_transform(X_train)
    X_test=imp.transform(X_test)
    
    scaler= StandardScaler().fit(X_train)
    X_test=scaler.transform(X_test)
    X_train= scaler.transform(X_train)
    
    return X_train, X_test

#Compte des valeurs non nan    
def count(V):
    return sum(not np.isnan(x) for x in V)
    

#Attention les tableaux d'entrée initiaux et new doivent avoir la même dimension...
def ajout_feature(X_train, X_test, X_train_new, X_test_new):
        return np.concatenate((X_train, X_train_new),axis=1), np.concatenate((X_test, X_test_new),axis=1)


#Fonctions d'évaluation des faux positifs, positifs, faux négatifs
def evaluate(y_pred, y_test):
    perf= matthews_corrcoef(y_test, y_pred)

    print("Prediction score:%s" % perf)
    tn,fp,fn,tp=confusion_matrix(y_test, y_pred).ravel()
    print("True negatives:%s" % tn)
    print("True positives: %s" % tp)
    print("False negatives: %s" % fn)
    print("False positives: %s" % fp)
    
    nn= tn + fp
    np= tp + fn
    
    ratio_tp = float(tp)/float(np) #Proche de 1 si on a bien prédit que ça échouait au test
    ratio_tn = float(tn)/float(nn) #Proche de 1 si on a bien prédit que ça passait le test
    ratio_fp = float(fp)/float(np+nn) #Proche de 0 si on se loupe pas
    ratio_fn = float(fn)/float(nn+nn) #Proche de 0 si on se loupe pas
    print("Ratio TP sur Nbre Pos:%s, Ratio TN sur Nbre N:%s, Ratio FP sur NTotal:%s, Ratio FN sur NTotal:%s" % (ratio_tp, ratio_tn, ratio_fp,ratio_fn))
    
    return perf


def GridSearch(clf, X_train, y_train, X_test):
    clf.fit(X_train, y_train)
    print("Best parameters:%s" % clf.best_params_)
    print("Cross Validation score: %s" % clf.best_score_)
            
    y_pred=clf.predict(X_test)
    
    return y_pred


def multiGridSearch(filename,classifiers,classNames, parameters, crossVal, Nlist,test_set_fraction,plotResults=False, impute_scale= True, parallel=False):
    allResults=[]
    scorer=make_scorer(matthews_corrcoef)
    for i in range(len(classifiers)):
        classif= classifiers[i]
        className=classNames[i]
        param=parameters[i]

        results=[]

        print("Evaluating performance for classifier: %s" % className)
        
        for N in Nlist:

            print("N=%s" % N)
            X_train, y_train, X_test, y_test, test_id = importData(filename,N,test_set_fraction)
            if impute_scale:
                X_train, X_test= imputeAndScale(X_train,X_test)
            if parallel:
                if __name__ == "__main__" :
                    clf=GridSearchCV(classif,param,scoring=scorer,cv=crossVal, verbose=1, n_jobs=-1)
                    y_pred= GridSearch(clf, X_train, y_train, X_test)
                    perf = evaluate(y_pred,y_test)
                    results.append(perf)
            else:
                clf=GridSearchCV(classif,param,scoring=scorer,cv=crossVal, verbose=1)
                y_pred= GridSearch(clf, X_train, y_train, X_test)
                perf = evaluate(y_pred,y_test)
                results.append(perf)

            del X_test
            del X_train
            del y_train
            del y_test
       
        allResults.append(results)
        if plotResults:
            plot(Nlist,results)
            show()
        print("_" * 10)
        print("\n"*5)
    return allResults
