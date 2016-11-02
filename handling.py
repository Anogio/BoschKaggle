# coding: utf-8
#Autorise les accents dans les commentaires

from pandas import *
from numpy import *
from scipy import *
try:
    from matplotlib import *
    from matplotlib.pyplot import plot, show
except ImportError :
    print("Not importing matplotlib")
from sklearn.preprocessing.data import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.metrics.scorer import make_scorer
from sklearn.model_selection import GridSearchCV ,train_test_split
from sklearn.metrics.classification import matthews_corrcoef, confusion_matrix

def importData(filename,N,testSize, split=True):
    data= read_csv(filename,nrows=N)
    y=data["Response"]
    X=data.drop("Response",axis=1)
    if split:
        X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=testSize)
        
        test_id= X_test["Id"]
        X_test= X_test.drop("Id", axis=1)
        X_train= X_train.drop("Id",axis=1)
    
        return X_train, y_train, X_test, y_test, test_id
    else:
        return X,y
        
def importData_chunks(filename, N_fraction=0.05, test_fraction=0.2, split= True):
    chunks = read_csv(filename,chunksize=50000, dtype=np.float32)
    X= concat([c.sample(frac=N_fraction) for c in chunks])
    y = X["Response"]
    X = X.drop("Response",axis=1)
    if split:
        X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=test_fraction)
        
        test_id= X_test["Id"]
        X_test= X_test.drop("Id", axis=1)
        X_train= X_train.drop("Id",axis=1)
        
        return X_train, y_train, X_test, y_test, test_id
    else:
        return X, y 

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


def multiGridSearch(filename,classifiers,classNames, parameters, crossVal, Nrows,nTests,test_set_fraction,plotResults=False, impute_scale= True, parallel=False):
    allResults=[]
    best=0
    bestEstim= None
    bestEstimName=None
    scorer=make_scorer(matthews_corrcoef)
    print("Grid search on N=%s" % Nrows)
    print("_" * 10)
    print("\n"*5)
    for i in range(len(classifiers)):
        classif= classifiers[i]
        className=classNames[i]
        param=parameters[i]

        results=[]

        print("Evaluating performance for classifier: %s" % className)
        for j in range(nTests) :
            print("Test number %d for %s:" % (j+1, className))
            X_train, y_train, X_test, y_test, test_id = importData_chunks(filename,Nrows/1000000,test_set_fraction)
            if impute_scale:
                X_train, X_test= imputeAndScale(X_train,X_test)
            if parallel:
                clf=GridSearchCV(classif,param,scoring=scorer,cv=crossVal, verbose=1, n_jobs=-1)
                y_pred= GridSearch(clf, X_train, y_train, X_test)
                perf = evaluate(y_pred,y_test)
                results.append(perf)
            else:
                clf=GridSearchCV(classif,param,scoring=scorer,cv=crossVal, verbose=1)
                print(X_train.shape)
                print(y_train.shape)
                print(X_test.shape)
                y_pred= GridSearch(clf, X_train, y_train, X_test)
                print(y_pred.shape)
                perf = evaluate(y_pred,y_test)
                results.append(perf)
            if perf > best:
                bestEstim = clf
                bestEstimName=className
            del X_test
            del X_train
            del y_train
            del y_test
       
        allResults.append(results)
        if plotResults:
            plot(linspace(1,nTests,nTests),results)
            show()
        print("_" * 10)
        print("\n"*5)
    print("Best results overall:")
    print("Best classifier:%s" % bestEstimName)
    print("Best parameters:")
    print(bestEstim.best_params_)
    return allResults, bestEstim

def clfSearch(filename, classifiers, classNames, Nrows, nTests, test_set_fraction,impute_scale = True):
    allResults=[]
    best=0
    bestEstim= None
    bestEstimName=None
    print("Classifier evaluation on N=%s" % Nrows)
    print("_" * 10)
    print("\n"*5)
    
    for i in range(len(classifiers)):
        classif= classifiers[i]
        className=classNames[i]

        results=[]
        print("Evaluating performance for classifier: %s" % className)
        for j in range(nTests) :
            print("Test number %d for %s:" % (j+1, className))
            X_train, y_train, X_test, y_test, test_id = importData_chunks(filename,0.1,test_set_fraction)
            print(X_train.shape)
            print(X_test.shape)
            print(y_train.shape)
            if impute_scale:
                X_train, X_test= imputeAndScale(X_train,X_test)
            classif.fit(X_train,y_train)
            y_pred = classif.predict(X_test)
            perf= evaluate(y_pred, y_test)
            results.append(perf)
            if perf > best:
                    bestEstim = classif
                    bestEstimName=className
            del X_test
            del X_train
            del y_train
            del y_test
            allResults.append(results)
        print("_" * 10)
        print("\n"*5)
    return allResults, bestEstim
    
    
def test_feature(X_array, y_array, feature_array, classifier,nRows, n_tests, testSize):
    concat_array = concatenate((data_array,feature_array),axis=1)
    perfWithout=[]
    perfWith=[]
    diff=[]
    for i in range(n_tests):
        print("Performing test number %d" % i+1)
        msk = random.rand(len(y_array)) < testSize 
        y_test= y_array[msk]
        y_train = y_array[-msk]
        
        X_testWithout = X_array[msk]
        X_trainWithout = X_array[-msk]
        
        X_testWith = concat_array[msk]
        X_trainWith = concat_array[-msk]
        
        classifier.fit(X_trainWithout,y_train)
        y_predWithout = classifier.transform(X_testWithout)
        
        classifier.fit(X_trainWith, y_train)
        y_predWith = classifier.transform(X_testWith)
        print("Evaluation without added features:")
        perfWithout.append(evaluate(y_predWithout,y_test))
        print("Evaluation with added features:")        
        perfWith.append(evaluate(y_predWith,y_test))
        diff.append(perfWith[i]- perfWithout[i])
    
    meanWith= sum(perfWith)/len(perfWith)
    meanWithout = sum(perfWithout)/len(perfWithout)    
    meanDiff = sum(diff)/len(diff)
    
    print("Performance without added features:")
    print(perfWithout)
    print("With mean %d" % meanWith)
    print("Performance with added features:")
    print(perfWith)
    print("With mean %d" % meanWithout)
    
    print("Difference:")
    print(diff)
    print("With mean %d" % meanDiff)
        
def xgb_gridsearch():
    clf= 1
        
        
        
        
        
        
        
        
        
        
        
        
        