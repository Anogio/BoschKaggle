# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 17:11:42 2016

@author: ogier


La fonction eval_date_feature prend en entrée:

- featFun : nrows*ncols ndArray -> nrows ndArray qui crée une feature à partir d'un jeu de données
- featName : une chaîne de caractères qui correspond au nom à utiliser pour la feature
- nRows : le nombre de lignes à importer pour faire l'évaluation. Actuellement, la sélection n'est PAS aléatoire
- chunksize: la taille d'échantillons à utiliser pour construire la feature. Plus grand = plus de mémoire mais plus rapide
- useallcols : par défaut élimine les colonnes de dates en double. Mettre à True pour importer toutes les colonnes à la place
- make_features : détermine s'il faut construire la feature avant de l'évaluer. Mettre False si le fichier existe déjà
"""

import pandas as pd
import globalVar as gv
import numpy as np

from xgboost import XGBClassifier
import gc

def make_date_feature_csv(featFun,featName,chunksize=10000,train=True, useallcols = False):
    if train:
        path = gv.dateTrainPath
    else:
        path= gv.dateTestPath
    ids= pd.read_csv(path,usecols=["Id"])["Id"]
    dateCols= pd.read_csv("train_date_colnames")["0"].values

    
    feats = []
    if useallcols :
        chunks = pd.read_csv(path,dtype=np.float,chunksize=chunksize)
    else:
        chunks = pd.read_csv(path,usecols=dateCols,dtype=np.float,chunksize=chunksize)
    
    for chunk in chunks :
        feat = featFun(chunk)
        feats.append(pd.DataFrame(feat))
    
    y = pd.concat(feats)
    
    out = pd.DataFrame(np.nan, index = range(len(ids)), columns = ["Id",featName])
    out["Id"]=ids
    out[featName] = y.values
    
    if train:
        outPath = gv.newFeaturesTrainPath + featName +".csv"
    else :
        outPath = gv.newFeaturesTestPath + featName + ".csv"
    
    out.to_csv(outPath, index=False)
    
def eval_date_feature(featFun,featName,nRows = 50000, chunksize = 10000, useallcols=False,make_features=True):
    if make_features:   
        print("Creating feature for train data")
        make_date_feature_csv(featFun,featName,chunksize,True,useallcols)
        print("Creating feature for test data")
        make_date_feature_csv(featFun,featName,chunksize,False,useallcols)
    
    #Import column filters
    dateCols= pd.read_csv("train_date_colnames")["0"].values
    catCols= pd.read_csv("train_categorical_colnames")["0"].values    
    
    print("Importing numeric data")
    dataTrain = pd.read_csv(gv.trainPath,dtype=np.float16,nrows= nRows)
    ids=pd.read_csv(gv.trainPath,usecols=["Id"],nrows=nRows)["Id"]
    dataTrain["Id"]=ids
    del ids
    print(dataTrain.shape)
    #Extract responses
    y=pd.read_csv(gv.trainPath,usecols=["Response"],nrows=nRows)["Response"]
    
    
    n_num_feat=dataTrain.shape[1]-2
    
    #Add date features
    print("Importing features")
    dataTrain= pd.concat([dataTrain,pd.read_csv(gv.featTrainPath,nrows=nRows,usecols=["Min","Max","Diff","Evol"])],axis=1)
    print(dataTrain.shape)
    n_new_feat=dataTrain.shape[1]-2-n_num_feat
    
    dataTrain= pd.concat([dataTrain,pd.read_csv(gv.featStationTrainPath,nrows=nRows,usecols=["NbFeats","NbStations","PathDupe","PathNoDupe"])],axis=1)
    print(dataTrain.shape)
    n_new_feat += 6
    
    gv.featNames.append(featName)
    for name in gv.featNames :
        dataTrain= pd.concat([dataTrain,pd.read_csv(gv.newFeaturesTrainPath+featName+".csv",nrows=nRows,usecols=[name])],axis=1)
        n_new_feat+=1
        
    print("Importing date data")
    dataTrain = pd.concat([dataTrain,pd.read_csv(gv.dateTrainPath,dtype=np.float16, usecols= dateCols,nrows=nRows)],axis=1,copy=False)
    gc.collect()
    print(dataTrain.shape)
    n_date_feat= dataTrain.shape[1]-2-n_num_feat - n_new_feat
    #Add categorical data
    print("Importing categorical data")
    dataTrain = pd.concat([dataTrain,pd.read_csv(gv.catTrainPath,dtype=np.float16,usecols=catCols,nrows=nRows).replace([np.inf, -np.inf], -1)],axis=1,copy=False)
    n_cat_feat = dataTrain.shape[1]-2-n_num_feat - n_new_feat - n_date_feat
    
    dataTrain=dataTrain.drop("Id",axis=1)
    dataTrain=dataTrain.drop("Response",axis=1)
    
    print("Number of columns for each category:")
    print([n_num_feat,n_new_feat,n_date_feat,n_cat_feat])
    
    print(dataTrain)
    print(dataTrain.shape)
    
    params = {'learning_rate': 0.05, 'max_depth': 6, 'colsample_bytree': 0.7, 'min_child_weight': 1, 'subsample': 0.5, 'gamma': 0.5, 'reg_lambda': 0, 'scale_pos_weight': 3}
    clfa=XGBClassifier(**params)
    
    print("Fitting")
    clfa.fit(dataTrain,y)
    
    cols = dataTrain.columns.values
    importance=clfa.feature_importances_
    
    ind = np.where(cols==featName)[0][0]
    score = importance[ind]
    
    tri=np.argsort(importance)
    importance= np.sort(importance)
    cols= cols[tri]
    rank= len(cols) - np.where(cols==featName)[0][0] 
    
    print("Score de la feature: %d" % score)
    print("Rang de la feature: %d sur %d" % (rank, len(cols)))
    
    print("Répartition des scores:")
    print(importance)
    