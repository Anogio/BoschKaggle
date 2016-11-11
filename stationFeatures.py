# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 10:51:26 2016

@author: ogier
"""
from pandas import *
import numpy as np
from JLMFun import nombre_station,hash_liste_station_unique
from globalVar import *

dateFile = dateTrainPath

dateCols= read_csv("train_date_colnames")["0"].values
X_date = read_csv(dateFile, dtype=np.float16,usecols=dateCols)
print("Import done")
ids= read_csv(dateFile,usecols=["Id"])
print("Calculating Features number")
feature_nombre_station = X_date.apply(nombre_station, axis = 1)
print("Computing piece path")
feature_station_unique= X_date.apply(hash_liste_station_unique, axis = 1)

stationStats = DataFrame(np.nan, index = range(len(ids)), columns = ["Id","Nb","Path"])

stationStats["Nb"]= feature_nombre_station
stationStats["Path"]=feature_station_unique
dateStats["Id"]=ids

dateStats.to_csv(feat2TrainPath)


dateFile = dateTestPath

dateCols= read_csv("train_date_colnames")["0"].values
X_date = read_csv(dateFile, dtype=np.float16,usecols=dateCols)
print("Import done")
ids= read_csv(dateFile,usecols=["Id"])
print("Calculating Features number")
feature_nombre_station = X_date.apply(nombre_station, axis = 1)
print("Computing piece path")
feature_station_unique= X_date.apply(hash_liste_station_unique, axis = 1)

stationStats = DataFrame(np.nan, index = range(len(ids)), columns = ["Id","Nb","Path"])

stationStats["Nb"]= feature_nombre_station
stationStats["Path"]=feature_station_unique
dateStats["Id"]=ids

dateStats.to_csv(feat2TestPath)