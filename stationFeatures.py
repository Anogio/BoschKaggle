# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 10:51:26 2016

@author: ogier
"""
from pandas import *
import numpy as np
from JLMFun import nombre_station,hash_liste_station_unique
from globalVar import *



dateCols= read_csv("train_date_colnames")["0"].values
X_date = read_csv(dateFile, dtype=np.float16,usecols=dateCols)
