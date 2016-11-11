# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 18:30:14 2016

@author: ogier
"""
from pandas import *
import numpy as np
from globalVar import *

def removeDupes(dateFile, outfile):
    
    chunks = read_csv(dateFile,chunksize=30000, dtype=np.float32)
    dat =concat( [c.to_sparse() for c in chunks])
    dat=dat.T
    dat=dat.drop_duplicates()
    dat=dat.T
    dat=dat.dropna(axis=1, how="all")
    dat.to_csv(outfile)
    

def dateStat(dateFile, outfile, getLines=False):
    dateCols= read_csv("train_date_colnames")["0"].values
    dates = read_csv(dateFile, dtype=np.float16,usecols=dateCols)
    print("Import done")
    ids= read_csv(dateFile,usecols=["Id"])
    
    dateStats = DataFrame(np.nan, index = range(len(ids)), columns = ["Id","Min", "Max","Diff","Evol"])
    dateStats["Min"]= dates.min(axis=1)
    print("Min done")
    dateStats["Max"]= dates.max(axis=1)
    print("Max done")
    dateStats["Diff"] = dateStats["Max"].sub(dateStats["Min"])
    print("Diff done")
    dateStats["Id"]=ids
    if getLines:
        nonnull=np.where(np.isnan(dateStats["Max"])!=True)[0]
        DataFrame(nonnull).to_csv("nonnullines.csv")

    dateStats= dateStats.sort_values(by=['Min','Id'],ascending=True)
    dateStats["Evol"]=dateStats["Id"].diff()
    print("Evol done")
    dateStats= dateStats.sort_values(by=["Id"])
    print("Sort done")
    dateStats.to_csv(outfile)
    print("Done writing")
