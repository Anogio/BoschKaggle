import numpy as np
from numpy.lib.format import ompen_memmap)
from pandas import *

def to_memmap(fromFile, toFile,nRows,nCols):
    memmap= open_memmap(toFile,mode='w+', dtype=np.float32, shape=(nRows,nCols))
    
    n=0
    
    for chunk in read_csv(fromFile, chunksize=10000):
        memmap[n:n+chunk.shape[0]]=chunk.values
        n+=chunk.shape[0]
        
    succ=np.allclose(memmap)
    print(succ)
    return succ