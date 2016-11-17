# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 23:18:18 2016

@author: JLM
"""

import numpy as np
from pandas import read_csv, DataFrame
from numpy import isnan , asarray

#%% FEATURE date_sortie, date_entree, temps de passage
#Date de sortie de la chaine

def date_entree(X_date):
    #Date d'entrée dans la chaine
    date_min_piece=np.nanmin(X_date,axis=1)
    max_date_min_piece=np.max(date_min_piece)
    date_min_piece = date_min_piece/max_date_min_piece
    date_min_piece = date_min_piece[:,np.newaxis]
    return date_min_piece
    
def date_sortie(X_date):
    date_max_piece= np.nanmax(X_date,axis=1)
    max_date_max_piece=np.max(date_max_piece)
    date_max_piece = date_max_piece/max_date_max_piece
    date_max_piece = date_max_piece[:,np.newaxis]
    return date_max_piece

def temps_passage_chaine(date_max, date_min): #Prend en entrée les deux features dejà calcule
    #Ecart des dates entrées et sorties
    ecart_date = date_max - date_min
    return ecart_date

#%% FEATURE Nombre de feature par objet
#Compte des valeurs non nan    


def fun_nbr_feature_piece(X_date):
    #Compte les non nan dans un tableau
    return X_date.count(axis=1)


#%% ANALYSE DES STATIONS
#Recherche le numéro d'une station dans un nom provenant du fichier data uniquement
def recherche_numero_station(name):
        if type(name)==type("a"):
            ini = name.find('S') +1
            fin = name.find('D') -2
            return int(name[ini:fin+1])
        else:
            return np.nan

vect_recherche_numero_station = np.vectorize(recherche_numero_station)

def numero_station(vector):
    #Renvoit un vecteur de numéro de stations à partir d'un vector de date
    tri = (isnan(vector) == False)
    vector_tri = vector[tri]
    return vect_recherche_numero_station(np.asarray(vector_tri.axes[0][:]))
        
def compte_nombre(vector):
    #Compte le nombre d'élements d'un tableau sans nan et doublons
    return np.unique(vector).shape[0]
    
def nombre_station(vector):
    if vector.isnull().all():
        return 0
    return compte_nombre(numero_station(vector))

#Recupére les valeurs des dates où l'objet est passé
def date_station(vector):
    tri = (isnan(vector)==False)
    return vector[tri == True]
        
    
#Liste des stations de la chaine
#station = []
#for name in date_feature_column:
#    station_numero = recherche_numero_station(name)
#    if station == []:
#        station.append(station_numero)
#    elif station[-1] != station_numero:
#        station.append(station_numero)


#%%FEATURE Détermine la station d'entrée de l'objet dans la chaine et sa station de sortie

def fun_station_entree(X_date):
  return asarray([recherche_numero_station(x) for x in X_date.idxmin(axis=1)])
    
def fun_station_sortie(X_date):
    return asarray([recherche_numero_station(x) for x in X_date.idxmax(axis=1)])
    

#%%FEATURE nombre de stations parcourues par l'objet

def fun_feature_nombre_station(X_date):
    feature_nombre_station = X_date.apply(nombre_station, axis = 1)
    return feature_nombre_station


#%%FEATURE Analyse plus fine des stations parcourues
#Idee: chaque liste de station parcourrue est un chemin 
#Attribuer un nombre à chaque chemin => Regrouper les objets par chemin empruntés
#Normer ce nombre

def fun_hash_parcours_unique(X_date):

    #Renvoit une valeur unique pour chaque parcours de la chaine en ayant éliminé les stations parcourues 2 fois
    def hash_liste_station_unique(vector):
        if vector.isnull().all():
            return 0
        station_vector_unique = np.unique(numero_station(date_station(vector))).astype(np.float)
        string = np.str(station_vector_unique)
        return float(hash(string))
        
    feature_station_unique= X_date.apply(hash_liste_station_unique, axis = 1)
    
    return feature_station_unique
    
def fun_hash_parcours_sort(X_date):

    #Renvoit une valeur unique pour chaque parcours de la chaine sans élimination des doublons
    def hash_liste_station_sort(vector):
        if vector.isnull().all():
            return 0
        station_vector_sort = np.sort(numero_station(date_station(vector))).astype(np.float)
        string = np.str(station_vector_sort)
        return float(hash(string))

    feature_station_sort= X_date.apply(hash_liste_station_sort, axis = 1)

    return feature_station_sort

#ATTENTION: Le problème dans mon hash c'est que deux parcours très similaires 
#peuvent avoir des hash très différents...

def allStationFeatures(dateFile,outFile):
    print("Exporting station features to %s" % outFile)
    print("Importing from %s" % dateFile)
    dateCols= read_csv("train_date_colnames")["0"].values
    X_date = read_csv(dateFile, dtype=np.float16,usecols=dateCols)
    print("Import done")
    ids= read_csv(dateFile,usecols=["Id"])
    stationFeats = DataFrame(np.nan, index = range(len(ids)), columns = ["Id","NbFeats", "Station1","StationLast","NbStations","PathDupe","PathNoDupe"])
    stationFeats["Id"]=ids
    #%% Calcule toutes les features
    
    
    stationFeats["NbFeats"] = fun_nbr_feature_piece(X_date)
    print("NbFeats done")
    stationFeats["Station1"] = fun_station_entree(X_date)
    print("Station1 done")
    stationFeats["StationLast"]= fun_station_sortie(X_date)
    print("StationLast done")
    stationFeats["NbStations"]= fun_feature_nombre_station(X_date)
    print("NbStations done")
    stationFeats["PathDupe"] = fun_hash_parcours_sort(X_date)
    print("PathDupe done")
    stationFeats["PathNoDupe"]= fun_hash_parcours_unique(X_date)
    print("PathNoDupe done")
    
    stationFeats.to_csv(outFile)
    print("Done writing")
