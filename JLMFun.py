# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 11:12:42 2016

@author: ogier
"""
import numpy as np

def count(V):
    return sum(not np.isnan(x) for x in V)
    
def recherche_numero_station(name):
    ini = name.find('S') +1
    fin = name.find('D') -2
    return name[ini:fin+1]
    
def fun_station_entree(vector):
    return recherche_numero_station(np.argmin(vector)) 

def fun_station_sortie(vector): 
    return recherche_numero_station(np.argmax(vector)) 
    
vect_recherche_numero_station = np.vectorize(recherche_numero_station)

def numero_station(vector):
    print(vector)
    #Renvoit un vecteur de numéro de stations à partir d'un vector de date
    tri = (np.isnan(vector) == False)
    vector_tri = vector[tri == True]
    return vect_recherche_numero_station(np.asarray(vector_tri.axes[0][:]))

def compte_nombre(vector):
    #Compte le nombre d'élements d'un tableau sans nan et doublons
    return np.unique(vector).shape[0]
    
def nombre_station(vector):
    return compte_nombre(numero_station(vector))
    
def date_station(vector):
    tri = (isnan(vector)==False)
    return vector[tri == True]

#Renvoit une valeur unique pour chaque parcours de la chaine en ayant éliminé les stations parcourues 2 fois
def hash_liste_station_unique(vector):
    station_vector_unique = np.unique(numero_station(date_station(vector))).astype(np.float)
    string = np.str(station_vector_unique)
    return hash(string)
 
#Renvoit une valeur unique pour chaque parcours de la chaine sans élimination des doublons
def hash_liste_station_sort(vector):
    print(vector)
    station_vector_sort = np.sort(numero_station(date_station(vector))).astype(np.float)
    string = np.str(station_vector_sort)
    return hash(string)
    
def erreur_station(vector):
    station_vector_unique = np.unique(numero_station(date_station(vector))).astype(np.float)
    vector_response = y_numeric[vector.name]
    for x in station_vector_unique:
        erreur[int(x)]=erreur[int(x)]+vector_response
        nbr[int(x)]=nbr[int(x)]+1