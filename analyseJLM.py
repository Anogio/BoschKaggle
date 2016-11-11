# -*- coding: utf-8 -*-
from tools import *

#Classer les features par proximité en nombre/type
#Selectionner les VP les plus grandes => Importance des features
#PCA: Cercle de correlation, toutes les features correlées 
#Select KBest
#PCA Incremental "Partial fit"
#=> Importance de la valeur numérique en station i sur le résultat final
#LDA (Linear Discriminant Analysis) => Separer les nuages de points
#Clustering Key Nearest Neighbourgs => Vote de la majorité

#%% Importe les fichiers de base

filepath_date ="C:/Users/JLM/Desktop/Kaggle/train_date.csv"
filepath_numeric = "C:/Users/JLM/Desktop/Kaggle/train_numeric.csv"

N = 10000

data= read_csv(filepath_numeric,nrows=N)
y_numeric=data["Response"]
X_numeric=data.drop("Response",axis=1)
X_numeric_id=data["Id"]
X_numeric=data.drop("Id", axis=1)

data= read_csv(filepath_date,nrows=N)
X_date_id=data["Id"]
X_date=data.drop("Id",axis=1)

#%% Vérifie que les id des 2 fichiers date et numérique correspondent bien en ligne 2 à 2
print (sum(X_date_id == X_numeric_id))

#%%Nettoie le tableau: enlève les lignes toutes nulles
tri = X_date.count(axis = 1)
X_numeric = X_numeric[tri != 0]
X_date = X_date[tri != 0]
y_numeric = y_numeric[tri != 0]
X_date_id = X_date_id[tri != 0]

#%% Nettoie le tableau des dates en supprimant les dates doublons

#Vérifie que les dates incohérentes sont bien des doublons
#print np.testing.assert_array_equal(X_date['L0_S6_D127'],X_date['L0_S6_D130'])
#print np.testing.assert_array_equal(X_date['L0_S6_D124'],X_date['L0_S6_D130'])

date_feature_column = list(X_date.columns)
numeric_feature_column = list(X_numeric.columns)
numeric_feature_column.remove('Response')

i = 0
for name in date_feature_column:
    position = name.find('D') +1
    numero_date = int(name[position:])
    
    position = numeric_feature_column[i].find('F')+1
    numero_feature = int(numeric_feature_column[i][position:])
    
    if numero_date < numero_feature:
        del X_date[name]
    else:
        i = i +1
    
date_feature_column = list(X_date.columns)
numeric_feature_column = list(X_numeric.columns)
numeric_feature_column.remove('Response')  
  
date_feature_number = []
for name in date_feature_column:
    position = name.find('D') +1
    date_feature_number.append(int(name[position:])-1)
    
numeric_feature_number = []
for name in numeric_feature_column:
    position = name.find('F') +1
    numeric_feature_number.append(int(name[position:]))

 
numeric_feature_number = np.asarray(numeric_feature_number)
date_feature_number = np.asarray(date_feature_number) 

#Soit la date c'est +1 soit la date c'est +2
#On dirait qu'il y a deux catégories de mesure de dates...
print (numeric_feature_number-date_feature_number)
print( np.min(numeric_feature_number-date_feature_number))


#%% FEATURE date_sortie, date_entree, temps de passage
#Date de sortie de la chaine
date_max_piece= np.nanmax(X_date,axis=1)
max_date_max_piece=np.max(date_max_piece)

#Date d'entrée dans la chaine
date_min_piece=np.nanmin(X_date,axis=1)
max_date_min_piece=np.max(date_min_piece)

#Ecart des dates entrées et sorties
ecart_date = date_max_piece - date_min_piece

#Normalisation
date_max_piece = date_max_piece/max_date_max_piece
date_min_piece = date_min_piece/max_date_min_piece

#Transforme en matrice
date_max_piece = date_max_piece[:,np.newaxis]
date_min_piece = date_min_piece[:,np.newaxis]

scaler= StandardScaler().fit(ecart_date)
ecart_date=scaler.transform(ecart_date)


#%% FEATURE Nombre de feature par objet
#Compte des valeurs non nan    
def count(V):
    return sum(not np.isnan(x) for x in V)
    
nbr_feature_piece= apply_along_axis(count,axis=1,arr=X_numeric)
scaler= StandardScaler().fit(nbr_feature_piece)
feature_station_sort=scaler.transform(nbr_feature_piece)


#%% ANALYSE DES STATIONS
#Recherche le numéro d'une station dans un nom provenant du fichier date uniquement
def recherche_numero_station(name):
    ini = name.find('S') +1
    fin = name.find('D') -2
    return name[ini:fin+1]

#Liste des stations de la chaine
station = []
for name in date_feature_column:
    station_numero = recherche_numero_station(name)
    if station == []:
        station.append(station_numero)
    elif station[-1] != station_numero:
        station.append(station_numero)


#%%FEATURE Détermine la station d'entrée de l'objet dans la chaine et sa station de sortie
def fun_station_entree(vector):
    return recherche_numero_station(np.argmin(vector)) 

def fun_station_sortie(vector): 
    return recherche_numero_station(np.argmax(vector)) 
   
station_entree = X_date.apply(fun_station_entree, axis = 1)
station_sortie = X_date.apply(fun_station_sortie, axis = 1)

#%%FEATURE nombre de stations parcourues par l'objet

vect_recherche_numero_station = np.vectorize(recherche_numero_station)

def numero_station(vector):
    #Renvoit un vecteur de numéro de stations à partir d'un vector de date
    tri = (isnan(vector) == False)
    vector_tri = vector[tri == True]
    return vect_recherche_numero_station(np.asarray(vector_tri.axes[0][:]))

def compte_nombre(vector):
    #Compte le nombre d'élements d'un tableau sans nan et doublons
    return np.unique(vector).shape[0]
    
def nombre_station(vector):
    return compte_nombre(numero_station(vector))

feature_nombre_station = X_date.apply(nombre_station, axis = 1)

#%%FEATURE Analyse plus fine des stations parcourues
#Idee: chaque liste de station parcourrue est un chemin 
#Attribuer un nombre à chaque chemin => Regrouper les objets par chemin empruntés
#Normer ce nombre

#Recupére les valeurs des dates où l'objet est passé
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
    station_vector_sort = np.sort(numero_station(date_station(vector))).astype(np.float)
    string = np.str(station_vector_sort)
    return hash(string)

feature_station_unique= X_date.apply(hash_liste_station_unique, axis = 1)
#feature_station_sort= X_date.apply(hash_liste_station_sort, axis = 1)

#Normalise les deux features précédentes
scaler= StandardScaler().fit(feature_station_unique)
feature_station_unique=scaler.transform(feature_station_unique)

scaler= StandardScaler().fit(feature_station_sort)
feature_station_sort=scaler.transform(feature_station_sort)

#ATTENTION: Le problème dans mon hash c'est que deux parcours très similaires 
#peuvent avoir des hash très différents...

#Il n'y a que 951 listes de stations differentes (en excluant les doublons) sur 9997 objets
nbr_chemins_unique = np.unique(feature_station_unique).shape
print (nbr_chemins_unique)

#Il n'y a que 1382 listes de stations differentes sur 9997 objets
nbr_chemins_sort = np.unique(feature_station_sort).shape
print( nbr_chemins_sort)

# => Ca m'étonne qu'il y en aie aussi peu lorsqu'on élimine pas les doublons...
#Ca veut dire que la chaine traite globalement 1400 objets différents... ?

#%% FEATURE Taux d'erreur par station
# Dédoubler les colonnes des stations avec le plus d'erreur pour leur donner plus de poids 
# Aider l'algorithme

#Nombre d'erreurs passées par la station i
erreur = np.zeros(52)
#Nombre d'objet passés par la station i
nbr= np.zeros(52)

def erreur_station(vector):
    station_vector_unique = np.unique(numero_station(date_station(vector))).astype(np.float)
    vector_response = y_numeric[vector.name]
    for x in station_vector_unique:
        erreur[int(x)]=erreur[int(x)]+vector_response
        nbr[int(x)]=nbr[int(x)]+1

#Cree le tableau des erreurs
X_date.apply(erreur_station, axis=1)

#Calcule le taux d'erreur par station et l'affiche
taux_erreur = erreur/nbr
print (taux_erreur[np.isnan(taux_erreur)==False]) #Les stations nan sont les stations qui n'existent pas

plt.figure()
plt.scatter(station,taux_erreur[np.isnan(taux_erreur)==False])


#%% RIEN DE TRES INTERESSANT EN DESSOUS

#%%

#%% Cellule d'essai pour le developpement de nouvelles features
# NE PAS SUPPRIMER
vector = X_date.iloc[9]
vector_response= y_numeric[9]
print (vector.name)
tri = (isnan(X_date.iloc[9]) == False)
vector_tri = vector[tri == True]

station_vector_unique = np.unique(numero_station(date_station(vector))).astype(np.float)
print (station_vector_unique)
print (station.count)


#%% EXPLOITATION DU TABLEAU NUMERIQUE

#Traite le tableau numeric sans les dates d'entrées
#In [20]: selector = [x for x in range(a.shape[1]) if x != 2]
#In [21]: a[:, selector]
#Out[21]: 
#array([[ 1,  2,  4],
#       [ 2,  4,  8],
#       [ 3,  6, 12]])

#Normalise tout sauf la dernière colonne dans train et test
#imp= Imputer()
#X_train[:,:-1]=imp.fit_transform(X_train[:,:-1])
#X_test[:,:-1]=imp.transform(X_test[:,:-1])

#scaler= StandardScaler().fit(X_train[:,:-1])
#X_test[:,:-1]=scaler.transform(X_test[:,:-1])
#X_train[:,:-1]= scaler.transform(X_train[:,:-1])

#Normalise le tableau de données numeriques
imp= Imputer()
X_numeric=imp.fit_transform(X_numeric)

#%% FEATURE 
vector = X_numeric.iloc[9]
vector_response= y_numeric[9]
vector_date = X_date.iloc[9]
print (vector)

#%% ANALYSE DES DONNEES

#%%
#Plot des données de tempse de sortie et d'entrée de l'usine pour train
plt.figure()
plt.scatter(id_train[y_train_num==1], date_max_piece_train[y_train_num==1], s=5, alpha=0.5, color='red')
plt.scatter(id_train[y_train_num==0], date_max_piece_train[y_train_num==0], s=1, alpha=0.2, color='blue')
plt.scatter(id_train[y_train_num==1], date_min_piece_train[y_train_num==1], s=5, alpha=0.5, color='black')
plt.scatter(id_train[y_train_num==0], date_min_piece_train[y_train_num==0], s=1, alpha=0.2, color='green')
plt.title("Bleu: Sortie si conforme, Rouge:Sortie si non conforme, Noir: Entree si non conforme, Vert: Entree si conforme")
plt.show()

#Plot des temps de sortie en fonction de l'id de l'objet et de la conformite
plt.figure()
plt.scatter(id_train[y_train_num==1], date_max_piece_train[y_train_num==1], s=5, alpha=0.5, color='red')
plt.scatter(id_train[y_train_num==0], date_max_piece_train[y_train_num==0], s=1, alpha=0.2, color='blue')
plt.title("Bleu: Sortie si conforme, Rouge:Sortie si non conforme")
plt.show()

#Plot des temps d'entrées en fonction de l'id de l'objet et de la conformite
plt.figure()
plt.scatter(id_train[y_train_num==1], date_min_piece_train[y_train_num==1], s=5, alpha=0.5, color='black')
plt.scatter(id_train[y_train_num==0], date_min_piece_train[y_train_num==0], s=1, alpha=0.2, color='green')
plt.title("Bleu: Entree si conforme, Rouge:Entree si non conforme")
plt.show()



#%%
#Plot du temps de passage dans l'usine et des conformite
plt.figure()
plt.scatter(ecart_date_train, y_train_num,s=5, alpha=0.2)
plt.title("TRAIN, Lien entre conformite et temps de passage dans l'usine")
plt.show()

plt.figure()
plt.scatter(ecart_date_test, y_test_num, s=5, alpha=0.2)
plt.title("TEST, Lien entre conformite et temps de passage dans l'usine")
plt.show()

#%%
#Plot des temps de passage dans l'usine
plt.figure()
plt.scatter(id_train[y_train_num==1], ecart_date_train[y_train_num==1], s=5, alpha=1, color='red')
plt.scatter(id_train[y_train_num==0], ecart_date_train[y_train_num==0], s=1, alpha=0.2, color='blue')
plt.title("Bleu: Temps de passage usine si conforme, Rouge: Temps de passage usine si non conforme")
plt.show()


#%%Plot des données de features
plt.figure()
plt.subplot(1,2,1)
plt.hist(nbr_feature_piece_train, normed=True)
plt.subplot(1,2,2)
plt.hist(nbr_feature_piece_test, normed=True)
plt.title("Repartition des pieces en fonction de leur nombre de features dans train et test")
plt.show()

plt.figure()
plt.subplot(1,2,1)
plt.scatter(nbr_feature_piece_train, y_train_num,s=10)
plt.subplot(1,2,2)
plt.scatter(nbr_feature_piece_test, y_test_num, s=10)
plt.title("Conformite en fonction de leur nombre de features dans train et test")
plt.show()

#%%Plot des données combinées date et feature
plt.figure()
plt.scatter(ecart_date_train[y_train_num==1],nbr_feature_piece_train[y_train_num==1],s=5, color='red', alpha=0.5)
plt.scatter(ecart_date_train[y_train_num==0],nbr_feature_piece_train[y_train_num==0],s=1, color='blue', alpha=0.2)
plt.show()

