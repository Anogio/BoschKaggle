# coding: utf-8
# Autorise les accents dans les commentaires

from math import sqrt
from pandas import *
from scipy import *
from numpy import *


def importData(filename, N):
    data = read_csv(filename, nrows=N)
    mid = int(N / 2)
    train = data.iloc[0:mid]
    test = data.iloc[mid:N]

    y_test = test["Response"].as_matrix()
    id_test = test["Id"].as_matrix()
    X_test = test.drop("Response", axis=1)
    X_test = X_test.drop("Id", axis=1)

    y_train = train["Response"].as_matrix()
    X_train = train.drop("Response", axis=1)
    X_train = X_train.drop("Id", axis=1)

    return train, X_train, y_train, X_test, y_test, id_test


# Compte des valeurs non nan
def count(V):
    return sum(not np.isnan(x) for x in V)


# Attention les tableaux d'entrée initiaux et new doivent avoir la même dimension...
def ajout_feature(X_train, X_test, X_train_new, X_test_new):
    return np.concatenate((X_train, X_train_new), axis=1), np.concatenate((X_test, X_test_new), axis=1)


# Fonctions d'évaluation des faux positifs, positifs, faux négatifs
def evaluate(y_pred, y_test):
    tp = 0  # On a bien prédit l'échec au contrôle qualité
    tn = 0  # On a bien prédit que ça n'échouait pas au controle
    fp = 0  # On a prédit que ça passait et en fait non => Faut introduire des critères supplémentaires au modèle
    fn = 0  # On a prédit que ça passait pas et en fait ça passe => Le modèle est trop restrictif...
    for i in range(len(y_pred)):
        if y_test[i] == 0:
            if y_pred[i] == 0:
                tn += 1
            else:
                fp += 1
        else:
            if y_pred[i] == 1:
                tp += 1
            else:
                fn += 1
    print("TP:%s,TN:%s,FP:%s,FN:%s" % (tp, tn, fp, fn))
    return tp, tn, fp, fn


def ratio_evaluation_modele(tp, tn, fp, fn, y_test):
    np = 0
    nn = 0
    for i in range(len(y_test)):
        if y_test[i] == 0:
            nn += 1
        else:
            np += 1
    print("Nbre Pos:%s, Nbre Neg:%s" % (np, nn))
    ratio_tp = float(tp) / float(np)  # Proche de 1 si on a bien prédit que ça échouait au test
    ratio_tn = float(tn) / float(nn)  # Proche de 1 si on a bien prédit que ça passait le test
    ratio_fp = float(fp) / float(np + nn)  # Proche de 0 si on se loupe pas
    ratio_fn = float(fn) / float(nn + nn)  # Proche de 0 si on se loupe pas
    print("Ratio TP sur Nbre Pos:%s, Ratio TN sur Nbre N:%s, Ratio FP sur NTotal:%s, Ratio FN sur NTotal:%s" % (
    ratio_tp, ratio_tn, ratio_fp, ratio_fn))
    return ratio_tp, ratio_tn, ratio_fp, ratio_fn


def mcc(tp, tn, fp, fn):
    up = (tp * tn) - (fp * fn)
    dn = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    res = up / sqrt(dn)
    print("MCC:%s" % res)
    return res
