# -*- coding: utf-8 -*-

from handling import *
from analysis_error import *

#Importation des données
train, X_train, y_train, X_test, y_test, id_test = importData("C:/Users/JLM/Desktop/Kaggle/train_numeric.csv",50000)

#Nombre de feature par objet
nbr_feature_piece_train= apply_along_axis(count,axis=1,arr=X_train)
nbr_feature_piece_test= apply_along_axis(count,axis=1,arr=X_test)

plt.figure()
plt.subplot(1,2,1)
plt.hist(nbr_feature_piece_train, normed=True)
plt.subplot(1,2,2)
plt.hist(nbr_feature_piece_test, normed=True)

#Train model
predictor = predictor(nbr_feature_piece_train,y_train,1)

#Predict
y_pred= predict(nbr_feature_piece_test,predictor,1)
tp,tn,fp,fn= evaluate(y_pred,y_test)
ratio_tp, ratio_tn, ratio_fp,ratio_fn= ratio_evaluation_modele(tp,tn,fp,fn,y_test)
mcc=mcc(tp,tn,fp,fn)
