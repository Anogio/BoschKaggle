
#Custom functions import
from handling import *
from analysis_error import *

#Import data
train, X_train, y_train, X_test, y_test, id_test = importData("C:/Users/JLM/Desktop/Kaggle/train_numeric.csv",50000)

#Devise new features
X_train,stats, nbr_feature_piece = featureEngineering(X_train)
X_test,stats, nbr_feature_piece = featureEngineering(X_test,stats)

#Ajoute une dimension aux tableaux pour pouvoir concathéner des colonnes
X_train = X_train[:,np.newaxis]
X_test = X_test[:,np.newaxis]

#Ajoute les colonnes dans un grand tableau X(entrainement) et X_testeur (Test)
X, X_testeur = ajout_feature(X_train, X_test, X_train**2, X_test**2)
X, X_testeur = ajout_feature(X, X_testeur, X_train**3, X_test**3)

#Train model
predictor = predictor(X,y_train,3)

#Predict
y_pred= predict(X_testeur,predictor,3)
tp,tn,fp,fn= evaluate(y_pred,y_test)
ratio_tp, ratio_tn, ratio_fp,ratio_fn= ratio_evaluation_modele(tp,tn,fp,fn,y_test)
mcc=mcc(tp,tn,fp,fn)
