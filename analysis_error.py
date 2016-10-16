#Calcule les features: moyenne/deviation    
def featureEngineering(X,stats=[]):
    if(stats==[]):
        means=mean(X,axis=0)
        deviations=std(X,axis=0)
    else:
        means=stats[0]
        deviations=stats[1]
    X= X-means
    X=X/deviations   
    X=X.fillna(0)
    err=apply_along_axis(linalg.norm,axis=1,arr=X)
    return err, [means,deviations], nbr_feature_piece


def predictor(X,y,k):
    regr= linear_model.LinearRegression()
    X=X.reshape(len(X),k)
    y=y.reshape(len(X),1)
    regr.fit(X,y)
    regr
    score=regr.score(X,y)
    print("Linear regression score: %s" % score)
    return regr

#k nombre de feature, X tableau d'entrée, pred = regression linéaire
def predict(X,pred,k): 
    X=X.reshape(len(X),k)
    infinite=[i for i,x in enumerate(X==inf) if x]
    X[infinite]=0
    y= pred.predict(X)
    print(mean(y))
    print(y)
    return y>mean(y)

