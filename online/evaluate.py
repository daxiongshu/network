import sys
from sklearn import metrics
import pandas as pd
import numpy as np
import scipy as sp
def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll
def multiclass_log_loss(y_true, y_pred, eps=1e-15):

    predictions = np.clip(y_pred, eps, 1 - eps)

    # normalize row sums to 1
    predictions /= predictions.sum(axis=1)[:, np.newaxis]

    actual = np.zeros(y_pred.shape)
    n_samples = actual.shape[0]
    #y_true-=1
    actual[np.arange(n_samples), y_true.astype(int)] = 1
    vectsum = np.sum(actual * np.log(predictions))
    loss = -1.0 / n_samples * vectsum
    return loss

name=sys.argv[1]
s1=pd.read_csv(name)
if 'predict_0' in s1.columns.values:
    yp=s1[['predict_0','predict_1','predict_2']].as_matrix()
    print multiclass_log_loss(s1['real'],yp)
else:
    yp=np.array(s1['fault_severity'])
    print logloss(s1['real'],yp)
