import pandas as pd
import sys
import numpy as np

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



name1=sys.argv[1]
name2=sys.argv[2]

s1=pd.read_csv(name1)
s2=pd.read_csv(name2)

train=pd.read_csv('../cv2/va.csv')
y=train['real']
best,bests=None,0
preds='predict_0,predict_1,predict_2'.split(',')
for i in range(11):
    yp=s1[preds]*i+s2[preds]*(10-i)
    yp/=10
    score=multiclass_log_loss(y,yp.as_matrix())
    if bests<score:
        bests=score
        best=yp
    print i,'score',score
if len(sys.argv)>3:
    s1['Response']=best
s1.to_csv('com1.csv',index=False)
