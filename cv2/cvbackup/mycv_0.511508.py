from xgb_classifier import xgb_classifier
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import AdaBoostClassifier,ExtraTreesClassifier,RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, f1_score, log_loss, make_scorer
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split,KFold,StratifiedKFold
from math import log, exp, sqrt,factorial
import numpy as np
from scipy import sparse 
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file

def rmsle(y,yp):
    return (np.mean((yp-y)**2))**0.5

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
def new_clf_train_predict(X,y,Xt):
    clf=single_model()
    clf.fit(X,y)
    return clf.predict_proba(Xt)
def cut(yp):
    yp[yp<0]=0
    yp[yp>7]=7
    yp=yp.astype(int)
    return yp
def kfold_cv(X_train, y_train,k):


    kf = StratifiedKFold(y_train,n_folds=k)

    xx=[]
    zz=[]
    ypred=np.zeros((y_train.shape[0],3))
    for train_index, test_index in kf:

        X_train_cv, X_test_cv = X_train[train_index,:],X_train[test_index,:]
        y_train_cv, y_test_cv = y_train[train_index],y_train[test_index]
        clf=xgb_classifier(eta=0.1,col=0.2,min_child_weight=1,depth=7,subsample=0.9,num_round=140)

        y_pred=clf.multi(X_train_cv,y_train_cv,X_test_cv,3,y_test=y_test_cv)
        xx.append(multiclass_log_loss(y_test_cv,y_pred))
        print xx[-1]#,y_pred.shape,zz[-1]
        ypred[test_index]=y_pred
    print xx
    print 'average:',np.mean(xx),'std',np.std(xx)
    return ypred,np.mean(xx)
mem = Memory("./mycache")

@mem.cache
def get_data(name):
    data = load_svmlight_file(name)
    return data[0], data[1]

X, _ = get_data('../sparse/rebuild1.svm')
X1, _ =get_data('../sparse/rebuild2.svm')
X2, _ = get_data('../sparse/rebuild3.svm')
X3, _ =get_data('../sparse/rebuild4.svm')
X4, _ =get_data('../sparse/rebuild5.svm')
X5, _ =get_data('../sparse/rebuild6.svm')


xx=[]
xx.append(np.sum(X.todense(),axis=1))
xx.append(np.sum(X1.todense(),axis=1))
xx.append(np.sum(X2.todense(),axis=1))
xx.append(np.sum(X3.todense(),axis=1))
xx.append(np.sum(X4.todense(),axis=1))
xx.append(np.std(X.todense(),axis=1))
xx.append(np.std(X1.todense(),axis=1))
xx.append(np.std(X2.todense(),axis=1))
xx.append(np.std(X3.todense(),axis=1))
xx.append(np.std(X4.todense(),axis=1))
#xx.append(np.sum(sparse.hstack([X,X1,X2,X3,X4],format='csr').todense(),axis=1))
#xx.append(np.max(X.todense(),axis=1)-np.min(X.todense(),axis=1))
#xx.append(np.max(X1.todense(),axis=1)-np.min(X1.todense(),axis=1))
#xx.append(np.max(X2.todense(),axis=1)-np.min(X2.todense(),axis=1))
#xx.append(np.max(X3.todense(),axis=1)-np.min(X3.todense(),axis=1))
#xx.append(np.max(X4.todense(),axis=1)-np.min(X4.todense(),axis=1))

xx=np.hstack(xx)

X=sparse.hstack([X,X1,X2,X3,X4,xx,pickle.load(open('../explore/X2.p'))],format='csr').todense()
train=pd.read_csv('../explore/train1.csv')
idname='id'
label='fault_severity'
idx=train[idname].as_matrix()
y=np.array(train[label])
import pickle
X=np.hstack([X,train.drop([label,idname],axis=1).as_matrix()])
#X=np.hstack([X,train[['location','volume']].as_matrix()])

print X.shape, y.shape
yp,score=kfold_cv(X,y,4)
print X.shape, y.shape
print yp.shape

s=pd.DataFrame({idname:idx,'predict_0':yp[:,0],'predict_1':yp[:,1],'predict_2':yp[:,2],'real':y})
s.to_csv('va.csv',index=False)

import subprocess
cmd='cp mycv.py cvbackup/mycv_%f.py'%(score)
subprocess.call(cmd,shell=True)

cmd='cp va.csv cvbackup/va_%f.csv'%(score)
subprocess.call(cmd,shell=True)


