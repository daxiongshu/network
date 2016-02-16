import inspect
import os
import sys
code_path = os.path.join(
    os.path.split(inspect.getfile(inspect.currentframe()))[0],"/home/jiwei/xgboost-master4/wrapper")
sys.path.append(code_path)
import xgboost as xgb
import numpy as np
class xgb_classifier:
    def __init__(self,eta,min_child_weight,depth,num_round,threads=8,exist_prediction=0,exist_num_round=20,col=1,subsample=1):
        self.eta=eta
        self.subsample=subsample
        self.col=col
        self.min_child_weight=min_child_weight
        self.depth=depth
        self.num_round=num_round
        self.exist_prediction=exist_prediction
        self.exist_num_round=exist_num_round  
        self.threads=threads
        self.bst=None
    def train_predict(self,X_train,y_train,X_test,y_test=[]):
        xgmat_train = xgb.DMatrix(X_train, label=y_train,missing=-999)
        test_size = X_test.shape[0]
        param = {}
        param['objective'] = 'reg:linear'#'binary:logistic'
        param['eta'] = self.eta
        param['colsample_bytree']=self.col
        param['min_child_weight']=self.min_child_weight
        param['max_depth'] = self.depth
        param['subsample']=self.subsample
        #param['eval_metric'] = 'logloss'
        param['silent'] = 1
        param['nthread'] = self.threads
        plst = list(param.items())

        #watchlist = [ (xgmat_train,'train') ]
        num_round = self.num_round
        if len(y_test):
            xgmat_test = xgb.DMatrix(X_test,missing=-999,label=y_test)
            watchlist = [ (xgmat_train,'train'),(xgmat_test,'test') ]
        else:
            xgmat_test = xgb.DMatrix(X_test,missing=-999)
            watchlist = [ (xgmat_train,'train') ]
    
        self.bst = xgb.train( plst, xgmat_train, num_round,  watchlist)
        #xgmat_test = xgb.DMatrix(X_test,missing=-999)
    
        
        ypred = self.bst.predict(xgmat_test)
        return ypred
    def predict(self,Xt):
        xgmat_test = xgb.DMatrix(Xt,missing=-999)
        ypred = self.bst.predict(xgmat_test)
        return ypred
    def multi(self,X_train,y_train,X_test,m,y_test=None):
        xgmat_train = xgb.DMatrix(X_train, label=y_train,missing=-999)
        test_size = X_test.shape[0]
        param = {}
        param['objective'] = 'multi:softprob'

        param['bst:eta'] = self.eta
        param['colsample_bytree']=self.col
        param['min_child_weight']=self.min_child_weight
        param['bst:max_depth'] = self.depth
        param['eval_metric'] = 'mlogloss'
        param['silent'] = 1
        param['num_class']=m
        param['nthread'] = self.threads
        plst = list(param.items())
        if y_test==None:
            xgmat_test = xgb.DMatrix(X_test,missing=-999)
            watchlist = [ (xgmat_train,'train') ]
        else:
            xgmat_test = xgb.DMatrix(X_test,missing=-999,label=y_test)
            watchlist = [ (xgmat_train,'train'),(xgmat_test,'test') ]
        num_round = self.num_round

        bst = xgb.train( plst, xgmat_train, num_round, watchlist )


        ypred = bst.predict(xgmat_test).reshape(X_test.shape[0],m)
        #print ypred.shape
        return ypred
      

        



