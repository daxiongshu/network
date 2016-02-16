import pandas as pd
import sys

s1=pd.read_csv(sys.argv[1])
s2=pd.read_csv(sys.argv[2])
s1['predict_0']=1-s1['fault_severity']
s1['predict_2']=s2['fault_severity']
s1['predict_1']=1-s1['predict_0']-s1['predict_2']
s1[['id','real','predict_0','predict_1','predict_2']].to_csv('f%s'%sys.argv[1],index=False)

