import csv


name='../input/test.csv'
testid={}
for c,row in enumerate(csv.DictReader(open(name))):
    testid[row['id']]=1

name='../input/log_feature.csv'
fea={}
feax={}
for c,row in enumerate(csv.DictReader(open(name))):
    if row['id'] not in fea:
        fea[row['id']]={}
    if int(row['log_feature'].split()[-1]) not in fea[row['id']]:
        fea[row['id']][int(row['log_feature'].split()[-1])]=0
    fea[row['id']][int(row['log_feature'].split()[-1])]+=1

def sortline(line,maxf,test=False):
    tmp=[]
    for i in sorted(line.keys()):
        if maxf<i and test==False:
            maxf=i
        if test and i>maxf:
            continue

        tmp.append('%d:%d'%(i,line[i]))
    return tmp,maxf

maxf=0

fo=open('rebuild5.svm','w')
name='../input/train.csv'
for c,row in enumerate(csv.DictReader(open(name))):
    if row['id'] in fea:
        tmp,maxf=sortline(fea[row['id']],maxf)
        line=' '.join(tmp)
    else:
        line=''
    fo.write('%s %s\n'%(row['id'],line))
fo.close()
fo=open('rebuild5_test.svm','w')
name='../input/test.csv'
for c,row in enumerate(csv.DictReader(open(name))):
    if row['id'] in fea:
        tmp,maxf=sortline(fea[row['id']],maxf,test=True)
        line=' '.join(tmp)
    else:
        line=''
    fo.write('%s %s\n'%(row['id'],line))
fo.close()


