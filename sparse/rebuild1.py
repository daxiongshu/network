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
        fea[row['id']]=[]
        feax[row['id']]=set()
    if row['log_feature'].split()[-1] not in feax[row['id']]:
        feax[row['id']].add(row['log_feature'].split()[-1])     
        value='%s:%s'%(row['log_feature'].split()[-1],row['volume'])
        fea[row['id']].append(value)
maxf=0
def sortline(line,maxf,test=False):
    dic={}
    for i in line:
        if maxf<int(i.split(':')[0]) and test==False:
            maxf=int(i.split(':')[0])
        if test and int(i.split(':')[0])>maxf:
            continue
        dic[int(i.split(':')[0])]=i
    return [dic[i] for i in sorted(dic.keys())],maxf

fo=open('rebuild1.svm','w')
name='../input/train.csv'
for c,row in enumerate(csv.DictReader(open(name))):
    if row['id'] in fea:
        tmp,maxf=sortline(fea[row['id']],maxf)
        line=' '.join(tmp)
    else:
        line=''
    fo.write('%s %s\n'%(row['id'],line))
fo.close()
fo=open('rebuild1_test.svm','w')
name='../input/test.csv'
for c,row in enumerate(csv.DictReader(open(name))):
    if row['id'] in fea:
        tmp,maxf=sortline(fea[row['id']],maxf,test=True)
        line=' '.join(tmp)
    else:
        line=''
    fo.write('%s %s\n'%(row['id'],line))
fo.close()


