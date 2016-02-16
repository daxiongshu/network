import csv


name='../input/test.csv'
testid={}
for c,row in enumerate(csv.DictReader(open(name))):
    testid[row['id']]=1
fname='severity_type'
name='../input/%s.csv'%fname
fea={}
feax={}
for c,row in enumerate(csv.DictReader(open(name))):
    if row['id'] not in fea:
        fea[row['id']]={}
    if int(row[fname].split()[-1]) not in fea[row['id']]:
        fea[row['id']][int(row[fname].split()[-1])]=0
    fea[row['id']][int(row[fname].split()[-1])]+=1

def sortline(line):
    tmp=[]
    for i in sorted(line.keys()):
        tmp.append('%d:%d'%(i,line[i]))
    return tmp

fo=open('rebuild7.svm','w')
name='../input/train.csv'
for c,row in enumerate(csv.DictReader(open(name))):
    if row['id'] in fea:
        line=' '.join(sortline(fea[row['id']]))
    else:
        line=''
    fo.write('%s %s\n'%(row['id'],line))
fo.close()
fo=open('rebuild7_test.svm','w')
name='../input/test.csv'
for c,row in enumerate(csv.DictReader(open(name))):
    if row['id'] in fea:
        line=' '.join(sortline(fea[row['id']]))
    else:
        line=''
    fo.write('%s %s\n'%(row['id'],line))
fo.close()


