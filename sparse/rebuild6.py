import csv

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

fo=open('rebuild6.svm','w')
name='../input/train.csv'
for c,row in enumerate(csv.DictReader(open(name))):
    tmp=['%s:1'%(row['location'].split()[-1])]
    if True:
        tmp,maxf=sortline(tmp,maxf)
        line=' '.join(tmp)
    else:
        line=''
    fo.write('%s %s\n'%(row['id'],line))
fo.close()
fo=open('rebuild6_test.svm','w')
name='../input/test.csv'
for c,row in enumerate(csv.DictReader(open(name))):
    tmp=['%s:1'%(row['location'].split()[-1])]
    if True:
        tmp,maxf=sortline(tmp,maxf,test=True)
        line=' '.join(tmp)
    else:
        line=''
    fo.write('%s %s\n'%(row['id'],line))
fo.close()


