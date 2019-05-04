lengths={}
sum=0
previous=0
dbl=0
for value in train['avg']:
    lengths[value-previous]+=1
    sum+=value-previous
    previous=value
minbar=(((sum)/train.shape[0])/2)
k=list(lengths.keys())
v=list(lengths.values())
minnum=k[v.index(max(v))]
if(minbar<=minnum):
    dbl=minbar
else:
    dbl=minnum
print(dbl)