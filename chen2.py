import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pyFTS.data import TAIEX 
data1=TAIEX.get_dataframe()
data2=data1.drop(["Openly","Highest","Lowermost","Close","Volume"],axis=1)
d = {}    
for i in range(16):
    d[i] = pd.DataFrame()
    d[i]=data2[data2["Date"].dt.year==1995+i]
for i in range(16):
    print(d[i])
    
train=d[0][d[0]["Date"].dt.month<=10]
test=d[0][d[0]["Date"].dt.month>10]


'''

1st step (Defining the universe of discourse and intervals for observations.)

'''
previous=0
sum=0
for value in train['avg']:
    sum+=value-previous
    previous=value
abl=round((sum/2)/train.shape[0])
print("abl=")
print((abl))

min_value=train['avg'].min()
max_value=train['avg'].max()

universe_of_discourse=[min_value,max_value]

u={}
ending=0
starting=universe_of_discourse[0]
i=0;
while(ending<universe_of_discourse[1]):
    ending=starting+abl
    mid=(ending+starting)/2
    u[i]=[starting,mid,ending]
    starting=mid
    i+=1

print(u)

'''

2nd step(Defining fuzzy sets for observations)

'''

# Triangular membership function

def triangular(x, a, b, c):
    return max( min( (x-a)/(b-a), (c-x)/(c-b) ), 0 )



universe=np.arange(universe_of_discourse[0],universe_of_discourse[1],0.1) 
plt.figure(figsize=(15,7))

lines = []

for fuzzy_set in u.keys():
  memberships = [ triangular(x, *u[fuzzy_set]) for x in universe]
  # plot the chart
  tmp, = plt.plot(universe, memberships, label=fuzzy_set)
  lines.append(tmp)

plt.legend(handles=lines)
plt.xlabel("Height")
plt.ylabel("Membership value")


for fuzzy_set in u.keys():
 ## print(u[fuzzy_set])
  print(fuzzy_set, triangular(7000, *u[fuzzy_set]))


i=0
set1=[]
for value in train['avg']:
    i=0
    min1=-1.0;
    for fuzzy_set in u.keys():
        if(min1<triangular(value, *u[fuzzy_set])):
            min1=triangular(value, *u[0])
            j=i
        i+=1
    set1.append(j)
print(set1)



'''

step 3(Fuzzifying observations & Establishing FLRGs)

'''

print(len(set1))

list1=list(set1)
x=0
FLR=[[] for i in range(512)]
for value in list1:
    FLR[x].append(value)
     
    x=value
print(FLR)


'''

step 4(Forecasting.)

'''
set2=[]
day=1;
start_value=4830.895
while(day<61):
    i=0
    min1=-1.0;
    for fuzzy_set in u.keys():
        if(min1<triangular(start_value, *u[fuzzy_set])):
            min1=triangular(start_value, *u[fuzzy_set])
            j=i
        i+=1
    print(j)
    set2.append(u[FLR[j][0]][1])
    start_value=u[FLR[j][0]][1]
    day+=1
    
print(train)    
print(set2)
print(test)



'''

chen's Model implemented

'''
