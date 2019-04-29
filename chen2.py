 

import os
os.chdir(r"/home/amarnath/Desktop/python")
train=pd.read_csv("train.csv")
np.savetxt(r'/home/amarnath/Desktop/python/temp.txt', train.values, fmt='%s')


'''
import numpy as np
import matplotlib.pyplot as plt

from pyFTS.data import TAIEX  
from pyFTS.partitioners import Grid 

from pyFTS.models import chen 
 
data = TAIEX.get_dataframe()  
plt.plot(data['Date'],data['avg'])  
temp=TAIEX.get_data()
train = temp[1:4000]         
test = temp[4000:5000]
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[10,5])
partitioner =Grid.GridPartitioner(data=train,n=10)
partitioner.plot(ax)
plt.show()

creating the chen's model
model =chen.ConventionalFTS(name="a",partitioner=partitioner)
model.fit(train)       
forecasts = model.predict(test)   

plt.plot(data['Date'].dt.year[4000:5000],test)            
plt.plot(data['Date'].dt.year[4000:5000],forecasts)
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
rmse_value=rmse(forecasts,test)
print(rmse_value)

print((rmse_value/test.mean())*100)
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pyFTS.data import TAIEX 
data1=TAIEX.get_dataframe()
data2=data1.drop(["Openly","Highest","Lowermost","Close","Volume"],axis=1)
dl = {}    
for i in range(16):
    dl[i] = pd.DataFrame()
    dl[i]=data2[data2["Date"].dt.year==1995+i]
for i in range(16):
    print(dl[i])

'''

tempn=dl[0][dl[0]["Date"].dt.month<=10]

tempn[-2:-1]['avg']
test=dl[0][dl[0]["Date"].dt.month>10]




'''
itr=0
while(itr<10):
    train=dl[itr][dl[itr]["Date"].dt.month<=10]
    test=dl[itr][dl[itr]["Date"].dt.month>10]
    itr+=1




 
    previous=0
    sum=0
    for value in train['avg']:
        sum+=value-previous
        previous=value
    abl=round((sum/2)/train.shape[0])
    #print("abl=")
    #print((abl))
    
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

    #print(u)
    
    '''
    
    2nd step(Defining fuzzy sets for observations)
    
    '''
    
    # Triangular membership function
    
    def triangular(x, a, b, c):
        return max( min( (x-a)/(b-a), (c-x)/(c-b) ), 0 )
    
    
    
    '''
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
    '''
    
    for fuzzy_set in u.keys():
        c=0
     ## print(u[fuzzy_set])
      #print(fuzzy_set, triangular(4555, *u[fuzzy_set]))
    
    
    i=0
    list1=[]
    for value in train['avg']:
        i=0
        min1=-1.0;
        for fuzzy_set in u.keys():
            if(min1<triangular(value, *u[fuzzy_set])):
                min1=triangular(value, *u[0])
                j=fuzzy_set
            i+=1
        list1.append(j)
   # print(set1)
    
        
    
    
    '''
    
    step 3(Fuzzifying observations & Establishing FLRGs)
    creating dictionary for storing FLRs
    
    '''
    from collections import defaultdict
    d= defaultdict(list)
    x=0
    for value in list1:
        d[x].append(value)
        x=value
        
     
    def sum_n(n):
        sum1=0
        for i in range(n+1):
            sum1+=i
        return sum1
    
    '''
    
    step 4(Forecasting.)
    
    '''
    list2=[]
    list3=[]
    day=1;
    start_value=float(train[-2:-1]['avg'])
    while(day<=len(test)):
        i=0
        j=0
        min1=-1.0;
        for fuzzy_set in u.keys():
            if(min1<triangular(start_value, *u[fuzzy_set])):
               # print(triangular(start_value, *u[fuzzy_set]))
                min1=triangular(start_value, *u[fuzzy_set])
                j=fuzzy_set
            i+=1
        #print(start_value)
       # print("j",j)
        #set2.append(u[FLR[j][0]][1])
        sum1=0
        sum2=0
        while(len(d[j])==0):
            j+=1
        count=1
        for i in range(len(d[j])):
            sum1+=u[d[j][i]][1]
            sum2+=(count*u[d[j][i]][1])/sum_n(len(d[j]))
        var=sum1/len(d[j])
        var2=sum2
        list2.append(var)
        list3.append(var2)
        start_value=var
        day+=1
        
    #print(train)    
    #print(set2)
    #print(test['avg'])
    
    test['avg_predicted']=list2
    test['weigted_predicted']=list3
    #plt.hist(test['avg_predicted'],test['avg'])
    '''
    calculating the error value and percentage error
    
    '''
    
    error=0
    for index, row in test.iterrows():
        error+=abs(((row['avg']-row['avg_predicted'])/row['avg'])*100)
         
            
        
    print("error is ",error/test.shape[0])
        
'''

chen's Model implemented

'''

list3=[45,89,65,9,23,88,77,76,4,4,5,6,789,9,86]
import pandas as pd
pd.DataFrame(list3)


    














 
 



















