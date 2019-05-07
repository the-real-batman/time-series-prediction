import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from pyFTS.data import TAIEX 
data1=TAIEX.get_dataframe()
data2=data1.drop(["Openly","Highest","Lowermost","Close","Volume"],axis=1)
dl = {}    
for i in range(10):
    dl[i] = pd.DataFrame()
    dl[i]=data2[data2["Date"].dt.year==2004+i]
for i in range(10):
    print(dl[i])

chen=[]
weighted=[]
itr=0
sum_of_percentage_error_dbl_chen=0
sum_of_percentage_error_dbl_weighted=0
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
    from collections import defaultdict
    lengths= defaultdict(list)
    sum=0
    previous=0
    dbl=0
    max8=-1
    for value in train['avg']:
        lengths[int(abs(value-previous))].append(0)
        if(max8<len(lengths[int(abs(value-previous))])):
            max8=len(lengths[int(abs(value-previous))])
            #print(value)
            #print(previous)
            #print(len(lengths[int(abs(value-previous))]))
            dbl=int(abs(value-previous))
        sum+=value-previous
        previous=value
    print(dbl)
    min_value=train['avg'].min()
    max_value=train['avg'].max()
    
    universe_of_discourse=[min_value,max_value]
    
    u={}
    ending=0
    starting=universe_of_discourse[0]
    i=0;
    while(ending<universe_of_discourse[1]):
        ending=starting+dbl
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
    
    step 4(Forecasting.) conventional model
    
    '''
    list2=[]
    list4=[]
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
        #print("j",j)
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
        list4.append(var2)
        d[start_value].append(var)
        start_value=var
        day+=1
    
    '''
    forecasting
    weighted model
    
    '''
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
            count+=1
        var=sum1/len(d[j])
        var2=sum2
        #list2.append(var)
        list3.append(var2)
        start_value=var2
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
    
    error1=0
    error2=0
    sum6=0
    sum7=0
    for index, row in test.iterrows():
        sum6+=(row['avg']-row['avg_predicted'])**2
        sum7+=(row['avg']-row['weigted_predicted'])**2
        error1+=abs(((row['avg']-row['avg_predicted'])/row['avg'])*100)
        error2+=abs(((row['avg']-row['weigted_predicted'])/row['avg'])*100)
         
    
    rmse1=(sum6/test.shape[0])**.5
    rmse2=(sum7/test.shape[0])**.5
    
    chen.append(rmse1)
    weighted.append(rmse2)
    print("error is ",error1/test.shape[0])
    print("error is ",error2/test.shape[0])
    sum_of_percentage_error_dbl_chen+=(error1/test.shape[0])
    sum_of_percentage_error_dbl_weighted+=(error2/test.shape[0])
        

d1=pd.DataFrame(chen)
d2=pd.DataFrame(weighted)
d1['chen']=chen
d1['weighted']=weighted
x=[2004,2005,2006,2007,2008,2009,2010,2011,2012,2013]
import matplotlib.pyplot as pyplot
pyplot.bar(x,d1['chen'],width=0.5,align='center',label='Chen')
pyplot.bar(x,d1['weighted'],width=0.5,align='edge',label='Weighted')
pyplot.xlabel('year')
pyplot.ylabel('RMSE')
pyplot.legend()
plt.title('chen vs weighted for distribution based length')
pyplot.show()

print(d1['chen'].sum()/d1.shape[0])
print(d1['weighted'].sum()/d1.shape[0])
print(sum_of_percentage_error_dbl_chen/10)
print(sum_of_percentage_error_dbl_weighted/10)


    














 
 




















