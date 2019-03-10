# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 10:02:51 2018

@author: ritesh
"""
import numpy as np
'''for ploting the graphs'''
import matplotlib.pyplot as plt
'''pyFTS is a python library to provide methods
to deal with fuzzy time series prediction'''
from pyFTS.data import TAIEX  

'''importing grid to create partitions among set'''
from pyFTS.partitioners import Grid 

'''importing chen for convetionanl fuzzy time series model'''

from pyFTS.models import chen 
'''getting the whole TAIEX dataframe'''
data = TAIEX.get_dataframe()  

'''Data Visualistion'''
plt.plot(data['Date'],data['avg']) 

'''getting target variable'''  

temp=TAIEX.get_data()
train = temp[1:4000]         
test = temp[4000:5000]
'''Universe of Discourse Partitioner'''


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[10,5])

partitioner =Grid.GridPartitioner(data=train,n=10)
partitioner.plot(ax)
plt.show()
'''creating the chen's model'''
 


 
model =chen.ConventionalFTS(name="a",partitioner=partitioner)
'''fitting data for training''' 
model.fit(train)       
''' Time series forecasting''' 
forecasts = model.predict(test)   

'''visualising the result for having rough idea of accuracy'''
plt.plot(data['Date'].dt.year[4000:5000],test)            
plt.plot(data['Date'].dt.year[4000:5000],forecasts)
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
rmse_value=rmse(forecasts,test)
print(rmse_value)

print((rmse_value/test.mean())*100)

 
