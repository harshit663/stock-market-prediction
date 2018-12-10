# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 19:09:33 2018

@author: agraw
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics

dataset = pd.read_csv('data.csv');
y = dataset.iloc[:,1].values;
x = dataset.iloc[:,2:6].values;

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2);

from sklearn.linear_model import LinearRegression
regressor = LinearRegression();
regressor.fit(x_train,y_train);

y_pred = regressor.predict(x_test);

print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)));

plt.scatter(x[:,0],y,color='red')
plt.plot(sorted(x[:,0]),sorted(regressor.predict(x)),color='blue')
plt.title('Stock Market (Multiple Linear Regression)')
plt.xlabel('High (X)')
plt.ylabel('Open (y)')
fig1 = plt.gcf()
plt.show();
plt.draw();
fig1.savefig('mlr.png')

import statsmodels.api as sm;
x_train = np.append(arr=np.ones((16,1)).astype(int),values=x_train,axis=1);
x_opt = x_train[:,0:2];
regressor_ols = sm.OLS(y_train,x_opt).fit();
print(regressor_ols.summary());
