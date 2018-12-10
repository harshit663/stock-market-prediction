# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 19:09:33 2018

@author: agraw
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics

dataset = pd.read_csv('data.csv')
y = dataset.iloc[:,1].values
x = dataset.iloc[:,2].values
x = x.reshape(-1,1)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=8)
x_poly = poly_reg.fit_transform(x)
poly_reg.fit(x_poly,y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_poly,y,test_size=0.2)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)));

import matplotlib.pyplot as plt
X_grid=np.arange(min(x),max(x),0.1)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(x, y, color='red')
plt.plot(X_grid, regressor.predict(poly_reg.fit_transform(X_grid)),color='blue')
plt.title('Stock Market (Polynomial Linear Regression)')
plt.xlabel('High (X)')
plt.ylabel('Open (y)')
fig1 = plt.gcf()
plt.show();
plt.draw();
fig1.savefig('plr.png')

import statsmodels.api as sm
x_train = np.append(arr=np.ones((16,1)).astype(int),values=x_train,axis=1);
x_opt=x_train[:,[0,2]];
regressor_ols=sm.OLS(y_train,x_opt).fit();
print(regressor_ols.summary());
