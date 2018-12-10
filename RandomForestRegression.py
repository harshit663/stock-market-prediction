# -*- coding: utf-8 -*-

# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics

# Importing the dataset
dataset = pd.read_csv('data.csv')
X = dataset.iloc[:, 2:6].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10)
regressor.fit(X_train,y_train)

# Predicting results
y_pred = regressor.predict(X_test)

print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)));

# Visualising the Regression results
plt.scatter(X[:,0],y,color='red')
plt.plot(sorted(X[:,0]),sorted(regressor.predict(X)),color='blue')
plt.title('Stock Market (Random Forest Regression)')
plt.xlabel('High (X)')
plt.ylabel('Open (y)')
fig1 = plt.gcf()
plt.show();
plt.draw();
fig1.savefig('rfr.png')

import statsmodels.api as sm
X_train = np.append(arr=np.ones((16,1)).astype(int),values=X_train,axis=1);
X_opt=X_train[:,[0,2]];
regressor_ols=sm.OLS(y_train,X_opt).fit();
print(regressor_ols.summary());