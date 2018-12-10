import numpy as np
import pandas as pd
from sklearn import metrics

dataset = pd.read_csv('data.csv')
y = dataset.iloc[:,1].values
X = dataset.iloc[:,2:6].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1,1))

# Splitting dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

# Fitting SVR model to the data set
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X_train, y_train.ravel())

# Predicting a new result
y_pred = sc_y.inverse_transform(regressor.predict(X_test))
print(np.sqrt(metrics.mean_squared_error(sc_y.inverse_transform(y_test),y_pred)));

import matplotlib.pyplot as plt
plt.scatter(sc_X.inverse_transform(X)[:,0], sc_y.inverse_transform(y), color='red')
plt.plot(sorted(sc_X.inverse_transform(X)[:,0]), sorted(sc_y.inverse_transform(regressor.predict(X))), color='blue')
plt.title('Stock Market (Support Vector Regression)')
plt.xlabel('High (X)')
plt.ylabel('Open (y)')
fig1 = plt.gcf()
plt.show();
plt.draw();
fig1.savefig('svr.png')

import statsmodels.api as sm
X_train = np.append(arr=np.ones((16,1)).astype(int),values=X_train,axis=1);
X_opt=X_train[:,[0,2]];
regressor_ols=sm.OLS(y_train,X_opt).fit();
print(regressor_ols.summary());
