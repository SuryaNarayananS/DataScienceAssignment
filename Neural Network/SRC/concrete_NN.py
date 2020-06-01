import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation,Layer,Lambda


df= pd.read_csv('../Data/concrete.csv')
print (df.head())

#Data Pre-Processing
print(df.isnull().sum())
X=df.iloc[:,0:8]
# print(X)
y=df.iloc[:,8]
# print(y)


#Train-Test-split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,random_state=0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


#model building using MLPRegressor
from sklearn.neural_network import MLPRegressor

mlp = MLPRegressor(hidden_layer_sizes=(8,8,8),max_iter=300, solver='lbfgs', alpha=1e-5, activation='relu')

m1=mlp.fit(X_train,y_train)
prediction_train=m1.predict(X_train)
prediction_test = m1.predict(X_test)
print(prediction_test)
input()

#RMSE value
rmse = prediction_train - y_train
print (np.sqrt(np.mean(rmse*rmse)))      #>> RMSE = 5.10252291327967
input()

#Corrrelation
corr=np.corrcoef(prediction_train,y_train)     #>> Correlation = 0.95278931]
print(corr)
