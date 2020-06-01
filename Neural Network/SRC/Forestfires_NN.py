import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation,Layer,Lambda


df= pd.read_csv('../Data/forestfires.csv')
print (df.head())

#Data Pre-Processing
print(df.isnull().sum())

df1= df.drop(["month","day"], axis=1)
print (df1)

X=df1.iloc[:,0:28]
print(X)
y=df1.iloc[:,28]
print(y)


#Train-Test-split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,random_state=0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


#model building using MLPRegressor
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(28,28,28),max_iter=300, solver='lbfgs', alpha=1e-5, activation='relu')

m1=mlp.fit(X_train,y_train)
prediction_train=m1.predict(X_train)
prediction_test = m1.predict(X_test)
print(prediction_test)
input()

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,prediction_test))
print(np.mean(y_test==prediction_test))                #>> Test-Accuracy= 0.9166666666666666
print(np.mean(y_train==prediction_train))              #>> Train-Accuracy= 1.0
print(classification_report(y_test, prediction_test))
