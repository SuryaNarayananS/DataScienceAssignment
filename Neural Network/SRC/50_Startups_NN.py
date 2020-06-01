import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Activation,Layer,Lambda


df= pd.read_csv('../Data/50_Startups.csv')
print (df.head())

#Data Pre-Processing
# LabelEncoder for catogorical column "State"
labelencoder = LabelEncoder()
df["State_Label"] = labelencoder.fit_transform(df['State'])
# print (df.head())

df1=df.drop(['State'], axis=1)
print (df1.head())
df1.columns = ["R&D_Spend", "Administration", "Marketing_Spend", "Profit", "State_Label"]

df2= df1[["R&D_Spend", "Administration", "Marketing_Spend", "State_Label","Profit"]]
print(df2.head())

X=df2.iloc[:,0:4]
# print(X)
y=df2.iloc[:,4]
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

mlp = MLPRegressor(hidden_layer_sizes=(4,4,4),max_iter=300, solver='lbfgs', alpha=1e-5, activation='relu')

m1=mlp.fit(X_train,y_train)
prediction_train=m1.predict(X_train)
prediction_test = m1.predict(X_test)
print(prediction_test)
input()

#RMSE value
rmse = prediction_train - y_train
print (np.sqrt(np.mean(rmse*rmse)))     #>> RMSE = 5144.264131860068
input()

#Correlation
corr=np.corrcoef(prediction_train,y_train)      #>> Correlation = 0.9928224]
print(corr)
