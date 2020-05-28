import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

#reading CSV file
df=pd.read_csv("../Data/Zoo.csv")
print(df.head())
print (df.shape)
print(df.isnull().sum())

#splitting datasets into features(x) and labels(y)
X = df.iloc[:, 1:17].values
y = df.iloc[:, 17].values
# print (X)
# print (y)

#Train_Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 1)


scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
print (X_train)
print (X_test)


#model1 with 5 nearest neighbour(randomly choosing)
knn = KNC(n_neighbors=5)
m1=knn.fit(X_train,y_train)
pred = m1.predict(X_test)
print (pred)                          #>> accuracy = 0.9047619047619048

print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

# Calculating the accuracy of model1
print(m1.score(X_test, y_test))

error = []

# Calculating error for K values between 1 and 40
for i in range(1, 40):
    knn = KNC(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

#visualization
plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()

#model2 with k=3 as it has the lowest mean error from the graph
knn = KNC(n_neighbors=3)
m2=knn.fit(X_train,y_train)
pred = m2.predict(X_test)
# print (pred)

print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

# Calculating the accuracy of model2
print(m2.score(X_test, y_test))        #>> accuracy = 0.9523809523809523