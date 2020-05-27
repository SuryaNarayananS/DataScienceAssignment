import pandas as pd 
import numpy as np 
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

#Reading CSV file
df = pd.read_csv("../Data/forestfires.csv")
print(df.head())
print(df.describe())
print(df.columns)
print(df.isnull().sum())

df1= df.drop(["month","day"], axis=1)
print (df1)

X=df1.iloc[:,0:28]
y=df1.iloc[:,28]
print(X)
print(y)
input()


#Train_Test Split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.3, random_state=0)
# print(train_X)
# print(test_X)
# print(train_y)
# print(test_y)

# Creating SVM classification object --> 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'

# kernel = linear
model_linear = SVC(kernel = "linear")
model_linear.fit(train_X,train_y)
pred_test_linear = model_linear.predict(test_X)
# print(pred_test_linear)
print("Accuracy of Kernal = Linear :", np.mean(pred_test_linear==test_y))    #>> Accuracy =  0.9871794871794872

#confusion_matrix and classification_report for model_linear
print(confusion_matrix(test_y,pred_test_linear))
print(classification_report(test_y,pred_test_linear))
input()

# Kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(train_X,train_y)
pred_test_poly = model_poly.predict(test_X)
# print(pred_test_poly)
print("Accuracy of Kernal = poly :", np.mean(pred_test_poly==test_y))     #>> Accuracy = 0.967948717948718

#confusion_matrix and classification_report for model_poly
print(confusion_matrix(test_y,pred_test_poly))
print(classification_report(test_y,pred_test_poly))
input()

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X,train_y)
pred_test_rbf = model_rbf.predict(test_X)
# print(pred_test_rbf)
print("Accuracy of Kernal = rbf :", np.mean(pred_test_rbf==test_y))     #>> Accuracy = 0.717948717948718

#confusion_matrix and classification_report for model_rbf
print(confusion_matrix(test_y,pred_test_rbf))
print(classification_report(test_y,pred_test_rbf))
input()

# kernel = sigmoid
model_sigmoid = SVC(kernel = "sigmoid")
model_sigmoid.fit(train_X,train_y)
pred_test_sigmoid = model_sigmoid.predict(test_X)
# print(pred_test_rbf)
print("Accuracy of Kernal = sigmoid :", np.mean(pred_test_sigmoid==test_y))     #>> Accuracy = 0.7051282051282052

#confusion_matrix and classification_report for model_sigmoid
print(confusion_matrix(test_y,pred_test_sigmoid))
print(classification_report(test_y,pred_test_sigmoid))