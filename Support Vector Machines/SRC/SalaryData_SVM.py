import pandas as pd 
import numpy as np 
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing


#Reading CSV file
df_train = pd.read_csv("../Data/SalaryData_Train(1).csv")
df_test = pd.read_csv("../Data/SalaryData_Test(1).csv")
print(df_train.head())
print(df_train.shape)
print(df_test.head())
print(df_test.shape)

#Preprocessing data
print(df_train.isnull().sum())
print(df_test.isnull().sum())


#Label_encoding for all catogorical dataset in train and test dataframe
cat_column_train =df_train[["workclass", "education", "maritalstatus", "occupation", "relationship", "race", "sex", "native"]]
# print(cat_column_train)
# # input()

cat_column_test =df_test[["workclass", "education", "maritalstatus", "occupation", "relationship", "race", "sex", "native"]]
# print(cat_column_test)
# input()

# creating instance of labelencoder
df_train1 = cat_column_train.apply(LabelEncoder().fit_transform)
# print(df_train1)
# input()

df_test1 = cat_column_test.apply(LabelEncoder().fit_transform)
# print(df_test1)
# input()

df_train2 = df_train.drop(["workclass", "education", "maritalstatus", "occupation", "relationship", "race", "sex", "native"], axis=1)
# print (df_train2)

df_test2 = df_test.drop(["workclass", "education", "maritalstatus", "occupation", "relationship", "race", "sex", "native"], axis=1)
# print (df_test2)

#Final Dataframe
final_train_df=pd.concat([df_train1, df_train2], axis=1)
print(final_train_df)

final_test_df=pd.concat([df_test1, df_test2], axis=1)
print(final_test_df)

train_X=final_train_df.iloc[:,0:13]
train_y=final_train_df.iloc[:,13]
print(train_X.head())
print(train_y.head())
input()


test_X=final_test_df.iloc[:,0:13]
test_y=final_test_df.iloc[:,13]
print(test_X.head())
print(test_y.head())
input()


# Creating SVM classification object

# kernel = linear
model_linear = SVC(kernel = "linear")
model_linear.fit(train_X,train_y)
pred_test_linear = model_linear.predict(test_X)
print(pred_test_linear)
print("Accuracy of Kernal = Linear :", np.mean(pred_test_linear==test_y))

#confusion_matrix and classification_report for model_linear
print(confusion_matrix(test_y,pred_test_linear))
print(classification_report(test_y,pred_test_linear))
input()

# Kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(train_X,train_y)
pred_test_poly = model_poly.predict(test_X)
# print(pred_test_poly)
print("Accuracy of Kernal = poly :", np.mean(pred_test_poly==test_y))

#confusion_matrix and classification_report for model_poly
print(confusion_matrix(test_y,pred_test_poly))
print(classification_report(test_y,pred_test_poly))
input()

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X,train_y)
pred_test_rbf = model_rbf.predict(test_X)
# print(pred_test_rbf)
print("Accuracy of Kernal = rbf :", np.mean(pred_test_rbf==test_y))

#confusion_matrix and classification_report for model_rbf
print(confusion_matrix(test_y,pred_test_rbf))
print(classification_report(test_y,pred_test_rbf))
input()

# kernel = sigmoid
model_sigmoid = SVC(kernel = "sigmoid")
model_sigmoid.fit(train_X,train_y)
pred_test_sigmoid = model_sigmoid.predict(test_X)
# print(pred_test_rbf)
print("Accuracy of Kernal = sigmoid :", np.mean(pred_test_sigmoid==test_y))

#confusion_matrix and classification_report for model_sigmoid
print(confusion_matrix(test_y,pred_test_sigmoid))
print(classification_report(test_y,pred_test_sigmoid))
input()