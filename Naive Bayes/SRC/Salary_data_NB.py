import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB as GB
from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

df_train=pd.read_csv("../Data/SalaryData_Train.csv")
print (df_train.head())
print (df_train.isnull().sum())

df_test=pd.read_csv("../Data/SalaryData_Test.csv")
print (df_test.head())
print (df_test.isnull().sum())

#Preprocessing data
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

# Naive Bayes model
ignb = GB()
imnb = MB()

# Building and predicting at the same time 
pred_gnb = ignb.fit(train_X,train_y).predict(test_X)  # GaussianNB model
pred_mnb = imnb.fit(train_X,train_y).predict(test_X)  # Multinomal model


# Confusion matrix GaussianNB model
print (confusion_matrix(test_y,pred_gnb))             
print(pd.crosstab(test_y.values.flatten(),pred_gnb))
print(classification_report(test_y,pred_gnb))         # classification report
print(np.mean(pred_gnb==test_y.values.flatten()))     #>> Accuracy = 0.7946879150066402
input()

# Confusion matrix Multinomal model
print(confusion_matrix(test_y,pred_mnb))              
print(pd.crosstab(test_y.values.flatten(),pred_mnb))
print(classification_report(test_y,pred_mnb))         # # classification report
print(np.mean(pred_mnb==test_y.values.flatten()))     #>> Accuracy = 0.7749667994687915

