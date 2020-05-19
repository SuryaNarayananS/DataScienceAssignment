import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import roc_curve
from sklearn import metrics


#Reading CSV file
df= pd.read_csv('../Data/creditcard.csv')
print (df.head())


df_new=df.drop(df.columns[[0]], axis = 1)
print (df_new.head())


print (df_new.describe()) # generate statistical summary for numerical columns
print (df_new.shape)
print(df_new['card'].value_counts())

#Data Preproccessing
print(df_new.isnull().sum())   #checking for null values

cat_var=['owner','selfemp']
subscription_dummy=pd.get_dummies(df,columns=cat_var)
print (subscription_dummy.head())
print(len(subscription_dummy.columns))
dummy_columns=subscription_dummy.columns
print (dummy_columns)

x = subscription_dummy.loc[:,subscription_dummy.columns !='card'] # select all rows and columns except card
y = subscription_dummy.loc[:,subscription_dummy.columns =='card'] # select all rows and column card only

#Train_Test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)# keep 0% data for test and 80% for train

# feature engineering
from sklearn.feature_selection import RFE
logreg = LogisticRegression()
rfe = RFE(logreg,10)
rfe = rfe.fit(x_train,y_train)
rfe_ranking= rfe.ranking_
print(rfe_ranking)

# manually mapping columns to ranking
cols = ['reports', 'age', 'income','expenditure', 'dependents', 'months', 'majorcards', 'active','owner_no', 'owner_yes', 'selfemp_no', 'selfemp_yes']
x =x_train[cols]
y =y_train['card']

#Train_Test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#Logistic Model
logreg = LogisticRegression()
logreg.fit(x_train,y_train)

y_pred = logreg.predict(x_test)
print (y_pred)
print(logreg.score(x_test,y_test))

from sklearn.metrics import confusion_matrix,classification_report
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
print(classification_report(y_test, y_pred))
input()

# calculating the fpr and tpr for all thresholds of the classification
# probs = model.predict_proba(x_test)[:,1]
# fpr, tpr, threshold = roc_curve(y_test, probs, pos_label=1)
# roc_auc = metrics.auc(fpr, tpr)

# ROC Curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(x_test)[:,1], pos_label=0)
log_roc_auc = roc_auc_score(y_test, logreg.predict_proba(x_test)[:,1])

plt.subplots(1, figsize=(10,10))
plt.title('Receiver Operating Characteristic - Logistic regression')
plt.plot(fpr, tpr)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()