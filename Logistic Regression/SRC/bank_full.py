import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import roc_curve

# sns.set(style="white")
# sns.set(style="whitegrid",color_codes=True)
# # %matplotlib inline

#Reading CSV file
df= pd.read_csv('../Data/bank-full.csv',delimiter=";")
print (df.head())

print(df.isnull().sum())

print (df.describe()) # generate statistical summary for numerical columns
print (df.shape)
print(df['y'].value_counts())

# Explore and Visualization
sns.countplot(x='y',data=df)
plt.show()
# input()

print (df.groupby('y').mean()) # group y column by Yes or no then apply mean on numerical col


df.groupby('education').mean()
pd.crosstab(df.education,df.y).plot(kind='bar')
plt.title('purchace freq with education')
plt.xlabel('education')
plt.ylabel('Freq of purchase')
plt.show()

df.groupby('duration').mean()
pd.crosstab(df.marital,df.y).plot(kind='bar')
plt.title('purchace freq with marital')
plt.xlabel('marital')
plt.ylabel('Freq of purchase')
plt.show()

df.groupby('duration').mean()
pd.crosstab(df.housing,df.y).plot(kind='bar')
plt.title('purchace freq with housing')
plt.xlabel('loan')
plt.ylabel('Freq of purchase')
plt.show()

df.groupby('duration').mean()
pd.crosstab(df.contact,df.y).plot(kind='bar',stacked='True')
plt.title('purchace freq with contact')
plt.xlabel('contact')
plt.ylabel('Freq of purchase')
plt.show()

cat_vars=['job','marital','education','default','housing','loan','contact','month','poutcome']
subscription_dummy=pd.get_dummies(df,columns=cat_vars)
print (subscription_dummy.head())
print(len(subscription_dummy.columns))
dummy_columns =subscription_dummy.columns
print (dummy_columns)

subscription_dummy.columns.values

#Train-Test split
x = subscription_dummy.loc[:,subscription_dummy.columns !='y'] # select all rows and columns except y
y = subscription_dummy.loc[:,subscription_dummy.columns =='y'] # select all rows and column y only 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)# keep 0% data for test and 80% for train


# feature engineering
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
rfe = RFE(logreg,10)
rfe = rfe.fit(x_train,y_train)
rfe_ranking= rfe.ranking_
print(rfe_ranking)



# manually map columns to ranking
cols = ['month_aug','month_dec', 'month_jan','month_may', 'month_oct', 'month_sep','poutcome_failure', 'poutcome_other', 'poutcome_success','poutcome_unknown']
x =x_train[cols]
y =y_train['y']
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
# print(x_train, x_test, y_train, y_test)

#Logistic Model
logreg = LogisticRegression()
model=logreg.fit(x_train,y_train)

y_pred = model.predict(x_test)
print (y_pred)
print(logreg.score(x_test,y_test))

from sklearn.metrics import confusion_matrix,classification_report
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
print(classification_report(y_test, y_pred))
input()

# calculating the fpr and tpr for all thresholds of the classification
probs = model.predict_proba(x_test)[:,1]
fpr, tpr, threshold = roc_curve(y_test, probs, pos_label=0)
roc_auc = metrics.auc(fpr, tpr)

# ROC Curve
import matplotlib.pyplot as plt
plt.subplots(1, figsize=(10,10))
plt.title('Receiver Operating Characteristic - Logistic regression')
plt.plot(fpr, tpr)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

