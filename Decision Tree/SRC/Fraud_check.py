import numpy as np 
import pandas as pd 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

#Reading CSV file
df = pd.read_csv("../Data/Fraud_check.csv")
print (df.head())
print(df.isnull().sum())

TI=[]
for i in df["Taxable.Income"]:
	if i <= 30000:
		TI.append("Risky") 
	else:
		TI.append("Good")

df["Taxable_income"] = TI    
print(df)

df1=df.drop(["Taxable.Income"], axis=1)

cat_columns =df1[["Undergrad", "Marital.Status", "Urban"]]
df2= cat_columns.apply(LabelEncoder().fit_transform)
df3= df1.drop(["Undergrad", "Marital.Status", "Urban"], axis=1)

#Final Dataframe
final_df=pd.concat([df2, df3], axis=1)
print(final_df)
input()

X_df=final_df.iloc[:,0:5]
X= preprocessing.normalize(X_df)
y=final_df.iloc[:,5]

# Splitting the dataset into train and test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 100)

# Decision tree with entropy 
clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth = 3, min_samples_leaf = 5) 
m1=clf_entropy.fit(X_train, y_train)
y_pred = m1.predict(X_test)
print(y_pred)

print("Confusion Matrix: ", confusion_matrix(y_test, y_pred)) 
      
print ("Accuracy : ", accuracy_score(y_test,y_pred)*100)      #>>  Accuracy = 77.77777777777779
      
print("Report : ", classification_report(y_test, y_pred))
input()


# Decision tree with gini
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5) 
m2=clf_gini.fit(X_train, y_train)
y_pred = m2.predict(X_test)
print(y_pred)

print("Confusion Matrix: ", confusion_matrix(y_test, y_pred)) 
      
print ("Accuracy : ", accuracy_score(y_test,y_pred)*100)      #>>  Accuracy = 77.77777777777779
      
print("Report : ", classification_report(y_test, y_pred))
