import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
import re
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.naive_bayes import GaussianNB as GB

df= pd.read_csv("../Data/sms_raw_NB.csv", encoding = "ISO-8859-1")

stop_words = []
with open("../Data/stop.txt") as f:
    stop_words = f.read()

stop_words = stop_words.split("\n")
print (stop_words)

#Cleaning data
def cleaning_text(i):
    i = re.sub("[^A-Za-z" "]+"," ",i).lower()
    i = re.sub("[0-9" "]+"," ",i)
    w = []
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return (" ".join(w))

#cleaned df
df.text = df.text.apply(cleaning_text)
print(df.text)

# removing empty rows 
df = df.loc[df.text != " ",:]

def split_into_words(i):
    return [word for word in i.split(" ")]


# splitting data into train and test data sets 
from sklearn.model_selection import train_test_split
df_train,df_test = train_test_split(df,test_size=0.3, random_state=0)


# Preparing df texts into word count matrix format
df_bow = CountVectorizer(analyzer=split_into_words).fit(df.text)
df_matrix = df_bow.transform(df.text)
print(df_matrix)
print(df_matrix.shape)                                    #>>  Shape = (5559, 6661)
train_df_matrix = df_bow.transform(df_train.text)
print (train_df_matrix.shape)                             #>>  Shape = (3891, 6661)
test_df_matrix = df_bow.transform(df_test.text)
print (test_df_matrix.shape)                              #>>  Shape = (1668, 6661)

#td_idf
tfidf_transformer = TfidfTransformer().fit(df_matrix)

train_tfidf = tfidf_transformer.transform(train_df_matrix)
print (train_tfidf.shape)                                  #>>  Shape = (3891, 6661)

test_tfidf = tfidf_transformer.transform(test_df_matrix)
print(test_tfidf.shape)                                    #>>  Shape = (1668, 6661)


# Preparing a naive bayes model on training data set 
# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(train_tfidf,df_train.type)
train_pred_m = classifier_mb.predict(train_tfidf)
print(train_pred_m)
accuracy_train_m = np.mean(train_pred_m==df_train.type)
print(accuracy_train_m)                                      #>>  Accuracy = 0.9681315857106142

test_pred_m = classifier_mb.predict(test_tfidf)
accuracy_test_m = np.mean(test_pred_m==df_test.type)
print(accuracy_test_m)                                       #>>  Accuracy = 0.9610311750599521
input()

# Gaussian Naive Bayes 
classifier_gb = GB()
classifier_gb.fit(train_tfidf.toarray(),df_train.type.values) # we need to convert tfidf into array format which is compatible for gaussian naive bayes
train_pred_g = classifier_gb.predict(train_tfidf.toarray())
print(train_pred_g)
accuracy_train_g = np.mean(train_pred_g==df_train.type)      #>>  Accuracy = 0.9036237471087124
print(accuracy_train_g)

test_pred_g = classifier_gb.predict(test_tfidf.toarray())
accuracy_test_g = np.mean(test_pred_g==df_test.type)
print(accuracy_test_g)                                       #>>  Accuracy = 0.8369304556354916