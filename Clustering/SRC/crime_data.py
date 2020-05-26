import pandas as pd
import matplotlib.pylab as plt
from	sklearn.cluster	import	KMeans
from scipy.spatial.distance import cdist 
import numpy as np
import seaborn as sns

#Reading CSV file
df = pd.read_csv("../Data/crime_data.csv")
print (df.head())
df.columns=["City","Murder", "Assault", "UrbanPop","Rape"]
print (df.head())

# Normalized data frame (considering the numerical part of data)
def norm_func(i):
    x = (i-i.min())	/	(i.max()	-	i.min())
    return (x)

df_norm = norm_func(df.iloc[:, 1:])
print (df_norm.head())


k = list(range(2,25))
# print(k)
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(df_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))


#ploting elbow curve or Scree plot
plt.plot(k,TWSS, 'ro-');
plt.xlabel("No_of_Clusters");
plt.ylabel("total_within_SS");
plt.xticks(k)
plt.show()

#KMeans Model
model1=KMeans(n_clusters=5, init='k-means++') 
y_predicted = model1.fit(df_norm)
print (model1.labels_)
centers = model1.cluster_centers_
print (centers)


model2=KMeans(n_clusters=7, init='k-means++') 
y_predicted = model2.fit(df_norm)
print (model2.labels_)
centers = model2.cluster_centers_
print (centers)


model3=KMeans(n_clusters=9, init='k-means++') 
y_predicted = model3.fit(df_norm)
print (model3.labels_)
centers = model3.cluster_centers_
print (centers)


#visualization
plt.figure(figsize=(10, 7))
plt.scatter(df_norm["Murder"], df_norm["Rape"], c=model1.labels_)
plt.show()

plt.figure(figsize=(10, 7))
plt.scatter(df_norm["Murder"], df_norm["Rape"], c=model2.labels_)
plt.show()

plt.figure(figsize=(10, 7))
plt.scatter(df_norm["Murder"], df_norm["Rape"], c=model3.labels_)
plt.show()


#Best fit model
final_model= KMeans(n_clusters=5, init='k-means++') 
y_predicted = final_model.fit(df_norm)
print (final_model.labels_)
centers = final_model.cluster_centers_
print (centers)

plt.figure(figsize=(10, 7))
plt.scatter(df_norm["Murder"], df_norm["Rape"], c=final_model.labels_)
plt.show()