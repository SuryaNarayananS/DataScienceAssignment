import pandas as pd
import matplotlib.pylab as plt
from	sklearn.cluster	import	KMeans
from scipy.spatial.distance import cdist 
import numpy as np
import seaborn as sns
import scipy.cluster.hierarchy as sch

#Reading excel file
df = pd.read_excel('../Data/EastWestAirlines.xlsx',sheet_name="data")
print (df.head())
print (df.shape)

#Preprocessing
df = df.drop(["ID#"], axis =1)
print (df.head())

print(df.isnull().sum())


# Normalized data frame (considering the numerical part of data)
def norm_func(i):
    x = (i-i.min())	/	(i.max()	-	i.min())
    return (x)

df_norm = norm_func(df.iloc[:])
print (df_norm.head())
print (df_norm.shape)


k = list(range(2,50))
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

#KMeans Model1
model1=KMeans(n_clusters=9) 
y_predicted = model1.fit(df_norm)
print (model1.labels_)
centers = model1.cluster_centers_
print (centers)

#KMeans Model2
model2=KMeans(n_clusters=16) 
y_predicted = model2.fit(df_norm)
print (model2.labels_)
centers = model2.cluster_centers_
print (centers)


#KMeans Model3
model3=KMeans(n_clusters=20) 
y_predicted = model3.fit(df_norm)
print (model3.labels_)
centers = model3.cluster_centers_
print (centers)


#Hirarchical Clustering
dendrogram = sch.dendrogram(sch.linkage(df_norm, method  = "ward"))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

from sklearn.cluster import AgglomerativeClustering 
hc = AgglomerativeClustering(n_clusters = 9, affinity = 'euclidean', linkage ='ward')
y_hc=hc.fit_predict(df_norm)
print (hc.labels_)

#visualization
plt.figure(figsize=(10, 7))
plt.scatter(df_norm["Balance"], df_norm["Bonus_miles"], c=hc.labels_)
plt.show()