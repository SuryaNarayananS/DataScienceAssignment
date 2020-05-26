import pandas as pd 
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

#Reading CSV file
df= pd.read_csv("../Data/wine.csv")
print (df.head())
print(df.isnull().sum())

# Normalizing the numerical data 
df_standardize = scale(df)
print (df_standardize)

#PCA
pca = PCA(n_components=13)
pca_values = pca.fit_transform(df_standardize)
# print (pca_values)

# The amount of variance that each PCA explains is 
var = pca.explained_variance_ratio_
print (var)
# print (pca.components_[0])
# input()

# Cumulative of variance 
var1 = np.cumsum(np.round(var,decimals = 4)*100)
print(var1)

# Variance plot for PCA components obtained 
plt.plot(var1,color="red")
plt.show()

# plot between PCA1 and PCA2 
x = pca_values[:,0]
y = pca_values[:,1]
z = pca_values[:,2]

#Visualization
plt.scatter(x,y, c='green', alpha=0.5)
plt.show()

#Clustering
new_df = pd.DataFrame(pca_values[:,0:4])

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import scipy.cluster.hierarchy as sch


k = list(range(2,50))
# print(k)
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(new_df)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(new_df.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,new_df.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))


#ploting elbow curve or Scree plot
plt.plot(k,TWSS, 'ro-');
plt.xlabel("No_of_Clusters");
plt.ylabel("total_within_SS");
plt.xticks(k)
plt.show()

kmeans = KMeans(n_clusters = 7)
kmeans.fit(new_df)
print(kmeans.labels_)

#Hirarchical Clustering
dendrogram = sch.dendrogram(sch.linkage(new_df, method  = "ward"))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

from sklearn.cluster import AgglomerativeClustering 
hc = AgglomerativeClustering(n_clusters = 7, affinity = 'euclidean', linkage ='ward')
y_hc=hc.fit_predict(new_df)
print (hc.labels_)

#visualization
plt.figure(figsize=(10, 7))
plt.scatter(new_df.iloc[:,1], new_df.loc[:,2], c=hc.labels_)
plt.show()

##############################Clustering for original data############################

k = list(range(2,50))
# print(k)
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(df.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))


#ploting elbow curve or Scree plot
plt.plot(k,TWSS, 'ro-');
plt.xlabel("No_of_Clusters");
plt.ylabel("total_within_SS");
plt.xticks(k)
plt.show()

kmeans = KMeans(n_clusters = 7)
kmeans.fit(df)
print(kmeans.labels_)

#visualization
plt.figure(figsize=(10, 7))
plt.scatter(df.iloc[:,1], df.iloc[:,2], c=kmeans.labels_)
plt.show()