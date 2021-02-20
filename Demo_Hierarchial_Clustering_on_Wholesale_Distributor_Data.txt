import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('./Wholesale customers data.csv')
df.head()

df.describe()

df.dtypes

df.isnull().sum()

#Standardizing or Normalizing our dataset to bring all the features in the same scale
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df, columns=df.columns)
scaled_df.head()

plt.figure(figsize=(12, 8))
plt.title("Dendrograms")  
dendrogram = shc.dendrogram(shc.linkage(scaled_df, method='ward'))#ward is one of the methods that is used to calculate distance between newly formed clusters
plt.xlabel('Samples')
plt.ylabel('Distance between Samples')

plt.figure(figsize=(12, 8))
plt.title("Dendrograms")  
dendrogram = shc.dendrogram(shc.linkage(scaled_df, method='ward'))#ward is used to calculate distance between newly formed clusters and can only be used with Euclidean Distance 
plt.axhline(y=30, color='k', linestyle='--')
plt.xlabel('Samples')
plt.ylabel('Distance between Samples')
plt.show()

#Agglomerative Hierarchial Clustering
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')#ward is one of the methods that is used to calculate distance between newly formed clusters
cluster.fit_predict(scaled_df)#Fit the hierarchical clustering from features or distance matrix, and return cluster labels.

#plot the clusters to see how actually our data has been clustered
plt.figure(figsize=(10, 7))
plt.scatter(scaled_df.iloc[:,1],scaled_df.iloc[:,3], c=cluster.labels_) 
plt.xlabel('Regions')
plt.ylabel('Milk')
plt.show()



