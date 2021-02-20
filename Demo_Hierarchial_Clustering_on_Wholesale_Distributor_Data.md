### Hierarchical Clustering

This method comes into practice mainly because of the fact that in case of K Means clustering we need to pre-decide the number of clusters (K centroids) which we need for the data & even to calculate WCSS, we need to iterate which is not compute friendly. 

Here, we are going to look into implementation of the bottom-up (or Agglomerative) approach of cluster building. We start by defining any sort of similarity between the datapoints. Generally, we consider the Euclidean distance. The points which are closer to each are more similar than the points which re farther away. The Algorithms starts with considering all points as separate clusters and then grouping pints together to form clusters.

Mainly, hierarchical clustering makes use of Agglomerative method of clustering using a concept of Linkage, which is the method which we use to link the sub-clusters into single clusters. There are 5 commonly used methods for linkage, while there are many others in place: 

- Single = cluster distance -> smallest pairwise distance 
- Complete = cluster distance -> largest pairwise distance 
- Average = cluster distance -> average pairwise distance 
- Centroid = cluster distance -> distance between the centroids of the clusters
- Ward = cluster distance/criteria -> minimize the variance in the cluster(s) 

**Single Linkage:** Minimal intercluster dissimilarity. Compute all pairwise dissimilarities between the observations in cluster A and the observations in cluster B, and record the smallest of these dissimilarities. Single linkage can result in extended, trailing clusters in which single observations are fused one-at-a-time.

- cluster distance is the smallest distance between any point in cluster 1 and any point in cluster 2
- highly sensitive to outliers when forming flat clusters
- works well for low-noise data with an unusual structure

**Complete Linkage:** Maximal intercluster dissimilarity. Compute all pairwise dissimilarities between the observations in cluster A and the observations in cluster B, and record the largest of these dissimilarities.

- cluster distance is the largest distance between any point in cluster 1 and any point in cluster 2
- less sensitive to outliers than single linkage

**Average Linkage:** Mean intercluster dissimilarity. Compute all pairwise dissimilarities between the observations in cluster A and the observations in cluster B, and record the average of these dissimilarities.

- cluster distance is the average distance of all pairs of points in clusters 1 and 2

**Centroid Linkage:** The dissimilarity between the centroid for cluster A (a mean vector of length p) and the centroid for cluster B. Centroid linkage can result in undesirable inversions.

- cluster distance is the distance of the centroids of both clusters

**Ward linkage:** Wikipidea says _Ward's minimum variance criterion minimizes the total within-cluster variance. To implement this method, at each step find the pair of clusters that leads to minimum increase in total within-cluster variance after merging._

- based on minimizing a variance criterion before and after merging


#**Demo: Hierarchial Clustering on Wholesale Distributor Data**



###**Problem Definition**

A wholesale distributor based out of Melbourne is using data analysis to understand how much people spend on different products. He is a novice data enthusiast who doesn't understand much about clustering. You as a data scientist is required to perform Hierarchial Clustering on the dataset to help him understand how much people spend on milk region-wise.  


###**Dataset Description**

The dataset refers to clients of a wholesale distributor. It includes the annual spending in monetary units (m.u.) on diverse product categories. It contains 8 columns with 400 instances.

Attribute Information:

>* **FRESH**: annual spending (m.u.) on fresh products (Continuous)
>* **MILK**: annual spending (m.u.) on milk products (Continuous)
>* **GROCERY**: annual spending (m.u.)on grocery products (Continuous)
>* **FROZEN**: annual spending (m.u.)on frozen products (Continuous)
>* **DETERGENTS_PAPER**: annual spending (m.u.) on detergents and paper products (Continuous)
>* **DELICATESSEN**: annual spending (m.u.)on and delicatessen products (Continuous)
>* **CHANNEL**: customers Channel - Horeca (Hotel/Restaurant/Cafe) or Retail channel (Nominal)
>* **REGION**: customers Region Lisnon, Oporto or Other (Nominal)



###**Tasks to be performed**

>* Importing Required Libraries
>* Loading the dataset
>* Analyzing and preparing the dataset
>* Understanding Hierarchical Clustering
>* Getting the Dendrograms
>* Applying Agglomerative Clustering
>* Visualizing the Clusters
>* Inference

####**Importing Required Libraries**


```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
import warnings
warnings.filterwarnings("ignore")
```

####**Loading the Dataset**


```
df = pd.read_csv('./Wholesale customers data.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Channel</th>
      <th>Region</th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicassen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>3</td>
      <td>12669</td>
      <td>9656</td>
      <td>7561</td>
      <td>214</td>
      <td>2674</td>
      <td>1338</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>3</td>
      <td>7057</td>
      <td>9810</td>
      <td>9568</td>
      <td>1762</td>
      <td>3293</td>
      <td>1776</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>3</td>
      <td>6353</td>
      <td>8808</td>
      <td>7684</td>
      <td>2405</td>
      <td>3516</td>
      <td>7844</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>3</td>
      <td>13265</td>
      <td>1196</td>
      <td>4221</td>
      <td>6404</td>
      <td>507</td>
      <td>1788</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>3</td>
      <td>22615</td>
      <td>5410</td>
      <td>7198</td>
      <td>3915</td>
      <td>1777</td>
      <td>5185</td>
    </tr>
  </tbody>
</table>
</div>



####**Analyzing and preparing the Dataset**


```
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Channel</th>
      <th>Region</th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicassen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>440.000000</td>
      <td>440.000000</td>
      <td>440.000000</td>
      <td>440.000000</td>
      <td>440.000000</td>
      <td>440.000000</td>
      <td>440.000000</td>
      <td>440.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.322727</td>
      <td>2.543182</td>
      <td>12000.297727</td>
      <td>5796.265909</td>
      <td>7951.277273</td>
      <td>3071.931818</td>
      <td>2881.493182</td>
      <td>1524.870455</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.468052</td>
      <td>0.774272</td>
      <td>12647.328865</td>
      <td>7380.377175</td>
      <td>9503.162829</td>
      <td>4854.673333</td>
      <td>4767.854448</td>
      <td>2820.105937</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>55.000000</td>
      <td>3.000000</td>
      <td>25.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>3127.750000</td>
      <td>1533.000000</td>
      <td>2153.000000</td>
      <td>742.250000</td>
      <td>256.750000</td>
      <td>408.250000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>8504.000000</td>
      <td>3627.000000</td>
      <td>4755.500000</td>
      <td>1526.000000</td>
      <td>816.500000</td>
      <td>965.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>16933.750000</td>
      <td>7190.250000</td>
      <td>10655.750000</td>
      <td>3554.250000</td>
      <td>3922.000000</td>
      <td>1820.250000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>112151.000000</td>
      <td>73498.000000</td>
      <td>92780.000000</td>
      <td>60869.000000</td>
      <td>40827.000000</td>
      <td>47943.000000</td>
    </tr>
  </tbody>
</table>
</div>




```
df.dtypes
```




    Channel             int64
    Region              int64
    Fresh               int64
    Milk                int64
    Grocery             int64
    Frozen              int64
    Detergents_Paper    int64
    Delicassen          int64
    dtype: object




```
df.isnull().sum()
```




    Channel             0
    Region              0
    Fresh               0
    Milk                0
    Grocery             0
    Frozen              0
    Detergents_Paper    0
    Delicassen          0
    dtype: int64




```
#Standardizing or Normalizing our dataset to bring all the features in the same scale
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df, columns=df.columns)
scaled_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Channel</th>
      <th>Region</th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicassen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.448652</td>
      <td>0.590668</td>
      <td>0.052933</td>
      <td>0.523568</td>
      <td>-0.041115</td>
      <td>-0.589367</td>
      <td>-0.043569</td>
      <td>-0.066339</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.448652</td>
      <td>0.590668</td>
      <td>-0.391302</td>
      <td>0.544458</td>
      <td>0.170318</td>
      <td>-0.270136</td>
      <td>0.086407</td>
      <td>0.089151</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.448652</td>
      <td>0.590668</td>
      <td>-0.447029</td>
      <td>0.408538</td>
      <td>-0.028157</td>
      <td>-0.137536</td>
      <td>0.133232</td>
      <td>2.243293</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.690297</td>
      <td>0.590668</td>
      <td>0.100111</td>
      <td>-0.624020</td>
      <td>-0.392977</td>
      <td>0.687144</td>
      <td>-0.498588</td>
      <td>0.093411</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.448652</td>
      <td>0.590668</td>
      <td>0.840239</td>
      <td>-0.052396</td>
      <td>-0.079356</td>
      <td>0.173859</td>
      <td>-0.231918</td>
      <td>1.299347</td>
    </tr>
  </tbody>
</table>
</div>



####**Getting the Dendrogram**

The Hierarchial Clustering technique can be visualized with a **Dendrogram**.

A dendrogram is a tree-like diagram showing hierarchical clustering. It shows the relationships between similar sets of data-points. We can also use the concept of Dendrogram to decide the number of clusters in Hieararchial Clustering.


```
plt.figure(figsize=(12, 8))
plt.title("Dendrograms")  
dendrogram = shc.dendrogram(shc.linkage(scaled_df, method='ward'))#ward is one of the methods that is used to calculate distance between newly formed clusters
plt.xlabel('Samples')
plt.ylabel('Distance between Samples')
```




    Text(0, 0.5, 'Distance between Samples')




    
![svg](Demo_Hierarchial_Clustering_on_Wholesale_Distributor_Data_files/Demo_Hierarchial_Clustering_on_Wholesale_Distributor_Data_14_1.svg)
    


#####**How to decide the Number of Clusters?**



From above, you can see that blue line has the maximum distance. We can select a threshold of 30 and the cut the dendrogram.


```
plt.figure(figsize=(12, 8))
plt.title("Dendrograms")  
dendrogram = shc.dendrogram(shc.linkage(scaled_df, method='ward'))#ward is used to calculate distance between newly formed clusters and can only be used with Euclidean Distance 
plt.axhline(y=30, color='k', linestyle='--')
plt.xlabel('Samples')
plt.ylabel('Distance between Samples')
plt.show()
```

**From above, you can see that the line cuts the dendrogram at two points. That means we are going to apply hierarchial clustering for two clusters**

###**Applying Hierarchial Clustering**


```
#Agglomerative Hierarchial Clustering
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')#ward is one of the methods that is used to calculate distance between newly formed clusters
cluster.fit_predict(scaled_df)#Fit the hierarchical clustering from features or distance matrix, and return cluster labels.
```

**From above, you can see two distinct values 0's and 1's beacuse we defined two clusters. 0 represents the points that belongs to the first cluster and 1 represents the points that belongs to the second cluster. These values represents the cluster labels**

###**Visualizing the Two Clusters**


```
#plot the clusters to see how actually our data has been clustered
plt.figure(figsize=(10, 7))
plt.scatter(scaled_df.iloc[:,1],scaled_df.iloc[:,3], c=cluster.labels_) 
plt.xlabel('Regions')
plt.ylabel('Milk')
plt.show()

```


    
![svg](Demo_Hierarchial_Clustering_on_Wholesale_Distributor_Data_files/Demo_Hierarchial_Clustering_on_Wholesale_Distributor_Data_23_0.svg)
    


**From above, you can see the data-points in the form of two clusters**


```

```
