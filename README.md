# README

## Requirement Environment

+ Please run **pip install -r requirements.txt** to make sure you have the required environment.
+ dim_reduce.py is required, it contains Autoencoder method for dimension reduction for mnist dataset.

## Documentation

```python
class Clustering(dataset='wine', method='FINCH')
```

Implement clustering with finch, K Means, DBScan on various dataset like *wine*, *mnist*, *seed*, *box*, *credit*, and evaluate the method with silhouette_score. Visualization is also provided.

**Parameters:**

+  **dataset : {'wine', 'mnist', 'seed', 'box', 'credit'}, default='wine'** 

  Specifies the dataset used in clustering experiment, see the data source on the appendix.

+ **method : {'FINCH', 'KMeans', 'dbscan'}**

  Specifies the clustering method, all of them wont specify the exact cluster number, FINCH and dbscan do not need this hype parameter and we apply a for loop to search the best cluster number for KMeans method.

**Attributes:**

+ **data : ndarray of shape (N, number of features) **

  The preprocessed (minmax scale...) features of dataset.

+ **lbs : ndarray of shape (N, 1) **

  The true cluster label of each data.

+ **res :  ndarray of shape (N, P) if method == 'FINCH' else (N, 1) where P indicates the partition**

  The predict cluster label of each data.

**Method:**

+ **load_data()** Load the specified dataset and assign feature to self.data, true labels to self.lbs.
+ **fit(req_clust=None)** fit the dataset with cluster method, if req_clust is only set when using FINCH method. It would force FINCH method to cluster data to <u>req_clust</u> clusters.
+ **reduce(data) @staticmethod** reduce high dimension data into lower space (2 dimension) for better visualization
+ **visualization()** plot the cluster results and save them into coordinate directory. Use points to represent each data sample, different color to represent their cluster.
+ **evaluation()** evaluate cluster method with silhouette_score, for FINCH method, it will print silhouette_score for different partition. 

**Examples**

```python
# You could simply run this proje
model = Clustering(dataset='box', method='FINCH')
model.load_data()
model.fit(req_clust=12)
model.visualization()
model.evaluation()
```





