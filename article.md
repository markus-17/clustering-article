### Stuff this article aims to cover

* KMeans
* Silhouette Score
* Marketing Segmentation


# Introduction

What is clustering? Clustering is a category of unsupervised machine learning models.

What is unsupervised learning then? Unsupervised learning is a class of algorithms that take a dataset of unlabeled examples and for each feature vector **x** as input either transforms it into another vector or into a value that can be used to solve a practical problem. For example, in **clustering** this useful value is the id of the cluster.

In other words, clustering algorithms seek to learn, from the properties of the data, an optimal division or discrete labeling of groups of points. There are a lot of clustering algorithms already implemented in libraries such as **scikit-learn** and others, so no need to worry about that part. One of the simplest to understand clustering algorithms is **KMeans**.


# KMeans

The **KMeans** algorithm searches for *k* number of clusters (the number *k* must be known in advance) within an unlabeled multidimensional dataset. For the **KMeans** algorithm the optimal clustering has the following properties:
    
* The "cluster center" is the arithmetic mean of all the points belonging to the cluster
* Each point is closer to its own cluster center than to other cluster centers

**KMeans** is implemented in **sklearn.cluster.KMeans**, so let's generate a two dimensional sample dataset and observe the k-means results.

```py
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=420, centers=3, cluster_std=0.40, random_state=0)
plt.scatter(X[ : , 0], X[ : , 1], s=15)
```

![image1](./images/image1.png)


Now, let's apply **KMeans** on this sample dataset. We will visualize the results by coloring each cluster using a different color. We will also plot the cluster centers.

```py
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
id = kmeans.predict(X)

plt.scatter(X[ : , 0], X[ : , 1], c=id, s=15, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[ : , 0], centers[ : , 1], c='red', s=100, alpha=0.5)
```

![image2](./images/image2.png)

As you can see the algorithm assigns the points to the clusters very similarly to how we might assign them by eye.

How does the algorithm work? **KMeans** uses an iterative approach known as *expectation–maximization*. In the context of the **KMeans** algorithm, the *expectation–maximization* consists of the following steps:

1. Randomly guess some cluster centers
2. Repeat until converged
    <ol type='a'>
        <li><i>E-Step<i>: assign points to the nearest cluster center</li>
        <li><i>M-Step<i>: set the cluster centers to the mean</li>
    </ol>

The **KMeans** algorithm is simple enough for us to write a really basic implementation of it in just a few lines of code.