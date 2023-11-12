# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the data. 
2. Choose the number of clusters.
3. Train the K-means clustering algorithm.
4. Predict the cluster labels for each data point.
5. Visualize the customer segments. 

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: SMRITI .B
RegisterNumber:  212221040156
*/
```
```
import pandas as pd 
import matplotlib.pyplot as plt
data=pd.read_csv("/content/Mall_Customers.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init="k-means++")
    kmeans.fit(data.iloc[:, 3:])
    wcss.append(kmeans.inertia_)
wcss = wcss[:10]
plt.plot(range(1,11),wcss)
plt.xlabel("No. of Clusters")
plt.ylabel("wcss")
plt.title("Elbow Method")
plt.show()
km = KMeans(n_clusters = 5)
km.fit(data.iloc[:,3:])
y_pred = km.predict(data.iloc[:,3:])
y_pred
data["cluster"] = y_pred
df0 = data[data["cluster"]==0]
df1 = data[data["cluster"]==1]
df2= data[data["cluster"]==2]
df3= data[data[ "cluster"]==3]
df4 = data[data["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="red", label="cluster0")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="black", label="cluster1")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="blue", label="cluster2")
plt.scatter(df3 ["Annual Income (k$)"],df3["Spending Score (1-100)"],c="green", label="cluster3")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="magenta", label="cluster4")
plt.legend()
plt.title("Customer Segments")
```

## Output:

## 1. data.head()
![image](https://github.com/smriti1910/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/133334803/e97cbb56-759c-4242-9584-d830cafd6148)
## 2. data.info()
![image](https://github.com/smriti1910/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/133334803/a0a4f7b8-be56-4c9d-af2f-74a47ae3201f)
## 3. data.isnull().sum()
![image](https://github.com/smriti1910/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/133334803/e139d71c-708e-4647-8fb9-e2147d805b48)
## 4. Elbow method graph
![image](https://github.com/smriti1910/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/133334803/890a8674-3f00-4f65-a94e-cdd9ceb200a9)
## 5. KMeans clusters
![image](https://github.com/smriti1910/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/133334803/f7ed3e63-3e3b-458c-91b3-dd1c00616ab8)
## 6. y_pred
![image](https://github.com/smriti1910/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/133334803/dee90cc3-b8e9-43fb-ac4f-6b9bbccb4677)
## 7. Customers Segments Graph
![image](https://github.com/smriti1910/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/133334803/4a555156-24e4-4d52-8b12-53496c2231ec)


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
