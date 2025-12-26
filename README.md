# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Preparation – Load the customer dataset and select relevant features for clustering.
2. Feature Selection – Choose Annual Income and Spending Score as inputs for K-Means.
3. Model Training – Apply K-Means clustering to group customers into clusters.
4. Visualization & Output – Visualize clusters with centroids and display the clustered dataset.

## Program:
```
#Ex 10 - Implementation of K Means Clustering for Customer Segmentation
# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ------------------------------
# Step 1: Sample dataset
# ------------------------------
data = {
    'CustomerID': [1,2,3,4,5,6,7,8,9,10],
    'Gender': ['Male','Female','Female','Male','Female','Male','Male','Female','Female','Male'],
    'Age': [19,21,20,23,31,22,35,30,25,28],
    'Annual Income (k$)': [15,16,17,18,19,20,21,22,23,24],
    'Spending Score (1-100)': [39,81,6,77,40,76,6,94,3,72]
}

df = pd.DataFrame(data)

# ------------------------------
# Step 2: Select features for clustering
# ------------------------------
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# ------------------------------
# Step 3: Apply K-Means (choose clusters, e.g., 3)
# ------------------------------
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(X)  # Automatically fits and assigns clusters

# ------------------------------
# Step 4: Visualize clusters
# ------------------------------
plt.figure(figsize=(8,6))
for i in range(3):
    plt.scatter(X[df['Cluster']==i]['Annual Income (k$)'],
                X[df['Cluster']==i]['Spending Score (1-100)'],
                label=f'Cluster {i+1}')

# Plot centroids
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],
            s=200, c='yellow', label='Centroids', marker='X')

plt.title('Customer Segmentation (K-Means)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# ------------------------------
# Step 5: Show dataset with clusters
# ------------------------------
print(df)


/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Shafi Ahmed MS
RegisterNumber:  25014933
*/
```

## Output:

<img width="856" height="678" alt="image" src="https://github.com/user-attachments/assets/95b5c7fb-81bf-4792-b593-f3dab5f4739f" />

<img width="1034" height="635" alt="image" src="https://github.com/user-attachments/assets/f375218f-df1f-486a-961e-8232833bdf2c" />

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
