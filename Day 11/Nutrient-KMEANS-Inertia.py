from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

nutrient = pd.read_csv("nutrient.csv", index_col=0)

scaler = StandardScaler().set_output(transform='pandas')
scaler.fit(nutrient)
nutrient_scaled = scaler.fit_transform(nutrient)

#*** ValueError: n_samples=25 should be >= n_clusters=26.
# k = np.arange(1,100)
k = np.arange(1,25)
inertia =[]

for i in k:
    clust = KMeans(n_clusters= i, random_state=24)
    clust.fit(nutrient_scaled)
    inertia.append(clust.inertia_)
    print(f"Cluster Inertia of {i}:",clust.inertia_)

print(inertia)

plt.plot(k, inertia)
plt.xlabel('No. of Clusters')
plt.ylabel('Inertia Values or WSS')
plt.title('Line Plot of K vs Inertia (WSS)')
plt.show()
#This is known as Elbow Method
plt.scatter(k, inertia, c="red")
plt.plot(k, inertia)
plt.xlabel('No. of Clusters')
plt.ylabel('Inertia Values or WSS')
plt.title('Scree Plot of K vs Inertia(WSS)')
plt.show()