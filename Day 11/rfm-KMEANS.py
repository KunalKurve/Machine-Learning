import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from warnings import filterwarnings
filterwarnings('ignore')

rfm = pd.read_csv("rfm_data_customer.csv", index_col=0)
rfm = rfm.drop(["most_recent_visit"], axis =1)

scaler = StandardScaler().set_output(transform='pandas')
scaler.fit(rfm)
rfm_scaled = scaler.fit_transform(rfm)

k = [2,3,4,5,6]
scores =[]

for i in k:
    clust = KMeans(n_clusters= i, random_state=24)
    clust.fit(rfm_scaled)
    scores.append(silhouette_score(rfm_scaled, clust.labels_))
    print(f"Cluster Inertia of {i}:",clust.inertia_)

i_max = np.argmax(scores)
print("\nBest no. of clusters:", k[i_max])
print("Best Score:", scores[i_max])


clust = KMeans(n_clusters=k[i_max], random_state=24)
clust.fit(rfm_scaled)

clust_data = rfm.copy()
clust_data['Clust'] = clust.labels_

print(clust_data.groupby('Clust').mean())
print(clust_data['Clust'].value_counts())

###############################################################################

g = sns.FacetGrid(clust_data, col="Clust")
g.map(sns.scatterplot, "number_of_orders", "revenue")
plt.show()

###############################################################################

g = sns.FacetGrid(clust_data, col="Clust")
g.map(sns.scatterplot, "number_of_orders", "recency_days")
plt.show()

###############################################################################

g = sns.FacetGrid(clust_data, col="Clust")
g.map(sns.scatterplot, "recency_days", "revenue")
plt.show()

###############################################################################

g = sns.FacetGrid(clust_data, col="Clust")
g.map(sns.histplot, 'revenue')
plt.show()

###############################################################################

g = sns.FacetGrid(clust_data, col="Clust")
g.map(sns.histplot, 'number_of_orders')
plt.show()

###############################################################################

g = sns.FacetGrid(clust_data, col="Clust")
g.map(sns.histplot, 'recency_days')
plt.show()

##############################################################################

plt.scatter(k, scores, c="red")
plt.plot(k, scores)
plt.xlabel('No. of Clusters')
plt.ylabel('Inertia Values or WSS')
plt.title('Scree Plot of K vs Inertia(WSS)')
plt.show()

##############################################################################

clust_corr1 = clust_data.groupby('Clust').corr()
sns.heatmap(clust_corr1, cmap="YlGnBu", annot=True)
plt.show()

##############################################################################
