from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

milk = pd.read_csv("milk.csv", index_col=0)

scaler = StandardScaler().set_output(transform='pandas')
milk_scaled = scaler.fit_transform(milk)

link = "single"
mergings = linkage(milk_scaled,method=link)

dendrogram(mergings, labels=list(milk_scaled.index))
plt.title(link+" linkage-Milk")
plt.show()

################################################

clust = AgglomerativeClustering(n_clusters=3)
clust.fit(milk_scaled)

print(clust.labels_)

milk_clust = milk.copy()
milk_clust['Clust'] = clust.labels_
milk_clust['Clust'] = milk_clust['Clust'].astype(str)

print(silhouette_score(milk_scaled, clust.labels_))

Ks = [2,3,4,5]
scores = []
for i in Ks:
    clust = AgglomerativeClustering(n_clusters=i)
    clust.fit(milk_scaled)
    scores.append(silhouette_score(milk_scaled, clust.labels_))

i_max = np.argmax(scores)
print("Best no. of clusters:", Ks[i_max])
print("Best Score:", scores[i_max])

############################################### PCA #########################

from sklearn.decomposition import PCA

pca = PCA().set_output(transform = "pandas")
pcomp = pca.fit_transform(milk_scaled)

print(pca.explained_variance_ratio_*100)
pcomp["Clust"] = clust.labels_
pcomp["Clust"] = pcomp["Clust"].astype(str)

sns.scatterplot(data = pcomp, x = "pca0", y= "pca1", hue="Clust")
plt.xlabel("PCA 0")
plt.xlabel("PCA 1")
plt.title("PCA Milk")
for i in np.arange(0,milk.shape[0]):
    plt.text(pcomp.values[i,0], pcomp.values[i, 1], list(milk.index)[i])
plt.show()


df_clust = pd.DataFrame({"Animal Milk":list(milk.index), "labels" : clust.labels_})
sns.barplot(df_clust, y = milk.index, x = "labels")
plt.show()
