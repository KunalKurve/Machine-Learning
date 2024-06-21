from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

nutrient = pd.read_csv("nutrient.csv", index_col=0)

scaler = StandardScaler().set_output(transform='pandas')
nutrient_scaled = scaler.fit_transform(nutrient)

link = "single"
mergings = linkage(nutrient_scaled,method=link)

dendrogram(mergings, labels=list(nutrient_scaled.index))
plt.title(link+" linkage-Nutrient")
plt.show()

################################################

clust = AgglomerativeClustering(n_clusters=4)
clust.fit(nutrient_scaled)

print(clust.labels_)

nutrient_clust = nutrient.copy()
nutrient_clust['Clust'] = clust.labels_
nutrient_clust['Clust'] = nutrient_clust['Clust'].astype(str)

print(silhouette_score(nutrient_scaled, clust.labels_))

Ks = [2,3,4,5]
scores = []
for i in Ks:
    clust = AgglomerativeClustering(n_clusters=i)
    clust.fit(nutrient_scaled)
    scores.append(silhouette_score(nutrient_scaled, clust.labels_))

i_max = np.argmax(scores)
print("Best no. of clusters:", Ks[i_max])
print("Best Score:", scores[i_max])

############################################### PCA #########################

from sklearn.decomposition import PCA

pca = PCA().set_output(transform = "pandas")
pcomp = pca.fit_transform(nutrient_scaled)

print(pca.explained_variance_ratio_*100)
pcomp["Clust"] = clust.labels_
pcomp["Clust"] = pcomp["Clust"].astype(str)

sns.scatterplot(data = pcomp, x = "pca0", y= "pca1", hue="Clust")
plt.xlabel("PCA 0")
plt.xlabel("PCA 1")
plt.title("PCA Nutrient")
for i in np.arange(0,nutrient.shape[0]):
    plt.text(pcomp.values[i,0], pcomp.values[i, 1], list(nutrient.index)[i])
plt.show()
