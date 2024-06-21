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
nutrient_scaled = scaler.fit_transform(nutrient)

link = "single"
mergings = linkage(nutrient_scaled,method=link)

dendrogram(mergings, labels=list(nutrient_scaled.index))
plt.title(link+" linkage-Nutrient")
plt.show()

################################################

Ks = [2,3,4,5]
scores = []
for i in Ks:
    clust = KMeans(n_clusters=i, random_state=24, init= 'random')
    clust.fit(nutrient_scaled)
    scores.append(silhouette_score(nutrient_scaled, clust.labels_))

i_max = np.argmax(scores)
print("Best no. of clusters:", Ks[i_max])
print("Best Score:", scores[i_max])

clust1 = KMeans(n_clusters = Ks[i_max], random_state=24)
clust1.fit(nutrient_scaled)

clust_data = nutrient.copy()
clust_data['Clust'] = clust.labels_

print(clust_data.groupby('Clust').mean())
print("silhouette_score: ")
print(silhouette_score(nutrient, clust.labels_))

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
