from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

pro = pd.read_csv("Protein.csv", index_col=0)

scaler = StandardScaler().set_output(transform='pandas')
pro_scaled = scaler.fit_transform(pro)

link = "single"
mergings = linkage(pro_scaled,method=link)

dendrogram(mergings, labels=list(pro_scaled.index))
plt.title(link+" linkage-Protein")
plt.show()

################################################

clust = AgglomerativeClustering(n_clusters=3)
clust.fit(pro_scaled)

print(clust.labels_)

pro_clust = pro.copy()
pro_clust['Clust'] = clust.labels_
pro_clust['Clust'] = pro_clust['Clust'].astype(str)

print(silhouette_score(pro_scaled, clust.labels_))

Ks = [2,3,4,5]
scores = []
for i in Ks:
    clust = AgglomerativeClustering(n_clusters=i)
    clust.fit(pro_scaled)
    scores.append(silhouette_score(pro_scaled, clust.labels_))

i_max = np.argmax(scores)
print("Best no. of clusters:", Ks[i_max])
print("Best Score:", scores[i_max])

############################################### PCA #########################

from sklearn.decomposition import PCA

pca = PCA().set_output(transform = "pandas")
pcomp = pca.fit_transform(pro_scaled)

print(pca.explained_variance_ratio_*100)
pcomp["Clust"] = clust.labels_
pcomp["Clust"] = pcomp["Clust"].astype(str)

sns.scatterplot(data = pcomp, x = "pca0", y= "pca1", hue="Clust")
plt.xlabel("PCA 0")
plt.xlabel("PCA 1")
plt.title("PCA Protein")
for i in np.arange(0,pro.shape[0]):
    plt.text(pcomp.values[i,0], pcomp.values[i, 1], list(pro.index)[i])
plt.show()
