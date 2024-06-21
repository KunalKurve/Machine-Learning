import pandas as pd
import numpy as np
import os
from warnings import filterwarnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from numpy import linalg as LA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

filterwarnings('ignore')

os.chdir(r"D:\March 2024\PML\Day 11\Unsupervised Learning on Country Data")

country = pd.read_csv('Country-data.csv', index_col=0)
scaler = StandardScaler().set_output(transform = "pandas")
country_scaled = scaler.fit_transform(country)

###############################################################################

link = "ward"
mergings = linkage(country_scaled, method=link)

dendrogram(mergings, labels=list(country_scaled.index))
plt.title(link+" linkage-Country")
plt.show()

###############################################################################

link = "average"
mergings = linkage(country_scaled, method=link)

dendrogram(mergings, labels=list(country_scaled.index))
plt.title(link+" linkage-Country")
plt.show()

###############################################################################

# clust = KMeans(n_clusters=3)
# clust.fit(country_scaled)

# #testing distance_threshold
# clust = KMeans(n_clusters=None, distance_threshold=1.0)
# clust.fit(country_scaled)


# print(clust.labels_)

# country_clust = country.copy()
# country_clust['Clust'] = clust.labels_
# country_clust['Clust'] = country_clust['Clust'].astype(str)

# print(silhouette_score(country, clust.labels_))

###############################################################################

Ks = [2,3,4,5]
scores = []
for i in Ks:
    clust = KMeans(n_clusters=i, random_state=24, init= 'random')
    clust.fit(country_scaled)
    scores.append(silhouette_score(country_scaled, clust.labels_))

i_max = np.argmax(scores)
print("Best no. of clusters:", Ks[i_max])
print("Best Score:", scores[i_max])

clust1 = KMeans(n_clusters = Ks[i_max], random_state=24)
clust1.fit(country_scaled)

clust_data = country.copy()
clust_data['Clust'] = clust.labels_

print(clust_data.groupby('Clust').mean())
print("silhouette_score: ")
print(silhouette_score(country, clust.labels_))

###############################################################################

pca = PCA().set_output(transform = "pandas")
pcomp = pca.fit_transform(country_scaled)

print(pca.explained_variance_ratio_*100)
pcomp["Clust"] = clust.labels_
pcomp["Clust"] = pcomp["Clust"].astype(str)

cust_country = clust_data.groupby('Clust').mean()
# cust_country.to_csv("Country-KMEANS.csv", index = True)

###############################################################################

child_mort = clust_data.groupby('Clust')['child_mort'].mean()
print(child_mort)
g = sns.FacetGrid(clust_data, col="Clust")
g.map(sns.histplot, "child_mort")
plt.show()

###############################################################################

health = clust_data.groupby('Clust')['health'].mean()
print(health)
g = sns.FacetGrid(clust_data, col="Clust")
g.map(sns.barplot, 'health')
plt.show()

###############################################################################

life_exp = clust_data.groupby('Clust')['life_expec'].mean()
print(life_exp)
g = sns.FacetGrid(clust_data, col="Clust")
g.map(sns.boxplot, 'life_expec')
plt.show()

###############################################################################

g = sns.FacetGrid(clust_data, col="Clust")
g.map(sns.scatterplot, "health", "child_mort")
plt.show()

###############################################################################

g = sns.FacetGrid(clust_data, col="Clust")
g.map(sns.scatterplot, "health", "income")
plt.show()

##############################################################################

clust_corr = clust_data.corr()
sns.heatmap(clust_corr, cmap="YlGnBu", annot=True)
plt.show()

##############################################################################

clust_corr1 = clust_data.groupby('Clust').corr()
sns.heatmap(clust_corr1, cmap="YlGnBu", annot=False)
plt.show()

##############################################################################

clust_corr1.to_csv("Country-KMEANS1.csv", index = True)

##############################################################################

# sns.scatterplot(data = pcomp, x = "pca0", y= "pca1", hue="Clust")
# plt.xlabel("PCA 0")
# plt.xlabel("PCA 1")
# plt.title("PCA Country")
# for i in np.arange(0, country.shape[0]):
#     plt.text(pcomp.values[i,0], pcomp.values[i, 1], list(country.index)[i])
# plt.show()

# df_clust = pd.DataFrame({"Country":list(country.index), "labels" : clust.labels_})
# sns.barplot(df_clust, y = "Country", x = "labels")
# plt.show()

