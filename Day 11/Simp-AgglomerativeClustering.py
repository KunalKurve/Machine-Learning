from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns

df = pd.read_csv("simp_clust.csv", index_col=0)

plt.scatter(df['X1'],df['X2'])
for i in range(0,df.shape[0]):
    plt.text(x=df['X1'].values[i], 
             y=df['X2'].values[i],
             s=list(df.index)[i])
plt.show()

scaler = StandardScaler().set_output(transform='pandas')
df_scaled = scaler.fit_transform(df)
link = "single"
mergings = linkage(df_scaled,method=link)
dendrogram(mergings,
           labels=list(df_scaled.index))
plt.title(link+" linkage")
plt.show()

################################################
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

clust = AgglomerativeClustering(n_clusters=3)
clust.fit(df_scaled)

print(clust.labels_)

df_clust = df.copy()
df_clust['Clust'] = clust.labels_
df_clust['Clust'] = df_clust['Clust'].astype(str)

sns.scatterplot(x=df_clust['X1'], y=df_clust['X2'],
                hue=df_clust['Clust'])
for i in range(0, df.shape[0] ):
    plt.text(df_clust['X1'].values[i], 
             df_clust['X2'].values[i], 
             list(df.index)[i])
plt.show()

print(silhouette_score(df_scaled, clust.labels_))

Ks = [2,3,4,5]
scores = []
for i in Ks:
    clust = AgglomerativeClustering(n_clusters=i)
    clust.fit(df_scaled)
    scores.append(silhouette_score(df_scaled, clust.labels_))

i_max = np.argmax(scores)
print("Best no. of clusters:", Ks[i_max])
print("Best Score:", scores[i_max])