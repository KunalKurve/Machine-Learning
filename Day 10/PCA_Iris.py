import pandas as pd
import numpy as np
import os
from warnings import filterwarnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from numpy import linalg as LA
import matplotlib.pyplot as plt
import seaborn as sns

filterwarnings('ignore')

os.chdir(r"D:\March 2024\PML\Day 10")

iris = pd.read_csv("iris.csv", index_col=0)
iris.columns

sns.pairplot(iris, hue = "Species")
plt.show()

std_scl = StandardScaler().set_output(transform='pandas')
std_scl.fit(iris.drop("Species", axis=1))
iris_std = std_scl.transform(iris.drop("Species", axis=1))

pca1 = PCA().set_output(transform='pandas')
pca_comp = pca1.fit_transform(iris_std)

print(np.cumsum(pca1.explained_variance_ratio_ * 100))

pca_comp.plot(kind='bar')
plt.xlabel("Principal Components")
plt.ylabel("Explained Variance")
plt.xticks(rotation=90)
plt.show()

pca_comp['Species'] = iris['Species']

sns.scatterplot(data=pca_comp, x='pca0', y='pca1', hue='Species')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

