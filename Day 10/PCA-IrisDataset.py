# -*- coding: utf-8 -*-
"""
Created on Tue May  7 08:15:33 2024

@author: Administrator

Topic: Principal Component Analysis

"""

import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

iris = sns.load_dataset("iris")
print(iris.head())

sns.pairplot(data=iris, hue = 'species')
plt.show()

le = LabelEncoder()

iris_train = iris.drop('species', axis = 1)
iris_test =  le.fit_transform(iris['species'])

scaler = StandardScaler().set_output(transform = "pandas")
scaler.fit(iris_train)
iris_scaled = scaler.transform(iris_train)



###################################### PCA ###################################

pca = PCA().set_output(transform = "pandas")
principalComponents = pca.fit_transform(iris_scaled)
### PCA columns are orthogonal to each other
principalComponents.corr()

print(principalComponents.var())
##variances of PC Cloumns are eigen values of variance-covariance matrix

values, vectors = np.linalg.eig(iris_scaled.cov())

print(pca.explained_variance_)
total_var = np.sum(pca.explained_variance_)
print(pca.explained_variance_/total_var)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_*100)

ys = np.cumsum(pca.explained_variance_*100)
xs = np.arange(1,6)
# plt.plot(xs, ys)
# plt.show()

principalComponents['species'] = iris['species']
sns.scatterplot(data = principalComponents, x = 'pca0', y='pca1', hue='species')
plt.show()


############################ PCA ##############################################

from pca import pca

model = pca()
results = model.fit_transform(iris_scaled, col_labels = iris.columns,
                              row_labels = list(iris.index))
model.biplot(label=True , legend=True)
for i in np.arange(0, iris.shape[0]):
    plt.text(principalComponents.values[i,0], principalComponents.values[i,1], list(iris.index)[i])
plt.show()

