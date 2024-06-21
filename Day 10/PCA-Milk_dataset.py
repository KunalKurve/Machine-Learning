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

milk = pd.read_csv("milk.csv", index_col=0)
milk.columns

std_scl = StandardScaler()

std_scl.fit(milk)
milk_scaled = std_scl.transform(milk)

pca = PCA().set_output(transform='pandas')

'''
Eigen Vector: it is a combination of Eigen values arranged in descending order
That is why the first 2 3 components are max.
Matrix : A, B, C:
  A -> m*p
  B -> p*p -> eige vector
  C -> m*p -> Values are linear combinations of A and B

'''

principalComponents = pca.fit_transform(milk_scaled)
# PCA columns are orthogonal to each other / independent of each other
principalComponents

principalComponents.corr()

# np.linalg.eig: Returns eigen values and eigen vectors of a square matrix

print(principalComponents.var())
# Varience of PC Columns are Eigen Values of var-cov matrix

values, vectors = LA.eig(milk_scaled.cov())



print(pca.explained_variance_)
# Varience: Story of the data

tot_var = np.sum(pca.explained_variance_)
print(tot_var)

print(pca.explained_variance_ / tot_var)

ys = np.cumsum(pca.explained_variance_ratio_ * 100)
xs = np.arange(1,6)
sns.pairplot(milk)

