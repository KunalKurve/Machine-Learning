import pandas as pd
import numpy as np
import os
from warnings import filterwarnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from numpy import linalg as LA
import matplotlib.pyplot as plt
import seaborn as sns
from pca import pca
import os

filterwarnings('ignore')

os.chdir(r"D:\March 2024\PML\Day 10\Unsupervised Learning on Country Data")

country = pd.read_csv('Country-data.csv', index_col=0)

# sns.pairplot(country, hue = 'country')
# plt.show()

scaler = StandardScaler().set_output(transform = "pandas")
scaler.fit(country)
country_scaled = scaler.transform(country)

pca1 = PCA().set_output(transform = "pandas")
principalComponents = pca1.fit_transform(country_scaled)

principalComponents.corr()

model = pca()
results = model.fit_transform(country_scaled, col_labels = country.columns,
                              row_labels = list(country.index))

model.biplot(label=True , legend=True)

for i in np.arange(0, country.shape[0]):
    plt.text(principalComponents.values[i,0], principalComponents.values[i,1], list(country.index)[i])
plt.show()
