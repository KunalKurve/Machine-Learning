from pca import pca
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

os.chdir(r"/content/drive/MyDrive/PML/PML/Day 10")

milk = pd.read_csv("milk.csv", index_col=0)
milk.columns

std_scl = StandardScaler().set_output(transform='pandas')
std_scl.fit(milk)
milk_scaled = std_scl.transform(milk)

model = pca()
model

pca = PCA().set_output(transform='pandas')
comp = pca.fit_transform(milk_scaled)

results = model.fit_transform(milk_scaled, col_labels = milk.columns, row_labels = list(milk.index))
model.biplot(label = True, legend = True, figsize = (15, 10))
for i in np.arange(0, milk.shape[0]):
  plt.text(comp.values[i, 0],
           comp.values[i, 1],
           list(milk.index)[i])
plt.show()

