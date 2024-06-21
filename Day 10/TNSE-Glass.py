import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings

filterwarnings('ignore')

glass =  pd.read_csv("Glass.csv")

le = LabelEncoder()

y = le.fit_transform(glass['Type'])
X = glass.drop('Type', axis = 1)

tnse = TSNE (n_components=2, random_state= 24, perplexity= 20).set_output(transform='pandas')
embed_tsne = tnse.fit_transform(X)

embed_tsne['Type'] = le.fit_transform(glass['Type'])
embed_tsne['Type'] = embed_tsne['Type'].astype(str)

sns.scatterplot(data = embed_tsne, x = 'tsne0', y = 'tsne1', hue = 'Type')
plt.xlabel('TSNE0')
plt.ylabel('TSNE1')
plt.show()