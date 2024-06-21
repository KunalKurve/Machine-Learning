from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.metrics import silhouette_score
import pandas as pd

milk = pd.read_csv("milk.csv",index_col=0)

from sklearn.preprocessing import StandardScaler
# Create scaler: scaler
scaler = StandardScaler().set_output(transform='pandas')
milkscaled=scaler.fit_transform(milk)

clust_DB = DBSCAN(eps=1, min_samples=2)
clust_DB.fit(milkscaled)
print(clust_DB.labels_)

clust_milk = milk.copy()
clust_milk["Clust"] = clust_DB.labels_
clust_milk.sort_values(by='Clust')

clust_milk.groupby('Clust').mean()
clust_milk.sort_values('Clust')


milkscaled['Clust'] = clust_DB.labels_
milk_scl_inliers = milkscaled[milkscaled['Clust']!=-1]
print( silhouette_score(milk_scl_inliers.iloc[:,:-1],
                 milk_scl_inliers.iloc[:,-1]) )

eps_range = [0.2,0.4,0.6,1]
mp_range = [2,3,4,5]
cnt = 0
a =[]
for i in eps_range:
    for j in mp_range:
        clust_DB = DBSCAN(eps=i, min_samples=j)
        clust_DB.fit(milkscaled.iloc[:,:5])
        if len(set(clust_DB.labels_)) > 2:
            cnt = cnt + 1
            milkscaled['Clust'] = clust_DB.labels_
            milk_scl_inliers = milkscaled[milkscaled['Clust']!=-1]
            sil_sc = silhouette_score(milk_scl_inliers.iloc[:,:-1],
                             milk_scl_inliers.iloc[:,-1])
            a.append([cnt,i,j,sil_sc])
            print(i,j,sil_sc)
    
a = np.array(a)
pa = pd.DataFrame(a,columns=['Sr','eps','min_pt','sil'])
print("Best Paramters:")
pa[pa['sil'] == pa['sil'].max()]

### Labels with best parameters
clust_DB = DBSCAN(eps=0.4, min_samples=2)
clust_DB.fit(milkscaled.iloc[:,:5])
print(clust_DB.labels_)

clust_milk = milk.copy()
clust_milk["Clust"] = clust_DB.labels_
clust_milk.sort_values(by='Clust')


m1 = clust_milk.groupby('Clust').mean()
m2 = clust_milk.sort_values('Clust')

print(m1)
print(m2)

print("Total Count: ", clust_milk['Clust'].value_counts())
print("Outliers Count: ", clust_milk['Clust'].value_counts()[-1])
print("Outliers : ", clust_milk[clust_milk['Clust'] ==  -1])