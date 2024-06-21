from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.metrics import silhouette_score
import pandas as pd
from sklearn.preprocessing import StandardScaler

nutrient = pd.read_csv("nutrient.csv",index_col=0)

# Create scaler: scaler

scaler = StandardScaler().set_output(transform='pandas')
nutrient_scaled = scaler.fit_transform(nutrient)

clust_DB = DBSCAN(eps=1, min_samples=2)
clust_DB.fit(nutrient_scaled)
print(clust_DB.labels_)

clust_nutrient = nutrient.copy()
clust_nutrient["Clust"] = clust_DB.labels_
clust_nutrient.sort_values(by='Clust')

clust_nutrient.groupby('Clust').mean()
clust_nutrient.sort_values('Clust')


nutrient_scaled['Clust'] = clust_DB.labels_
nutrient_scaled_inliers = nutrient_scaled[nutrient_scaled['Clust'] != -1]
print("Silhouette_score: ")
print( silhouette_score(nutrient_scaled_inliers.iloc[:, : -1], nutrient_scaled_inliers.iloc[:, -1]) )

eps_range = [0.2,0.4,0.6,1]
mp_range = [2,3,4,5]
cnt = 0
a =[]

for i in eps_range:
    for j in mp_range:
        clust_DB = DBSCAN(eps=i, min_samples=j)
        clust_DB.fit(nutrient_scaled.iloc[:,:5])
        if len(set(clust_DB.labels_)) > 2:
            cnt = cnt + 1
            nutrient_scaled['Clust'] = clust_DB.labels_
            nutrient_scaled_inliers = nutrient_scaled[nutrient_scaled['Clust'] != -1]
            sil_sc = silhouette_score(nutrient_scaled_inliers.iloc[:,:-1], nutrient_scaled_inliers.iloc[:,-1])
            a.append([cnt,i,j,sil_sc])
            print("i, j, sil_sc:",i,j,sil_sc)
    
a = np.array(a)
pa = pd.DataFrame(a,columns=['Sr','eps','min_pt','sil'])
print("Best Paramters:")
pa[pa['sil'] == pa['sil'].max()]

### Labels with best parameters

clust_DB = DBSCAN(eps=0.4, min_samples=2)
clust_DB.fit(nutrient_scaled.iloc[:,:5])
print(clust_DB.labels_)

clust_nutrient = nutrient.copy()
clust_nutrient["Clust"] = clust_DB.labels_
clust_nutrient.sort_values(by='Clust')

n1 = clust_nutrient.groupby('Clust').mean()
n2 = clust_nutrient.sort_values('Clust')

print(n1)
print(n2)

print("Total Count: ", clust_nutrient['Clust'].value_counts())
print("Outliers Count: ", clust_nutrient['Clust'].value_counts()[-1])
print("Outliers : ", clust_nutrient[clust_nutrient['Clust'] ==  -1])
