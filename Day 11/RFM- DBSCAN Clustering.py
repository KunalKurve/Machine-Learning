from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.metrics import silhouette_score
import pandas as pd
from sklearn.preprocessing import StandardScaler

rfm = pd.read_csv("rfm_data_customer.csv", index_col=0)
rfm = rfm.drop(["most_recent_visit"], axis =1)

# Create scaler: scaler

scaler = StandardScaler().set_output(transform='pandas')
rfm_scaled = scaler.fit_transform(rfm)

clust_DB = DBSCAN(eps=1, min_samples=2)
clust_DB.fit(rfm_scaled)
print(clust_DB.labels_)

clust_rfm = rfm.copy()
clust_rfm["Clust"] = clust_DB.labels_
clust_rfm.sort_values(by='Clust')

clust_rfm.groupby('Clust').mean()
clust_rfm.sort_values('Clust')


rfm_scaled['Clust'] = clust_DB.labels_
rfm_scaled_inliers = rfm_scaled[rfm_scaled['Clust'] != -1]
# print("Silhouette_score: ")
# print(silhouette_score(rfm_scaled_inliers.iloc[:, : -1], rfm_scaled_inliers.iloc[:, -1]) )

eps_range = [0.2,0.4,0.6,1]
mp_range = [25,50,100]
cnt = 0
a =[]

for i in eps_range:
    for j in mp_range:
        clust_DB = DBSCAN(eps=i, min_samples=j)
        clust_DB.fit(rfm_scaled.iloc[:,:5])
        if len(set(clust_DB.labels_)) > 2:
            cnt = cnt + 1
            rfm_scaled['Clust'] = clust_DB.labels_
            rfm_scaled_inliers = rfm_scaled[rfm_scaled['Clust'] != -1]
            sil_sc = silhouette_score(rfm_scaled_inliers.iloc[:,:-1], rfm_scaled_inliers.iloc[:,-1])
            a.append([cnt,i,j,sil_sc])
            print("i, j, sil_sc:",i,j,sil_sc)
    
a = np.array(a)
pa = pd.DataFrame(a,columns=['Sr','eps','min_pt','sil'])
print("Best Paramters:")
pa[pa['sil'] == pa['sil'].max()]

### Labels with best parameters

clust_DB = DBSCAN(eps=0.4, min_samples=2)
clust_DB.fit(rfm_scaled.iloc[:,:5])
print(clust_DB.labels_)

clust_rfm = rfm.copy()
clust_rfm["Clust"] = clust_DB.labels_
clust_rfm.sort_values(by='Clust')

r1 = clust_rfm.groupby('Clust').mean()
r2 = clust_rfm.sort_values('Clust')

print(r1)
print(r2)

print("Total Count: ", clust_rfm['Clust'].value_counts())
print("Outliers Count: ", clust_rfm['Clust'].value_counts()[-1])
print("Outliers : ", clust_rfm[clust_rfm['Clust'] ==  -1])