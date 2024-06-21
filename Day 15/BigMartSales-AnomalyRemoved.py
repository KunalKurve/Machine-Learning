# %%
from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer, make_column_selector
import seaborn as sns 
import os

# %%
os.chdir(r"D:\March 2024\PML\Kaggle\BigMartSales PracticeProblem")

# %%
train = pd.read_csv("train.csv", index_col=0)
train

test = pd.read_csv("test.csv", index_col=0)
test

# %%
train.isnull().sum()

# %%
train.drop(['Outlet_Identifier'], inplace = True, axis=1)
test.drop(['Outlet_Identifier'], inplace = True, axis=1)

# %%
X_train=train.drop('Item_Outlet_Sales', axis=1)
y_train=train['Item_Outlet_Sales']

# %%
impu_num=SimpleImputer(strategy='median').set_output(transform='pandas')
impu_cat=SimpleImputer(strategy='most_frequent').set_output(transform='pandas')

ct=make_column_transformer((impu_num,make_column_selector(dtype_include=np.number)),
                        (impu_cat,make_column_selector(dtype_include=object))).set_output(
                        transform='pandas')

# %%
X_train=ct.fit_transform(X_train)
test=ct.transform(test)

# %%
X_train = pd.get_dummies(X_train,drop_first=True)
test = pd.get_dummies(test,drop_first=True)

# %%
# clf = IsolationForest(random_state=24)
# clf.fit(X_train)
# predictions = clf.predict(X_train)

# %%
# print("%age of outliers="+ str((predictions<0).mean()*100)+ "%")
# abn_ind = np.where(predictions < 0)
# print("Outliers:")
# print(X_train.index[abn_ind])

# %%
# X_train = X_train.drop(X_train.index[abn_ind])
# X_train.shape

# %%
cb=CatBoostRegressor()
cb.fit(X_train,y_train)

# %%
y_pred = cb.predict(test)
df_test =pd.read_csv(r"D:\March 2024\PML\Kaggle\BigMartSales PracticeProblem\test.csv",index_col=0)

# %%
y_pred[y_pred<0]=0

submit = pd.DataFrame({'Item_Identifier':df_test.index,'Outlet_Identiifer':df_test["Outlet_Identifier"],"Item_Outlet_Sales":y_pred})
print(submit)

# submit.to_csv("Big_mart_cat_anlytics_vidhya.csv",index=False)



