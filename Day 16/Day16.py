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
train = pd.read_csv("train.csv")
train
#%%
test = pd.read_csv("test.csv")
test

# %%
train.isnull().sum()

# %%
test.isnull().sum()

# %%
item_train = train[['Item_Identifier', 'Item_Weight']]
item_train

# %%
item_test = test[['Item_Identifier', 'Item_Weight']]
item_test

# %%
items = pd.concat([item_train, item_test],axis=0)
items.drop_duplicates(inplace=True)
items.dropna(inplace=True)
print(items.isnull().sum())
items

# %%
outlet_train = train[['Outlet_Identifier', 'Outlet_Establishment_Year',
                      'Outlet_Location_Type', 'Outlet_Type', 'Outlet_Size']]
outlet_train

# %%
outlet_test = test[['Outlet_Identifier', 'Outlet_Establishment_Year',
                    'Outlet_Location_Type', 'Outlet_Type', 'Outlet_Size']]
outlet_test

# %%
outlet = pd.concat([outlet_train, outlet_test],axis=0)
outlet.drop_duplicates(inplace=True)
print(outlet.isnull().sum())
outlet.fillna({'Outlet_Size':'Small'}, inplace=True)
outlet

# %%
print(outlet.shape)
print(items.shape)

# %%
data_train = pd.merge(train, items)
data_train = pd.merge(data_train, outlet)
print(data_train.isnull().sum())
data_train

#%%

data_test = pd.merge(test, items)
data_test = pd.merge(data_test, outlet)
print(data_test.isnull().sum())
data_test
# %%
data_train.drop(['Outlet_Identifier'],inplace=True,axis=1)
data_train.index = data_train['Item_Identifier']
data_test.drop(['Outlet_Identifier'],inplace=True,axis=1)
data_test.index = data_test['Item_Identifier']

# %%
X_train=data_train.drop('Item_Outlet_Sales',axis=1)
y_train=data_train['Item_Outlet_Sales']

X_test = data_test

# %%
cb=CatBoostRegressor()
cb.fit(X_train,y_train)

# %%
y_pred = cb.predict(X_test)
df_test =pd.read_csv(r"D:\March 2024\PML\Kaggle\BigMartSales PracticeProblem\test.csv",index_col=0)

# %%
y_pred[y_pred<0]=0

submit = pd.DataFrame({'Item_Identifier':df_test.index,'Outlet_Identiifer':df_test["Outlet_Identifier"],"Item_Outlet_Sales":y_pred})
print(submit)

submit.to_csv("Big_mart_cat_anlytics-Day16-1.csv",index=False)



