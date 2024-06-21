import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import make_column_transformer, make_column_selector

house =  pd.read_csv("D:\March 2024\PML\Day 7\Housing.csv")
ohc =OneHotEncoder(sparse_output=False)
scaler = StandardScaler()

ct = make_column_transformer((ohc, make_column_selector(dtype_include=object)), 
                             ('passthrough', make_column_selector(dtype_include=['int64', 'float64']))).set_output(transform='pandas')


dum_np = ct.fit_transform(house)

print(ct.get_feature_names_out())

# col_names = ct.named_transformers_['onehotencoder'].get_feature_names_out().tolist() +\
#     house.columns[house.dtypes == 'float64'].tolist() +\
#         house.columns[house.dtypes == 'int64'].tolist()

# dum_np_pd = pd.DataFrame(dum_np, columns=col_names)

#############################################################################################################################################

ohc = OneHotEncoder(sparse_output=False)

