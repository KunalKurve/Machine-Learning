import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.compose import make_column_transformer, make_column_selector

house =  pd.read_csv("D:\March 2024\PML\Day 7\Housing.csv")

ohc = OneHotEncoder(sparse_output=False,drop='first')
scaler = StandardScaler()

ct = make_column_transformer((ohc, make_column_selector (dtype_include = object)),
                             ("passthrough", make_column_selector (dtype_include = ['int64','float64'])),
                              verbose_feature_names_out = False).set_output (transform ='pandas')

dum_house = ct.fit_transform(house)

X = dum_house.drop(['price'], axis = 1)
y = dum_house['price']

kfold = KFold (n_splits = 5, shuffle = True, random_state = 24)

dtc = DecisionTreeRegressor (random_state = 24)
dtc.fit(X, y)

params = {'min_samples_split': [2,4,6,10,20],
          'min_samples_leaf' : [1,5,10,15],
          'max_depth' : [None, 4, 3, 2]}
gcv = GridSearchCV (dtc, param_grid = params, cv = kfold, scoring = 'r2')
gcv.fit(X,y)

print(gcv.best_score_)
print(gcv.best_params_)

best_tree = gcv.best_estimator_

plt.figure(figsize=(15,10))        #best tree with using best parameter using gcv search
plot_tree (best_tree,feature_names = list(X.columns), class_names = ['left', 'not_left'], filled = True, fontsize = 9)
plt.show()
