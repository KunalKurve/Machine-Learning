import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings("ignore")

boston = pd.read_csv('Boston.csv')
print(boston)

X = boston.drop('medv', axis = 1)
y = boston['medv']

kfold = KFold(n_splits = 5, shuffle = True, random_state = 24)

std_scl = StandardScaler()
scl_mm = MinMaxScaler()
knn = KNeighborsRegressor()

pipe = Pipeline([('SCL',None),('KNN',knn)])

params = {'KNN__n_neighbors': [1,2,3,4,5,6,7,8,9,10],'SCL': [std_scl, scl_mm, None]}

gcv = GridSearchCV(pipe, param_grid = params, cv = kfold, scoring = 'r2')
gcv.fit(X, y)

print(gcv.best_params_)
print(gcv.best_score_)

best_model = gcv.best_estimator_
print(best_model)
