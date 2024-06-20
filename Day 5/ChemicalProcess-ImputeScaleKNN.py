import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

import warnings
warnings.filterwarnings("ignore")

chem = pd.read_csv("ChemicalProcess.csv")

y = chem['Yield']
X = chem.drop('Yield', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3,random_state=24)
imp = SimpleImputer()
lr = LinearRegression()

kfold = KFold(n_splits = 5, shuffle = True, random_state = 24)
std_scl = StandardScaler()
scl_mm = MinMaxScaler()
knn = KNeighborsRegressor()

pipe = Pipeline([('IMP',imp), ('SCL',None),('KNN',knn)])

params = {'IMP__strategy':['mean', 'median', 'most-frequent'],
          'KNN__n_neighbors': [1,2,3,4,5,6,7,8,9,10],
          'SCL': [std_scl, scl_mm, None]}

gcv = GridSearchCV(pipe, param_grid = params, cv = kfold)
gcv.fit(X, y)

print(gcv.best_params_)
print(gcv.best_score_)

best_model = gcv.best_estimator_
print(best_model)
