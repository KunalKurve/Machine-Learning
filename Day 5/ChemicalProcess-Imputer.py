import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score

import warnings
warnings.filterwarnings("ignore")

chem = pd.read_csv("ChemicalProcess.csv")

y = chem['Yield']
X = chem.drop('Yield', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3,random_state=24)

lr = LinearRegression()

###############################################################################

si = SimpleImputer(strategy='mean').set_output(transform = 'pandas')

pipe = Pipeline([('IMP',si), ('LR', lr)])

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

print("r2 score - mean: ",r2_score(y_test, y_pred))

###############################################################################

si = SimpleImputer(strategy='median').set_output(transform = 'pandas')

pipe = Pipeline([('IMP',si), ('LR', lr)])

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

print("r2 score - median: ",r2_score(y_test, y_pred))

###############################################################################

si = SimpleImputer(strategy='most_frequent').set_output(transform = 'pandas')

pipe = Pipeline([('IMP',si), ('LR', lr)])

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

print("r2 score - most-frequent: ",r2_score(y_test, y_pred))

###############################################################################

kfold = KFold(n_splits = 5, shuffle = True, random_state=24)
imp = SimpleImputer()
lr = LinearRegression()

pipe = Pipeline([('IMP',imp), ("LR", lr)])

params = {'IMP__strategy':['mean', 'median', 'most-frequent']}

gcv = GridSearchCV(pipe, param_grid = params, cv = kfold)
gcv.fit(X, y)

print(gcv.best_params_)
print(gcv.best_score_)