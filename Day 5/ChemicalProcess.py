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

print(X.isnull().sum())
print(X.isnull().sum().sum())

imp = SimpleImputer(strategy = 'mean').set_output(transform = 'pandas')
X_imputed = imp.fit_transform(X)

print(X_imputed.isnull().sum().sum())

'''
.fit() - Calculate mean of every column
.fit_transform() - transform it
'''
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3,random_state=24)

imp = SimpleImputer(strategy = 'mean').set_output(transform = 'pandas')
X_imp_train = imp.fit_transform(X_train)
X_imp_test = imp.fit_transform(X_test)

lr = LinearRegression()
lr.fit(X_imp_train,y_train)
y_pred = lr.predict(X_imp_test)

print("r2 score",r2_score(y_test, y_pred))
print(X_imputed.isnull().sum().sum())
