import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

kyp = pd.read_csv('Kyphosis.csv')
le = LabelEncoder()

y = le.fit_transform(kyp['Kyphosis'])
X = kyp.drop('Kyphosis', axis=1)
knn = KNeighborsRegressor(n_neighbors=3)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3,random_state=24)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
y_pred_prob = knn.predict(X_test)

print(r2_score(y_test, y_pred))

############################################# GSCV ############################################

kfold = KFold(n_splits = 5, shuffle = True, random_state = 24)
params = {'n_neighbors':[1,2,3,4,5,6,7,8]}

knn = KNeighborsRegressor()

gcv = GridSearchCV (knn, param_grid = params, cv = kfold, scoring = 'r2')
gcv.fit(X, y)

print(gcv.best_params_)
print(gcv.best_score_)


