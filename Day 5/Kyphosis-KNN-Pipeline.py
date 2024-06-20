import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

kyp = pd.read_csv('Kyphosis.csv')
le = LabelEncoder()

y = le.fit_transform(kyp['Kyphosis'])
X = kyp.drop('Kyphosis', axis=1)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=24)
knn = KNeighborsClassifier(n_neighbors=5)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3,random_state=24, stratify = y)

###############################################################################

std_scl = StandardScaler()

pipe_std = Pipeline([('SCL', std_scl) , ('KNN', knn)])

pipe_std.fit(X_train, y_train)
y_pred = pipe_std.predict(X_test)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
y_pred_prob = knn.predict_proba(X_test)

print("With Standard Scaling")
print("Accuracy Score with Scaling: ", accuracy_score(y_test, y_pred))
print("ROC AUC Score with Scaling: ", roc_auc_score(y_test, y_pred_prob[:,1]))
print("Log Loss with Scaling: ", log_loss(y_test, y_pred_prob))

###############################################################################

scl_mm = MinMaxScaler()
knn = KNeighborsClassifier()

pipe = Pipeline([('SCL',None),('knn',knn)])

params = {'KNN__n_neighbors': [1,2,3,4,5,6,7,8,9,10],
          'SCL': [std_scl, scl_mm, None]}

gcv = GridSearchCV (pipe, param_grid = params, cv = kfold, scoring = 'r2')
gcv.fit(X, y)

print(gcv.best_params_)
print(gcv.best_score_)