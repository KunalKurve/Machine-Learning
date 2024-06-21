import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import  log_loss, accuracy_score
from warnings import filterwarnings

filterwarnings('ignore')

kyp = pd.read_csv("Kyphosis.csv")

le = LabelEncoder()

X = kyp.drop('Kyphosis', axis = 1)
y = le.fit_transform(kyp['Kyphosis'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 24, stratify = y)

######################################################################################################################

lr = LogisticRegression()

bgg = BaggingClassifier(lr, n_estimators = 25, n_jobs = -1)

bgg.fit(X_train, y_train)
y_pred = bgg.predict(X_test)
y_pred_prob = bgg.predict_proba(X_test)

print("With Logistic Regression")
print("Accuracy Score:",accuracy_score(y_test, y_pred))
print("Log Loss:",log_loss(y_test, y_pred_prob))

######################################################################################################################

dtr = DecisionTreeClassifier(random_state=24)

bgg = BaggingClassifier(dtr, n_estimators = 25, n_jobs = -1)

bgg.fit(X_train, y_train)
y_pred = bgg.predict(X_test)
y_pred_prob = bgg.predict_proba(X_test)

print("\nWith Decision Tree")
print("Accuracy Score:",accuracy_score(y_test, y_pred))
print("Log Loss:",log_loss(y_test, y_pred_prob))

######################################################################################################################

# bgg = BaggingClassifier(dtr, n_estimators = 25, n_jobs = -1)

# kfold = KFold(n_splits=5, random_state=24, shuffle=True)

# params = {'Ridge__alpha': np.linspace(0.001, 3, 5),
#           'Lasso__alpha': np.linspace(0.001, 3, 5),
#           'TREE__max_depth': [None, 3, 4, 5],
#           'TREE__min_samples_leaf': [2, 5, 10],
#           'TREE__min_samples_split': [1, 5, 10]}

gcv = GridSearchCV(voting, param_grid = params, cv = kfold, scoring = 'r2', n_jobs = -1)
# Verbose is used to see one progress at a time
# n_jobs: Multi processing. None: default value. -1: All cores of CPU gets activated

gcv.fit(X,y)
print("\nUsing GSCV:")
print("R2 Score:", gcv.best_score_)
print("Best Parameters:\n", gcv.best_params_)
