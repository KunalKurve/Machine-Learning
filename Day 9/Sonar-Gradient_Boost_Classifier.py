import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier

sonar =  pd.read_csv("Sonar.csv")

le = LabelEncoder()

y = le.fit_transform(sonar['Class'])
X = sonar.drop('Class', axis=1)

gbc = GradientBoostingClassifier(random_state=24)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=24, stratify=y)

gbc.fit(X_train, y_train)

y_pred = gbc.predict(X_test)
y_pred_proba = gbc.predict_proba(X_test)

print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Log Loss:", log_loss(y_test, y_pred_proba))

kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 24)

params = {'learning_rate': np.linspace(0.001, 0.9, 10),
          'max_depth': [None, 4,3,2],
          'n_estimators': [25, 50, 100]}

gsv = GridSearchCV (gbc, param_grid=params, cv=kfold, scoring='neg_log_loss')

gsv.fit(X,y)

pd_cv = pd.DataFrame(gsv.cv_results_)

print(gsv.best_params_)
print("Log Loss:", gsv.best_score_)