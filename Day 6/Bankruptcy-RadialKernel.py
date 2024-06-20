import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

bank = pd.read_csv("Bankruptcy.csv")
X = bank.drop(['D','NO'], axis=1)
y = bank['D']


stdscaler = StandardScaler()
minmax = MinMaxScaler()
kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 24)

############################# POLY #############################################

print("Poly kernel Results: ")

svm = SVC(kernel='poly', probability = True, random_state=24)

pipe = Pipeline([("SCL", None),("SVM", svm)])
params3 = {'SCL': [stdscaler, minmax, None], 
           'SVM__C':np.linspace(0.001, 5, 10),
           'SVM__degree':[2,3],
           'SVM__coef0':np.linspace(0, 3, 5),
           'SVM__gamma': np.linspace(0.001, 5,5)}

gsv = GridSearchCV (pipe, param_grid = params3, cv = kfold, scoring = 'neg_log_loss') #, verbose=2 shows calculation
gsv.fit(X, y)

pd_cv = pd.DataFrame(gsv.cv_results_)
print(gsv.best_params_)
print(gsv.best_score_)

############################# RADIAL #############################################

print("Radial kernel Results: ")

svm = SVC(kernel='rbf', probability = True, random_state=24)

pipe = Pipeline([("SCL", None),("SVM", svm)])
params3 = {'SCL': [stdscaler, minmax, None], 
           'SVM__C':np.linspace(0.001, 5, 10),
           'SVM__degree':[2,3],
           'SVM__coef0':np.linspace(0, 3, 5),
           'SVM__gamma': np.linspace(0.001, 5,5)}

gsv = GridSearchCV (pipe, param_grid = params3, cv = kfold, scoring = 'neg_log_loss') #, verbose=2 shows calculation
gsv.fit(X, y)

pd_cv = pd.DataFrame(gsv.cv_results_)
print(gsv.best_params_)
print(gsv.best_score_)

############################# Linear #############################################

print("Linear kernel Results: ")

svm = SVC(kernel='linear', probability = True, random_state=24)

pipe = Pipeline([("SCL", None),("SVM", svm)])
params3 = {'SCL': [stdscaler, minmax, None], 
           'SVM__C':np.linspace(0.001, 5, 10),
           'SVM__degree':[2,3],
           'SVM__coef0':np.linspace(0, 3, 5),
           'SVM__gamma': np.linspace(0.001, 5,5)}

gsv = GridSearchCV (pipe, param_grid = params3, cv = kfold, scoring = 'neg_log_loss') #, verbose=2 shows calculation
gsv.fit(X, y)

pd_cv = pd.DataFrame(gsv.cv_results_)
print(gsv.best_params_)
print(gsv.best_score_)

############################# Sigmoid #############################################

print("Sigmoid kernel Results: ")

svm = SVC(kernel='sigmoid', probability = True, random_state=24)

pipe = Pipeline([("SCL", None),("SVM", svm)])
params3 = {'SCL': [stdscaler, minmax, None], 
           'SVM__C':np.linspace(0.001, 5, 10),
           'SVM__degree':[2,3],
           'SVM__coef0':np.linspace(0, 3, 5),
           'SVM__gamma': np.linspace(0.001, 5,5)}

gsv = GridSearchCV (pipe, param_grid = params3, cv = kfold, scoring = 'neg_log_loss') #, verbose=2 shows calculation
gsv.fit(X, y)

pd_cv = pd.DataFrame(gsv.cv_results_)
print(gsv.best_params_)
print(gsv.best_score_)

############################# precomputed #############################################

print("precomputed kernel Results: ")

svm = SVC(kernel='precomputed', probability = True, random_state=24)

pipe = Pipeline([("SCL", None),("SVM", svm)])
params3 = {'SCL': [stdscaler, minmax, None], 
           'SVM__C':np.linspace(0.001, 5, 10),
           'SVM__degree':[2,3],
           'SVM__coef0':np.linspace(0, 3, 5),
           'SVM__gamma': np.linspace(0.001, 5,5)}

gsv = GridSearchCV (pipe, param_grid = params3, cv = kfold, scoring = 'neg_log_loss') #, verbose=2 shows calculation
gsv.fit(X, y)

pd_cv = pd.DataFrame(gsv.cv_results_)
print(gsv.best_params_)
print(gsv.best_score_)
