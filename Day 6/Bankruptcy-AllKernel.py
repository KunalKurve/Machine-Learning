import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
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
# params1 = {'C': np.linspace(0.001, 5, 10)}

pipe = Pipeline([("SCL", None),("SVM", svm)])
params3 = {'SCL': [stdscaler, minmax, None], 
            'SVM__C':[1,2,3],
            'SVM__degree':[2,3],
            'SVM__coef0':np.linspace(0, 3, 5),
            'SVM__gamma': np.linspace(0.001, 5,5)}

gsv = GridSearchCV (pipe, param_grid = params3, cv = kfold, scoring = 'neg_log_loss') #, verbose=2 shows calculation
# gsv = GridSearchCV (svm, param_grid = params1, cv = kfold, scoring = 'neg_log_loss')
gsv.fit(X, y)

pd_cv = pd.DataFrame(gsv.cv_results_)
print(gsv.best_params_)
print(gsv.best_score_)

############################# RADIAL #############################################

print("Radial kernel Results: ")

svm = SVC(kernel='rbf', probability = True, random_state=24)

pipe = Pipeline([("SCL", None),("SVM", svm)])
params3 = {'SCL': [stdscaler, minmax, None], 
            'SVM__C':[1,2,3],
            'SVM__degree':[2,3],
            'SVM__coef0':np.linspace(0, 3, 5),
            'SVM__gamma': np.linspace(0.001, 5,5)}

gsv = GridSearchCV (pipe, param_grid = params3, cv = kfold, scoring = 'neg_log_loss') #, verbose=2 shows calculation
# gsv = GridSearchCV (svm, param_grid = params1, cv = kfold, scoring = 'neg_log_loss')
gsv.fit(X, y)

pd_cv = pd.DataFrame(gsv.cv_results_)
print(gsv.best_params_)
print(gsv.best_score_)

############################# Linear #############################################

print("Linear kernel Results: ")

svm = SVC(kernel='linear', probability = True, random_state=24)

pipe = Pipeline([("SCL", None),("SVM", svm)])
params3 = {'SCL': [stdscaler, minmax, None], 
            'SVM__C':[1,2,3],
            'SVM__degree':[2,3],
            'SVM__coef0':np.linspace(0, 3, 5),
            'SVM__gamma': np.linspace(0.001, 5,5)}

gsv = GridSearchCV (pipe, param_grid = params3, cv = kfold, scoring = 'neg_log_loss') #, verbose=2 shows calculation
# gsv = GridSearchCV (svm, param_grid = params1, cv = kfold, scoring = 'neg_log_loss')
gsv.fit(X, y)

pd_cv = pd.DataFrame(gsv.cv_results_)
print(gsv.best_params_)
print(gsv.best_score_)

############################# Sigmoid #############################################

print("Sigmoid kernel Results: ")

svm = SVC(kernel='sigmoid', probability = True, random_state=24)

pipe = Pipeline([("SCL", None),("SVM", svm)])
params3 = {'SCL': [stdscaler, minmax, None], 
            'SVM__C':[1,2,3],
            'SVM__degree':[2,3],
            'SVM__coef0':np.linspace(0, 3, 5),
            'SVM__gamma': np.linspace(0.001, 5,5)}

gsv = GridSearchCV (pipe, param_grid = params3, cv = kfold, scoring = 'neg_log_loss') #, verbose=2 shows calculation

# gsv = GridSearchCV (svm, param_grid = params1, cv = kfold, scoring = 'neg_log_loss')
gsv.fit(X, y)

pd_cv = pd.DataFrame(gsv.cv_results_)
print(gsv.best_params_)
print(gsv.best_score_)

############################# precomputed #############################################

print("precomputed kernel Results: ")

svm = SVC(kernel='precomputed', probability = True, random_state=24)

pipe = Pipeline([("SCL", None),("SVM", svm)])
params3 = {'SCL': [stdscaler, minmax, None], 
            'SVM__C':[1,2,3],
            'SVM__degree':[2,3],
            'SVM__coef0':np.linspace(0, 3, 5),
            'SVM__gamma': np.linspace(0.001, 5,5)}

gsv = GridSearchCV (pipe, param_grid = params3, cv = kfold, scoring = 'neg_log_loss') #, verbose=2 shows calculation

#gsv = GridSearchCV (svm, param_grid = params3, cv = kfold, scoring = 'neg_log_loss')
gsv.fit(X, y)

pd_cv = pd.DataFrame(gsv.cv_results_)
print(gsv.best_params_)
print(gsv.best_score_)

############################# OUTPUT #############################################
'''
Radial kernel Results: 
{'SCL': StandardScaler(), 'SVM__C': 3, 'SVM__coef0': 0.0, 'SVM__degree': 2, 'SVM__gamma': 0.001}
-0.48923622940287476

Linear kernel Results: 
{'SCL': None, 'SVM__C': 1, 'SVM__coef0': 0.0, 'SVM__degree': 2, 'SVM__gamma': 0.001}
-0.4786507728333083

Sigmoid kernel Results: 
{'SCL': StandardScaler(), 'SVM__C': 1, 'SVM__coef0': 0.0, 'SVM__degree': 2, 'SVM__gamma': 5.0}
-0.5105153385040555

precomputed kernel Results:
ValueError: 
All the 2250 fits failed.
It is very likely that your model is misconfigured.
You can try to debug the error by setting error_score='raise'

'''