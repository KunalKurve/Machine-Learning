import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
glass =  pd.read_csv("Glass.csv")
le = LabelEncoder()

y = le.fit_transform(glass['Type'])
X = glass.drop('Type', axis = 1)

print(le.classes_)

stdscaler = StandardScaler()
minmax = MinMaxScaler()
kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 24)

params3 = {'SCL': [stdscaler, minmax, None], 
            'SVM__C':[1,2,3],
            'SVM__degree':[2,3],
            'SVM__coef0':np.linspace(0, 3, 5),
            'SVM__gamma': np.linspace(0.001, 5,5)}


############################# OVO RADIAL #############################################

print("Radial kernel Results: ")

svm = SVC(kernel='rbf', probability = True, random_state=24, decision_function_shape= 'ovo')

pipe = Pipeline([("SCL", None),("SVM", svm)])

gsv = GridSearchCV (pipe, param_grid = params3, cv = kfold, scoring = 'neg_log_loss') #, verbose=2 shows calculation

gsv.fit(X, y)

pd_cv = pd.DataFrame(gsv.cv_results_)
print(gsv.best_params_)
print(gsv.best_score_)

############################# OVO Linear #############################################

print("Linear kernel Results: ")

svm = SVC(kernel='linear', probability = True, random_state=24, decision_function_shape= 'ovo')

pipe = Pipeline([("SCL", None),("SVM", svm)])

gsv = GridSearchCV (pipe, param_grid = params3, cv = kfold, scoring = 'neg_log_loss') #, verbose=2 shows calculation

gsv.fit(X, y)

pd_cv = pd.DataFrame(gsv.cv_results_)
print(gsv.best_params_)
print(gsv.best_score_)

############################# OVO Sigmoid #############################################

print("Sigmoid kernel Results: ")

svm = SVC(kernel='sigmoid', probability = True, random_state=24, decision_function_shape= 'ovo')

pipe = Pipeline([("SCL", None),("SVM", svm)])

gsv = GridSearchCV (pipe, param_grid = params3, cv = kfold, scoring = 'neg_log_loss') #, verbose=2 shows calculation

# gsv = GridSearchCV (svm, param_grid = params1, cv = kfold, scoring = 'neg_log_loss')
gsv.fit(X, y)

pd_cv = pd.DataFrame(gsv.cv_results_)
print(gsv.best_params_)
print(gsv.best_score_)

############################# OVR RADIAL #############################################

print("Radial kernel Results: ")

svm = SVC(kernel='rbf', probability = True, random_state=24, decision_function_shape= 'ovr')

pipe = Pipeline([("SCL", None),("SVM", svm)])

gsv = GridSearchCV (pipe, param_grid = params3, cv = kfold, scoring = 'neg_log_loss') #, verbose=2 shows calculation

gsv.fit(X, y)

pd_cv = pd.DataFrame(gsv.cv_results_)
print(gsv.best_params_)
print(gsv.best_score_)

############################# OVR Linear #############################################

print("Linear kernel Results: ")

svm = SVC(kernel='linear', probability = True, random_state=24, decision_function_shape= 'ovr')

pipe = Pipeline([("SCL", None),("SVM", svm)])

gsv = GridSearchCV (pipe, param_grid = params3, cv = kfold, scoring = 'neg_log_loss') #, verbose=2 shows calculation

gsv.fit(X, y)

pd_cv = pd.DataFrame(gsv.cv_results_)
print(gsv.best_params_)
print(gsv.best_score_)

############################# OVR Sigmoid #############################################

print("Sigmoid kernel Results: ")

svm = SVC(kernel='sigmoid', probability = True, random_state=24, decision_function_shape= 'ovr')

pipe = Pipeline([("SCL", None),("SVM", svm)])

gsv = GridSearchCV (pipe, param_grid = params3, cv = kfold, scoring = 'neg_log_loss') #, verbose=2 shows calculation

# gsv = GridSearchCV (svm, param_grid = params1, cv = kfold, scoring = 'neg_log_loss')
gsv.fit(X, y)

pd_cv = pd.DataFrame(gsv.cv_results_)
print(gsv.best_params_)
print(gsv.best_score_)

############################# OVO RESULT #############################################

'''
Radial kernel Results: 
{'SCL': None, 'SVM__C': 2, 'SVM__coef0': 0.0, 'SVM__degree': 2, 'SVM__gamma': 1.2507499999999998}
-0.7960854538525931
Linear kernel Results: 
{'SCL': None, 'SVM__C': 1, 'SVM__coef0': 0.0, 'SVM__degree': 2, 'SVM__gamma': 0.001}
-0.9335800130111664
Sigmoid kernel Results: 
{'SCL': StandardScaler(), 'SVM__C': 2, 'SVM__coef0': 0.0, 'SVM__degree': 2, 'SVM__gamma': 0.001}
-1.037377548548131
'''

############################# OVR RESULT #############################################

'''
Radial kernel Results: 
{'SCL': None, 'SVM__C': 2, 'SVM__coef0': 0.0, 'SVM__degree': 2, 'SVM__gamma': 1.2507499999999998}
-0.7960854538525931
Linear kernel Results: 
{'SCL': None, 'SVM__C': 1, 'SVM__coef0': 0.0, 'SVM__degree': 2, 'SVM__gamma': 0.001}
-0.9335800130111664
Sigmoid kernel Results: 
{'SCL': StandardScaler(), 'SVM__C': 2, 'SVM__coef0': 0.0, 'SVM__degree': 2, 'SVM__gamma': 0.001}
-1.037377548548131
'''