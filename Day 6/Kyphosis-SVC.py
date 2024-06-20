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

kyp = pd.read_csv("D:\March 2024\PML\Day 6\Kyphosis.csv")
le = LabelEncoder()

y = le.fit_transform(kyp['Kyphosis'])
X = kyp.drop('Kyphosis', axis=1)

svm = SVC(C=1.0, degree=3, kernel='linear', probability = True, random_state=24)


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, random_state=24, stratify=y)

svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print("Accuracy Score: ",accuracy_score(y_test, y_pred))
print("R2 Score: ",r2_score(y_test, y_pred))
y_pred_prob = svm.predict_proba(X_test)
print("Log Loss of LDA: ",log_loss(y_test, y_pred_prob))
# Output: This 'SVC' has no attribute 'predict_proba' without probability = True
# Output: 'predict_proba' with only probability = True gives multiple log loss

##########################################################################################################

params1 = {'C': np.linspace(0.001, 5, 10)}
kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 24)
gsv = GridSearchCV(svm, param_grid=params1, cv=kfold, scoring='neg_log_loss')
gsv.fit(X,y)

print(gsv.best_params_)
print(gsv.best_score_)

params2 = {'C': [0.1, 1, 0.5, 2, 3]}
kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 24)
gsv = GridSearchCV(svm, param_grid=params2, cv=kfold)
gsv.fit(X,y)
pd_cv = pd.DataFrame(gsv.cv_results_)
print(gsv.best_params_)
print(gsv.best_score_)

########################################### Scaling #########################################################

stdscaler = StandardScaler()
minmax = MinMaxScaler()

svm = SVC(kernel='linear', probability = True, random_state=24)

pipe = Pipeline([("SCL", None),("SVM", svm)])
params3 = {'SCL': [stdscaler, minmax, None], 
           'SVM__C':np.linspace(0.001, 5, 20),
           'SVM__degree':[2,3],
           'SVM__coef0':np.linspace(0, 3, 5),
           'SVM__gamma': np.linspace(0.001, 5,5)}
# In params: the parameters should match the given/preloaded parameters. 
# Make sure the syntax is correct. If not, it gives warnings and incorrect results.

kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 24)

gsv = GridSearchCV (pipe, param_grid = params3, cv = kfold, scoring = 'neg_log_loss') #, verbose=2 shows calculation
gsv.fit(X, y)

pd_cv = pd.DataFrame(gsv.cv_results_)
print(gsv.best_params_)
print(gsv.best_score_)
