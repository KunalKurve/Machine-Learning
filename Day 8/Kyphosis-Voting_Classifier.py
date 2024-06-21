# -*- coding: utf-8 -*-
"""
Voting Classifier: 
    Estimator: List of tuples
    
Hard Voting: y_pred = voting.predict(X_test) -> Take accuracy score
Soft Voting: y_pred_prob = voting.predict_proba(X_test)[:,1] -> Take accuracy score -> Gives error
To avoid this:
    In VotingClassifier, take a parameter as "voting = 'soft'"
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import log_loss

kyp = pd.read_csv(r"D:/March 2024/PML/Day 8/Kyphosis.csv")

le = LabelEncoder()

y = le.fit_transform(kyp['Kyphosis'])
X = kyp.drop('Kyphosis', axis=1)

std_slr = StandardScaler()

svm_l = SVC(kernel = 'linear', probability=True, random_state=24)
pipe_l = Pipeline([('SCL', std_slr),('SVM', svm_l)])

svm_r = SVC(kernel='rbf', probability = True, random_state=24)
pipe_r = Pipeline([('SCL', std_slr),('SVM', svm_r)])


lr = LogisticRegression()
lda = LinearDiscriminantAnalysis()
dtc = DecisionTreeClassifier(random_state=24)

voting = VotingClassifier([('LR',lr),('SVML',pipe_l),
                           ('SVM_R',pipe_r),('LDA',lda),
                           ('TREE', dtc)], voting='soft')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=24, stratify=y)

voting.fit(X_train, y_train)
y_pred = voting.predict(X_test)
y_pred_prob = voting.predict_proba(X_test)
print("Accuracy Score: ", accuracy_score(y_test, y_pred))
print("Log Loss: ", log_loss(y_test, y_pred_prob))

###################################################################################################

kfold = StratifiedKFold(n_splits=5, random_state=24, shuffle=True)

params = {'SVML__SVM__C': np.linspace(0.001, 3, 5),
          'SVM_R__SVM__gamma': np.linspace(0.001, 3, 5),
          'SVM_R__SVM__C': np.linspace(0.001, 3, 5),
          'LR__C': np.linspace(0.001, 3, 5),
          'TREE__max_depth': [None, 3,2]}

gcv = GridSearchCV(voting, param_grid = params, cv = kfold, scoring = 'neg_log_loss', n_jobs = -1)
# Verbose is used to see one progress at a time

gcv.fit(X,y)
print(gcv.best_score_)
print(gcv.best_params_)

################################################################################################

