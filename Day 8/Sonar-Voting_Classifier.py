import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

sonar =  pd.read_csv("Sonar.csv")

le = LabelEncoder()

y = le.fit_transform(sonar['Class'])
X = sonar.drop('Class', axis=1)
print(le.classes_)

std_slr = StandardScaler()
lr = LogisticRegression()
lda = LinearDiscriminantAnalysis()
dtc = DecisionTreeClassifier(random_state=24)

svm_l = SVC(kernel = 'linear', probability= True, random_state= 24)
pipe_l = Pipeline([('SCL', std_slr),('SVM', svm_l)])

svm_r = SVC(kernel = 'rbf', probability= True, random_state= 24)
pipe_r = Pipeline([('SCL', std_slr),('SVM', svm_r)])

voting = VotingClassifier([('LR',lr), ('SVML', pipe_l),
                           ('SVM_R',pipe_r), ('LDA',lda),
                           ('TREE', dtc)], voting = 'soft')

kfold = StratifiedKFold(n_splits=5, random_state=24, shuffle=True)

params = {'SVML__SVM__C': np.linspace(0.001, 3, 5),
          'SVM_R__SVM__gamma': np.linspace(0.001, 3, 5),
          'SVM_R__SVM__C': np.linspace(0.001, 3, 5),
          'LR__C': np.linspace(0.001, 3, 5),
          'TREE__max_depth': [None, 3,2]}

gcv = GridSearchCV(voting, param_grid = params, cv = kfold, scoring = 'neg_log_loss', n_jobs = -1)
# Verbose is used to see one progress at a time
# n_jobs: Multi processing. None: default value. -1: All course of CPU gets activated

gcv.fit(X,y)
print(gcv.best_score_)
print(gcv.best_params_)