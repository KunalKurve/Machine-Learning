import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from warnings import filterwarnings

filterwarnings('ignore')

kyp = pd.read_csv("Kyphosis.csv")
le = LabelEncoder()

y = le.fit_transform(kyp['Kyphosis'])
X = kyp.drop('Kyphosis', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=24, stratify=y)

lr = LogisticRegression()
svm = SVC(kernel='linear', probability= True, random_state= 24)
dtc = DecisionTreeClassifier(random_state=24)

rf = RandomForestClassifier(random_state=24)

stack = StackingClassifier([('LR',lr), ('SVM', svm), ('TREE',dtc)], final_estimator = rf)

stack.fit(X_train, y_train)

print("For StackingClassifier without PassThrough ")
y_pred = stack.predict(X_test)
y_pred_proba = stack.predict_proba(X_test)
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Log Loss:", log_loss(y_test, y_pred_proba))

#############################################################################################

stack = StackingClassifier([('LR',lr), ('SVM', svm), ('TREE',dtc)], final_estimator = rf, passthrough= True)

stack.fit(X_train, y_train)

print("\nFor StackingClassifier with PassThrough ")
y_pred = stack.predict(X_test)
y_pred_proba = stack.predict_proba(X_test)
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Log Loss:", log_loss(y_test, y_pred_proba))

#############################################################################################

kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 24)

params = {'LR__C': np.linspace(0.01, 3, 5),
          'SVM__C': np.linspace(0.01, 3, 5),
          'TREE__max_depth': [None, 2,3,4],
          'final_estimator__max_features': [2,3],
          'passthrough': [False, True]}

gcv = GridSearchCV(stack, param_grid=params, cv=kfold, scoring='neg_log_loss', n_jobs= -1)

gcv.fit(X,y)

print(gcv.best_params_)
print("Neg Log Loss Score:",gcv.best_score_)