import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

os.chdir("D:\March 2024\PML\Day 7\Multi-Class Prediction of Cirrhosis Outcomes")

train = pd.read_csv("train.csv", index_col=0)
print(train.isnull().sum().sum())
test = pd.read_csv("test.csv", index_col=0)
print(test.isnull().sum().sum())

train_dum = pd.get_dummies(train)
test_dum = pd.get_dummies(test, drop_first=True)

le = LabelEncoder()

X = train.drop(['Status'],axis=1)
y = le.fit_transform(train['Status'])

X_dum = pd.get_dummies(X, drop_first=True)

params = {'min_samples_split': np.arange(2,35,5),
          'min_samples_leaf': np.arange(1,35,5),
          'max_depth': [None, 4,3,2],
          'max_features': [3,4,5,6,7]}

kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 24)

rf = RandomForestClassifier(random_state=24)

gcv = GridSearchCV(rf, param_grid = params, cv = kfold, scoring = 'neg_log_loss')

gcv.fit(X_dum,y)

print(gcv.best_score_)
print(gcv.best_params_)

best_tree = gcv.best_estimator_

plt.figure(figsize=(15,10))                  #dtc tree with max_depth=None, and direct fitting 
                                             #without using gcv search
plot_tree(best_tree, feature_names = list(X_dum.columns),
               class_names = ['Status_C', 'Status_CL', 'Status_D'],
               filled = True,fontsize = 9)
plt.show()

print(best_tree.feature_importances_)

df_imp1 = pd.DataFrame({'Feature': list(X_dum.columns), 'Importance':best_tree.feature_importances_ })

df_imp1.plot(kind='barh',x='Feature')
plt.show()

y_pred_prob=gcv.predict_proba(test_dum)

submit=pd.DataFrame({'id':test_dum.index, 
                     'Status_C': y_pred_prob[:,0] , 
                     'Status_CL': y_pred_prob[:,1], 
                     'Status_D': y_pred_prob[:,2]})

print(submit)

# submit.to_csv("CirrosisSubmit-1.csv", index=False)
