import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
glass =  pd.read_csv("Glass.csv")
le = LabelEncoder()

y = le.fit_transform(glass['Type'])
X = glass.drop('Type', axis = 1)

params = {'min_samples_split': np.arange(2,35,5),
          'min_samples_leaf': np.arange(1,35,5),
          'max_depth': [None, 4,3,2]}

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=24, stratify=y)

kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 24)
dtc = DecisionTreeClassifier(random_state=24)
gsv = GridSearchCV(dtc, param_grid=params, cv=kfold, scoring='neg_log_loss')

gsv.fit(X,y)

print(gsv.best_params_)
print(gsv.best_score_)

best_tree = gsv.best_estimator_

dtc.fit(X_train, y_train)

plt.figure(figsize=(25,20))
plot_tree(best_tree, feature_names = list(X.columns), class_names = list(le.classes_), filled = True, fontsize = 25)
# plot_tree(dtc, feature_names=list(X.columns), class_names=['0','1'], filled=True, fontsize=18)Z
# Output: IndexError: list index out of range
plt.title("Best Tree")
plt.show()

print(best_tree.feature_importances_)

df_imp = pd.DataFrame({'Features': list(X.columns),
                       'Importance': best_tree.feature_importances_})

plt.bar(df_imp['Features'], df_imp['Importance'])
plt.title("Feature Importance")
plt.show()


m_left, m_right = 183, 31
g_left, g_right = 0.679, 0.287
m = 214

ba_split = (m_left/m)*g_left + (m_right/m)*g_right
ba_reduction = 0.737* ba_split

m_left, m_right = 113, 70
g_left, g_right = 0.6, 0.584
m = 183

al_split = (m_left/m)*g_left + (m_right/m)*g_right