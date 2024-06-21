import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

kyp = pd.read_csv("Kyphosis.csv")
le = LabelEncoder()

y = le.fit_transform(kyp['Kyphosis'])
X = kyp.drop('Kyphosis', axis=1)

params = {'min_samples_split': [2,4,6,10,20],
          'min_samples_leaf': [1,5,10,15],
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
plot_tree(dtc, feature_names=list(X.columns), class_names=['0','1'], filled=True, fontsize=18)
plt.title("Best Tree")
plt.show()