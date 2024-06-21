import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.preprocessing import LabelEncoder

hr =  pd.read_csv("HR_comma_sep.csv")
le = LabelEncoder()

hr_data = pd.get_dummies(hr, drop_first=True)

y = hr_data['left']
X = hr_data.drop('left', axis = 1)

params = {'min_samples_split': np.arange(2,35,5),
          'min_samples_leaf': np.arange(1,35,5),
          'max_depth': [None, 4,3,2]}

# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=24, stratify=y)

kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 24)
dtc = DecisionTreeClassifier(random_state=24)
gsv = GridSearchCV(dtc, param_grid=params, cv=kfold, scoring='neg_log_loss')

gsv.fit(X,y)

print(gsv.best_params_)
print(gsv.best_score_)

best_tree = gsv.best_estimator_

# dtc.fit(X_train, y_train)

plt.figure(figsize=(25,20))
plot_tree(best_tree, feature_names = list(X.columns), class_names = ['Left', 'Not Left'], filled = True, fontsize = 10)
plt.title("Best Tree")
plt.show()


print(best_tree.feature_importances_)

df_imp = pd.DataFrame({'Features': list(X.columns),
                       'Importance': best_tree.feature_importances_})

plt.barh(df_imp['Features'], df_imp['Importance'])
plt.title("Feature Importance")
plt.show()