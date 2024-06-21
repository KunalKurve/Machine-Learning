import pandas as pd
import numpy as np 
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from catboost import CatBoostRegressor
from sklearn.compose import make_column_selector
from warnings import filterwarnings
import matplotlib.pyplot as plt

filterwarnings('ignore')

house = pd.read_csv("Housing.csv")

y = house['price']
X = house.drop('price', axis=1)

cat = ['driveway', 'recroom', 'fullbase', 'gashw', 'prefarea', 'airco']

cbr = CatBoostRegressor(random_state=24, cat_features= cat)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=24)

cbr.fit(X_train, y_train)

y_pred = cbr.predict(X_test)

# CatBoostRegressor does not have predict_proba

print("R2 Score:", r2_score(y_test, y_pred))

####################################### GCV ########################################

kfold = KFold(n_splits = 5, shuffle = True, random_state = 24)

params = {'learning_rate': np.linspace(0.001, 0.9, 10),
          'max_depth': [None, 4,3,2],
          'n_estimators': [25, 50, 100]}

gsv = GridSearchCV (cbr, param_grid=params, cv=kfold, scoring='r2')

gsv.fit(X,y)

pd_cv = pd.DataFrame(gsv.cv_results_)

best_model = gsv.best_estimator_

print(gsv.best_params_)
print("R2 Score:",gsv.best_score_)

df_imp = pd.DataFrame({"Feature": list(X.columns), "Importances": best_model.feature_importances_})

plt.barh(df_imp)
plt.title("Feature Importances")
plt.show()