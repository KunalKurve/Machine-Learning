import pandas as pd
import numpy as np 
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


conc = pd.read_csv(r"D:\\March 2024\\PML\\Day 5\\Concrete_Data.csv")
X = conc.drop('Strength', axis = 1)
y = conc['Strength']

kfold = KFold(n_splits = 5, shuffle = True, random_state = 24)

################################ Ridge ########################################

rd = Ridge()
poly = PolynomialFeatures()
pipe = Pipeline([('POLY', poly),('RIDGE', rd)])
print(pipe.get_params())
params = {'POLY':[1, 2, 3],
          'alpha':np.linspace(0.001, 100,40)}
gcv_rd = GridSearchCV(pipe, param_grid = params, cv = kfold)
gcv_rd.fit(X, y)

print("Best Degree (Ridge):", gcv_rd.best_score_)
print("Best Score (Ridge):", gcv_rd.best_params_)

############################## ElasticNet #####################################

el = ElasticNet()
poly = PolynomialFeatures()
pipe = Pipeline([('POLY', poly),('LR', el)])
print(pipe.get_params())
params = {'POLY_degree':[1, 2, 3],
          'LR_alpha': np.linspace(0.001, 5, 10),
          'LR_L1_ratio': np.linspace(0, 1, 5)}
gcv_el = GridSearchCV(pipe, param_grid = params, cv = kfold)
gcv_el.fit(X, y)

print("Best Degree (ElasticNet):", gcv_el.best_score_)
print("Best Score (ElasticNet):", gcv_el.best_params_)