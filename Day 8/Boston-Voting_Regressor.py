import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, r2_score

boston = pd.read_csv('Boston.csv')

X = boston.drop('medv', axis = 1)
y = boston['medv']

lr = LinearRegression()
std_slr = StandardScaler()
ridge = Ridge()
lasso = Lasso()
dtr = DecisionTreeRegressor(random_state=24)

##########################################################################################
 
voting = VotingRegressor([('LR',lr), ('Ridge', ridge), ('Lasso',lasso),('TREE',dtr)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=24)

lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
r2_lr = r2_score(y_test, y_pred)

ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)
r2_ridge = r2_score(y_test, y_pred)

lasso.fit(X_train, y_train)
y_pred = lasso.predict(X_test)
r2_lasso = r2_score(y_test, y_pred)

dtr.fit(X_train, y_train)
y_pred = dtr.predict(X_test)
r2_dtr = r2_score(y_test, y_pred)

voting.fit(X_train, y_train)
y_pred = voting.predict(X_test)
r2_voting = r2_score(y_test, y_pred)

print("R2 Scores for:")
print("Linear R: ", r2_lr)
print("Ridge: ", r2_ridge)
print("Lasso: ", r2_lasso)
print("Tree: ", r2_dtr)
print("Voting: ", r2_voting)

##########################################################################################

# weights = np.array([[r2_lr, r2_ridge, r2_lasso, r2_dtr]])

voting = VotingRegressor([('LR',lr), ('Ridge', ridge), ('Lasso',lasso),('TREE',dtr)], 
                         weights= [r2_lr, r2_ridge, r2_lasso, r2_dtr] )

voting.fit(X_train, y_train)
y_pred = voting.predict(X_test)
r2_voting = r2_score(y_test, y_pred)

print("Weighted Voting R2 Score: ", r2_voting)


