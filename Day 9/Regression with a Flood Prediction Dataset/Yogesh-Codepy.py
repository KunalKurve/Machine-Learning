from sklearn.ensemble import BaggingRegressor
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression,ElasticNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split,StratifiedKFold,KFold
from sklearn.metrics import accuracy_score,log_loss
from sklearn.tree import DecisionTreeClassifier
from warnings import filterwarnings

filterwarnings('ignore')

train = pd.read_csv(r"C:\Users\Administrator.DAI-PC2\Desktop\MachineLearning\Day6\train.csv", index_col=0)
print(train.isnull().sum().sum())
test = pd.read_csv(r"C:\Users\Administrator.DAI-PC2\Desktop\MachineLearning\Day6\test.csv")
print(test.isnull().sum().sum())

X_train = train.drop('FloodProbability', axis=1)
y_train = train['FloodProbability']
X_test = test.drop('id', axis=1)
 

lr = LinearRegression()
bagg = BaggingRegressor(lr,n_estimators=25,random_state=24,n_jobs=-1)


bagg.fit(X_train,y_train)

y_pred = bagg.predict(X_test)


submit = pd.DataFrame({'id':test['id'],
                       'FloodProbability': y_pred})


submit.to_csv("sbt_lr_bagging.csv",index=False)


################################################ElasticNet#####################

el = ElasticNet()
kfold = KFold(n_splits=5,shuffle=True,random_state=24)
bag = BaggingRegressor(el,n_estimators=25,random_state=24)
print(bag.get_params())
params={'estimator__alpha':np.linspace(0.001,5,5),
        'estimator__l1_ratio':np.linspace(0,1,8)}
gsv = GridSearchCV(bag, param_grid=params, cv=kfold, n_jobs=-1)
gsv.fit(X_train,y_train)
best_model = gsv.best_estimator_
y_pred = best_model.predict(X_test)
submit = pd.DataFrame({'id':test['id'],
                       'FloodProbability': y_pred})


submit.to_csv("sbt_elasticnet_bagging.csv",index=False)