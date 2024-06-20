import pandas as pd
from sklearn.naive_bayes import GaussianNB
import os

os.chdir(r"D:\March 2024\PML\playground-series-s3e12")
train = pd.read_csv("train.csv", index_col=0)
print(train.isnull().sum().sum())
test = pd.read_csv("test.csv")
print(test.isnull().sum().sum())


X_train = train.drop('target', axis = 1)
y_train = train['target']

X_test = test.drop('id', axis = 1)
# y_test = test['id']


gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_prob = gnb.predict_proba(X_test)[:,1]

submit = pd.DataFrame({'id':test['id'],
                       'target': y_pred_prob})
print(submit)
submit.to_csv("submit-gnb.csv", index=False)