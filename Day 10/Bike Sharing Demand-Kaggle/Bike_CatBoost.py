import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from xgboost import XGBRFRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from warnings import filterwarnings
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import StackingRegressor
import os
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns

filterwarnings('ignore')

os.chdir(r"D:\March 2024\PML\Day 10\Bike Sharing Demand-Kaggle")
train = pd.read_csv("train.csv", index_col=0)
print(train.isnull().sum().sum())
test = pd.read_csv("test.csv")
print(test.isnull().sum().sum())

train.groupby('season')['count'].mean().plot(kind = 'bar')
plt.show()

# X_train = train.reset_index(inplace = True)
X_train = train.drop(['casual','registered','count'], axis = 1)
y_train = train['count']
X_test = test.drop('datetime', axis = 1)

cat = CatBoostRegressor(random_state = 24, logging_level = 'Silent')

cat.fit(X_train, y_train)

y_pred = cat.predict(X_test)
y_pred[y_pred < 0] = 0

submit = pd.DataFrame({'datetime': test['datetime'], 'count': y_pred})
submit.min()

submit.to_csv('Bike_CatBoost.csv', index = False)

registered = train['registered']
casual = train['casual']

cat.fit(X_train, registered)

y_pred_reg = cat.predict(X_test)

cat.fit(X_train, casual)

y_pred_cas = cat.predict(X_test)

final_pred = y_pred_reg + y_pred_cas
final_pred[final_pred < 0] = 0 
# Because we have some -ve values. To exclude them, use this condition
final_pred.min()
print(final_pred)

submit1 = pd.DataFrame({'datetime': test['datetime'], 'count': final_pred})

submit1.to_csv('Bike_CatBoost_Split.csv', index = False)

####### Using Date Time Module ###############

df_train = pd.read_csv("train.csv", parse_dates=['datetime'])
df_test = pd.read_csv("test.csv", parse_dates=['datetime'])

def date_features(df):
  df['year'] = df['datetime'].dt.year
  df['month'] = df['datetime'].dt.month
  df['day'] = df['datetime'].dt.day
  df['hour'] = df['datetime'].dt.hour
  df['weekday'] = df['datetime'].dt.weekday
  return df

train_df = date_features(df_train)
test_df = date_features(df_test)

print(train_df.head())
print(test_df.head())

X_train = train_df.drop(['datetime','casual','registered','count'], axis = 1)
y_train = train_df['count']
X_test = test_df.drop('datetime', axis = 1)

registered = train['registered']
casual = train['casual']

cat_reg = CatBoostRegressor(random_state = 24, logging_level = 'Silent', cat_features=['weather', 'season'])
cat_cas = CatBoostRegressor(random_state = 24, logging_level = 'Silent', cat_features=['weather', 'season'])

cat_reg.fit(X_train, registered)
cat_cas.fit(X_train, casual)

y_pred_reg = cat_reg.predict(X_test)
y_pred_cas = cat_cas.predict(X_test)

y_pred_reg[y_pred_reg < 0] = 0
y_pred_cas[y_pred_cas < 0] = 0

y_pred1 = y_pred_reg + y_pred_cas

submit2 = pd.DataFrame({'datetime': test_df['datetime'], 'count': y_pred1})

submit2.to_csv('Bike_CatBoost_Split_DateTime.csv', index = False)

sns.scatterplot(x = 'hour', y = 'registered', data = train_df)
plt.xlabel('Hour')
plt.ylabel('Registered')
plt.show()

sns.scatterplot(x = 'hour', y = 'casual', data = train_df)
plt.xlabel('Hour')
plt.ylabel('Casual')
plt.show()

g = sns.FacetGrid(train_df, col = 'weekday')
g.map(sns.scatterplot, 'hour', 'casual')
plt.show()

g = sns.FacetGrid(train_df, col = 'weekday')
g.map(sns.scatterplot, 'hour', 'registered')
plt.show()

