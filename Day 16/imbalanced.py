import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN

df = pd.read_csv(r"C:\Training\Academy\Statistics (Python)\Cases\human-resources-analytics\HR_comma_sep.csv")
dum_df = pd.get_dummies(df,drop_first=True)
X = dum_df.drop('left',axis=1)
y = dum_df['left']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,
                                                 stratify=y,
                                                 random_state=2022)

lr = LogisticRegression()
################### w/o Balancing ######################
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(classification_report(y_test,y_pred))

################### Over-Sampling(Naive) ###############
ros = RandomOverSampler(random_state=24)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
lr.fit(X_resampled, y_resampled)
y_pred = lr.predict(X_test)
print(classification_report(y_test,y_pred))

################# Over-Sampling(SMOTE) #################

smote = SMOTE(random_state=2022)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
lr.fit(X_resampled, y_resampled)
y_pred = lr.predict(X_test)
print(classification_report(y_test,y_pred))

################# Over-Sampling(ADASYN) #################

adasyn = ADASYN(random_state=2021)
X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)
lr.fit(X_resampled, y_resampled)
y_pred = lr.predict(X_test)
print(classification_report(y_test,y_pred))

