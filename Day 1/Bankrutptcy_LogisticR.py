# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 18:14:13 2024

@author: Administrator
"""

import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

bank =  pd.read_csv("Bankruptcy.csv")
# dum_df = pd.get_dummies(bank, drop_first=True)


# le = LabelEncoder() # No need because the data is already encoded



X = bank.drop(['D','NO'], axis=1)
y = bank['D']


# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.3, 
                                                    random_state=2021,
                                                    stratify=y)

lr = LogisticRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
y_pred_prob = lr.predict_proba(X_test)

############## Model Evaluation ##############
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy Score: ",accuracy_score(y_test, y_pred))

# Compute predicted probabilities: y_pred_prob
y_probs = lr.predict_proba(X_test)
y_pred_prob = y_probs[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
import matplotlib.pyplot as plt
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.show()

print("roc_auc_score", roc_auc_score(y_test, y_pred_prob))
print("Log Loss: ", log_loss(y_test, y_pred_prob))