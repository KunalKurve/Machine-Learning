import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

bank =  pd.read_csv("Bankruptcy.csv")

y = bank['D']
X = bank.drop(['D','NO'], axis = 1)


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, random_state=24, stratify=y)
kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 24)

lda = LinearDiscriminantAnalysis()
qda = QuadraticDiscriminantAnalysis()

lda.fit(X_train, y_train)
y_pred = lda.predict(X_test)
print("Accuracy Score of LDA: ",accuracy_score(y_test, y_pred))
y_pred_prob = lda.predict_proba(X_test)
print("Log Loss of LDA: ",log_loss(y_test, y_pred_prob))
scores_lda = cross_val_score(lda, X, y, cv = kfold, scoring='neg_log_loss')
print("Cross Value Score of LDA: ",scores_lda.mean())

qda.fit(X_train, y_train)
# Output: warnings.warn("Variables are collinear")
y_pred = qda.predict(X_test)
print("Accuracy Score of QDA: ",accuracy_score(y_test, y_pred))
y_pred_prob = qda.predict_proba(X_test)
print("Log Loss of QDA: ",log_loss(y_test, y_pred_prob))
# Output: ValueError: Input contains NaN.
scores_qda = cross_val_score(qda, X, y, cv = kfold, scoring='neg_log_loss')
print("Cross Value Score of QDA: ",scores_qda.mean())

