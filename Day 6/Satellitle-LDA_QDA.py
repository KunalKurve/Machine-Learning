import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

satellite =  pd.read_csv("Satellite.csv", sep=';')

le = LabelEncoder()

y = le.fit_transform(satellite['classes'])
X = satellite.drop('classes', axis = 1)

kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 24)

lda = LinearDiscriminantAnalysis()
qda = QuadraticDiscriminantAnalysis()

scores_lda = cross_val_score(lda, X, y, cv = kfold, scoring='neg_log_loss')
print("Cross Value Score of LDA: ",scores_lda.mean())

scores_qda = cross_val_score(qda, X, y, cv = kfold, scoring='neg_log_loss')
print("Cross Value Score of QDA: ",scores_qda.mean())

'''
Classification: Log Loss
Regression: R2 score, mean sqared error

Either use train test split or use kfold
Stratified used for Classification

'''