import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from warnings import filterwarnings

filterwarnings('ignore')

glass =  pd.read_csv("Glass.csv")
le = LabelEncoder()

y = le.fit_transform(glass['Type'])
X = glass.drop('Type', axis = 1)

scaler  = StandardScaler().set_output(transform='pandas')
prc_comp = PCA(n_components=11).set_output(transform='pandas')

############################# Logistic Regression ##############################

lr = LogisticRegression()

pipe = Pipeline([('SCL',scaler), ('PCA',prc_comp), ('LR',lr)]) 

params = {'PCA__n_components': [5,6,7,8,9],
          'LR__C': np.linspace(0.001, 3, 5),
          'LR__multi_class': ['ovr','multinomial']}

kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 24)

gscv = GridSearchCV(pipe, param_grid=params, cv=kfold, scoring='neg_log_loss')

gscv.fit(X,y)

print("\nUsing Logistic Regression")
print("Best Parameters:")
print(gscv.best_params_)
print("Best neg log loss:",gscv.best_score_)

################################# Gaussian NB ###################################

gnb = GaussianNB()

pipe = Pipeline([('SCL',scaler), ('PCA',prc_comp), ('GNB',gnb)]) 

params = {'PCA__n_components': [5,6,7,8,9]}

kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 24)

gscv = GridSearchCV(pipe, param_grid=params, cv=kfold, scoring='neg_log_loss')

gscv.fit(X,y)

print("\nUsing Gaussian NB")
print("Best Parameters:")
print(gscv.best_params_)
print("Best neg log loss:",gscv.best_score_)

############################### Random Forest ##################################

rfc = RandomForestClassifier(random_state = 24)

pipe = Pipeline([('SCL',scaler), ('PCA',prc_comp), ('TREE',rfc)]) 

params = {'PCA__n_components': [5,6,7,8,9],
          'TREE__max_depth': [None, 4,3,2]}

kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 24)

gscv = GridSearchCV(pipe, param_grid=params, cv=kfold, scoring='neg_log_loss')

gscv.fit(X,y)

print("\nUsing Random Forest")
print("Best Parameters:")
print(gscv.best_params_)
print("Best neg log loss:",gscv.best_score_)

#################################### TNSE ######################################

params = {}

kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 24)

gscv = GridSearchCV(lr, param_grid=params, cv=kfold, scoring='neg_log_loss')

tnse = TSNE (n_components=2, random_state= 24, perplexity= 20).set_output(transform='pandas')
embed_tsne = tnse.fit_transform(X)

gscv.fit(embed_tsne,y)

print("\nUsing TSNE and LogReg")
print("Best Parameters:")
print(gscv.best_params_)
print("Best neg log loss:",gscv.best_score_)