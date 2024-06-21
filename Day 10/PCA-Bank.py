import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import log_loss, accuracy_score
from warnings import filterwarnings

filterwarnings('ignore')

bank = pd.read_csv(r"Bankruptcy.csv")
X = bank.drop(['D','NO'], axis=1)
y = bank['D']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=24,stratify=y)

lr = LogisticRegression()
scaler  = StandardScaler().set_output(transform='pandas')
prc_comp = PCA(n_components=11).set_output(transform='pandas')

# X_trn_scl = scaler.fit_transform(X_train)
# X_trn_pca = prcomp.fit_transform(X_trn_scl)

pipe = Pipeline([('SCL',scaler), ('PCA',prc_comp), ('LR',lr)]) 

pipe.fit(X_train, y_train)

print(np.cumsum(prc_comp.explained_variance_ratio_ * 100))

y_pred = pipe.predict(X_test)
y_pred_prob = pipe.predict_proba(X_test)

print("Accuracy Score:",accuracy_score(y_test, y_pred))
print("Log Loss:",log_loss(y_test, y_pred_prob))

#######################################################################################################

params = {'PCA__n_components': np.arange(6, 12),
          'LR__C': np.linspace(0.001, 3, 5)}

kfold = StratifiedKFold(n_splits=5, shuffle= True, random_state= 24)

gscv = GridSearchCV(pipe, param_grid = params, cv = kfold, scoring = 'neg_log_loss')

gscv.fit(X, y)

print("Best Parameters:")
print(gscv.best_params_)
print("Best neg log loss:",gscv.best_score_)