# %%

# On BreastCancer

# %%

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
import os

# %%

os.chdir("D:\March 2024\PML\Cases")

df = pd.read_csv(r"D:\March 2024\PML\Cases\Wisconsin\BreastCancer.csv")
dum_df = pd.get_dummies(df,drop_first=True)
X = dum_df.drop('left',axis=1)
y = dum_df['left']