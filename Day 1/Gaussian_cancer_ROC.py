import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, roc_auc_score

sonar =  pd.read_csv("D:\March 2024\PML\Day1\Sonar.csv")
y = sonar['Class']
X = sonar.drop('Class', axis=1)