import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x = np.array([[2,4],
              [3,5],
              [12,18],
              [15,20],
              [34,56],
              [35,60],
              [78, 26],
              [80, 23]])
y = np.array(["1","1","0","0","1","1","0","0"])

p_df = pd.DataFrame(x, columns=['x1', 'x2'])
p_df['y'] = y

sns.scatterplot(data=p_df, x='x1',y='x2', hue='y')
plt.show()

X = p_df[['x1','x2']]
y = p_df['y']

dtc = DecisionTreeClassifier(random_state=24)
dtc.fit(X, y)

plt.figure(figsize=(15,10))
plot_tree(dtc, feature_names=list(X.columns), class_names=['0','1'], filled=True, fontsize=18)

plt.show()

## data 2

x = np.array([[2,4],
              [3,5],
              [12,18],
              [15,20],
              [34,56],
              [35,60],
              [78, 26],
              [80, 23],
              [40, 20],
              [50, 30]])
y = np.array(["1","1","0","0","1",
              "1","0","0","1","1"])

p_df = pd.DataFrame(x, columns=['x1', 'x2'])
p_df['y'] = y

sns.scatterplot(data=p_df, x='x1',y='x2', hue='y')
plt.show()

X = p_df[['x1','x2']]
y = p_df['y']

dtc = DecisionTreeClassifier(random_state=24)
dtc.fit(X, y)

plt.figure(figsize=(15,10))
plot_tree(dtc, feature_names=list(X.columns), class_names=['0','1'], filled=True, fontsize=18)

plt.show()


## data 3

x = np.array([[2,4],
              [3,5],
              [12,18],
              [15,20],
              [34,56],
              [35,60],
              [78, 26],
              [80, 23],
              [40, 20],
              [50, 30],
              [40, 30],
              [10, 30],
              [80,40],
              [5,40]])
y = np.array(["1","1","0","0","1",
              "1","0","0","1","1",
              "0","1","1","0"])

p_df = pd.DataFrame(x, columns=['x1', 'x2'])
p_df['y'] = y

sns.scatterplot(data=p_df, x='x1', y='x2', hue='y')
plt.show()

X = p_df[['x1','x2']]
y = p_df['y']

dtc = DecisionTreeClassifier(random_state=24)
dtc.fit(X, y)

plt.figure(figsize=(25,20))
plot_tree(dtc, feature_names=list(X.columns), class_names=['0','1'], filled=True, fontsize=18)

plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=24, stratify=y)

dtc = DecisionTreeClassifier(random_state=24, max_depth=1)
dtc.fit(X_train, y_train)

plt.figure(figsize=(25,20))
plot_tree(dtc, feature_names=list(X.columns), class_names=['0','1'], filled=True, fontsize=18)

plt.show()

y_pred = dtc.predict(X_test)
print(accuracy_score(y_test, y_pred))


# tst = np.array([[30,40],
#                 [5, 20],
#                 [50, 50]])

# X_test = pd.DataFrame(tst,columns=['x1', 'x2'] )
# dtc.predict(X_test)

###############################################################################################



