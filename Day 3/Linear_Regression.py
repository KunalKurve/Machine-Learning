import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

pizza = pd.read_csv('pizza.csv')
print(pizza)

xi = pizza['Promote']
yi = pizza['Sales']
n = pizza.shape[0]

xbar = np.mean(xi)
ybar = np.mean(yi)

m_xi_yi = np.sum(xi * yi) / n
m_xi_2 = np.sum(xi ** 2) / n

b1 = (m_xi_yi - (xbar * ybar))/(m_xi_2 - (xbar ** 2))
b0 = ybar - (b1 * xbar)

###############################################################################

# Why do we put '[[]]'?
# X should be 2D Numpy Array to perform dot fit. y can be 1D or 2D array

X = pizza[['Promote']]
y = pizza['Sales']

lr = LinearRegression()
lr.fit(X, y)

print(lr.intercept_)
print(lr.coef_)

###############################################################################

# linag: in numpy, used to perform linear algebra

A = np.array([[2,1],[3,2]])
b = np.array([4,7])

print(np.linalg.solve(A, b))

###############################################################################

insure = pd.read_csv('Insure_auto.csv', index_col = 0)
insure

x1i = insure['Home']
x2i = insure['Automobile']
yi = insure['Operating_Cost']
n = insure.shape[0]

x1bar = np.mean(x1i)
x2bar = np.mean(x2i)
ybar = np.mean(yi)

sum_x1i_yi = np.sum(x1i * yi)
sum_x1i = np.sum(x1i)
sum_x1i_2 = np.sum(x1i ** 2)
sum_x1i_x2i = np.sum(x1i * x2i)
sum_x2i_yi = np.sum(x2i * yi)
sum_x2i = np.sum(x2i)
sum_x2i_2 = np.sum(x2i ** 2)

# b1 = (m_xi_yi - (xbar * ybar))/(m_xi_2 - (xbar ** 2))
# b0 = ybar - (b1 * xbar)

A = np.array([[sum_x1i, sum_x1i_2, sum_x1i_x2i],
              [sum_x2i, sum_x1i_x2i, sum_x2i_2],
              [1, x1bar, x2bar]])

b = np.array([sum_x1i_yi, sum_x2i_yi, ybar])

print("Using Linear Algebra:", np.linalg.solve(A, b))

X = insure[['Home', 'Automobile']]
y = insure['Operating_Cost']

lr = LinearRegression()
lr.fit(X, y)

print("Linear Regression Intercept:",lr.intercept_)
print("Linear Regression Coefficient:",lr.coef_)


# Sklearn has generalised what we have done above. 