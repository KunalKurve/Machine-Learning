import numpy as np

# All the items
a = [3,5,6,8,1,2,4]

# Items rated by the user
b = [8,1,4]

# Items not rated by the user
print(np.setdiff1d(a,b))