# -*- coding: utf-8 -*-
"""
Created on Wed May  1 12:20:22 2024

@author: Administrator
"""

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
a = pd.DataFrame({'x1':[10, 9, 11],
                  'x2':[0.1, 0.7, 0.1]})

scl_std = StandardScaler()

scl_std.fit(a)
print("Means:")
print(scl_std.mean_)
print("Standard Deviations:")
print(scl_std.scale_)

scl_std.transform(a)
# or
scl_std.fit_transform(a)

scl_m = MinMaxScaler()

scl_m.fit(a)
print("Min Datas:")
print(scl_m.data_min_)
print("Max Datas:")
print(scl_m.data_max_)

scl_m.transform(a)
