import numpy as np
import pandas as pd

kyp = pd.read_csv("Kyphosis.csv")

kyp_ind = list(kyp.index)

#This is called "Simple Random Sampling Without Replacement", often abbrievated as SRSWOR

samp_ind1 = np.random.choice(kyp_ind, size = 60, replace = False)
samp_kyp1 = kyp.iloc[samp_ind1, : ]

#This is called "Simple Random Sampling With Replacement", often abbrievated as SRSWR
#generally 33 percent of the data is repeated.

samp_ind2 = np.random.choice(kyp_ind, size = 60, replace = True)

# this is also called as a Bootstrap Sample.
samp_kyp2 = kyp.iloc[samp_ind2, : ]