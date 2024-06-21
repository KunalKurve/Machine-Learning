import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('D:/March 2024/PML/Day 13/monthly-milk-production-pounds-p.csv', index_col=0)

df.index = pd.to_datetime(df.index).to_period("M")

df.plot()
plt.title("Monthly Milk Production")
plt.xlabel("Months")
plt.show()

downsampled = df.resample('Q').sum()
downsampled.index.rename('Quarter', inplace = True)
downsampled.plot()
plt.title("Quarterly Milk Production")
plt.xlabel("Quarters")
plt.show()

#####################################################################################

from statsmodels.tsa.seasonal import seasonal_decompose

df = pd.read_csv('D:/March 2024/PML/Day 13/monthly-milk-production-pounds-p.csv')

series = df['Milk']

result = seasonal_decompose(series, model='addtime', period=12)

result.plot()
plt.title("Additive Decomposition")
plt.show()