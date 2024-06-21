import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r'D:/March 2024/PML/Day 13/Bike Sharing Demand-Kaggle/train.csv', parse_dates=['datetime'])
df.set_index('datetime', inplace = True)

casual = df['casual']
monthly_cas = casual.resample('M').sum()
monthly_cas.index.rename('Month', inplace = True)

monthly_cas.plot()
plt.title("Monthly Casual Rentals")
plt.xlabel("Months")
plt.show()

monthly_cas = casual.resample('Q').sum()
monthly_cas.index.rename('Quarter', inplace = True)
monthly_cas.plot()
plt.title("Quarterly Casual Rentals")
plt.xlabel("Quarters")
plt.show()

register = df['registered']
monthly_reg = register.resample('Q').sum()
monthly_reg.index.rename("Quarter", inplace = True)
monthly_reg.plot()
plt.title("Quarterly Registered Rentals")
plt.xlabel("Quarters")
plt.show()