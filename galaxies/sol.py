import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('sol.csv')
print(df.columns)
print(df)

fig, ax = plt.subplots()
x, y = df['Time'], df['Velocity']
ax.scatter(x, y)
plt.errorbar(x, y, yerr=df['Velocity error'], fmt='none', capsize=2)
ax.set_xlabel('Time (s)'); ax.set_ylabel('Doppler Velocity (m/s)')
plt.savefig('sol.png')

