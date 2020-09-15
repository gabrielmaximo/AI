import matplotlib.pyplot as plt
import pandas as pd

df =pd.read_csv('data.csv')


print(df['thinkingTimeTotal'].mean())
print(df['score'].mean())



plt.boxplot(df.score)
#plt.boxplot(df.thinkingTimeTotal)

plt.show()
