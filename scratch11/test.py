import pandas as pd

df = pd.DataFrame([[1, 2],
                   [3, 4],
                   [5, 6]])
print('\nDF')
print(df)
print('\nmean')
print(df.mean())
print('\nmean axis=1')
print(df.mean(axis=1))
df.info()
