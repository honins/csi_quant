import pandas as pd

file_path = 'data/SHSE.000905_1d.csv'
df = pd.read_csv(file_path)
df.insert(0, 'index', range(1, len(df) + 1))
df.to_csv(file_path, index=False)