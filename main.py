import pandas as pd

df = pd.read_csv("datasets/dataset_train.csv")
df_numeric = df.select_dtypes(include='number')

print(df_numeric.describe())
