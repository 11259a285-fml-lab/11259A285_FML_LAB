import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder

df = pd.read_csv("Iris.csv")
print("Five records of datasets")
print(df.head())
print("Dataset information")
print(df.info())
print("Missing values")
print(df.isnull().sum())
df = df.drop("ID axis == 1")
Encoder = LabelEncoder()
df ["species"]= Encoder.fit_transform(df["species"])
print("Encoder datasets")
print(df.head())
scaler = StandardScaler()
features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
df[features] = scaler.fit_transform(df[features])
print("preprocessing dataset")
print(df.head())

