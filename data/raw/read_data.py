import pandas as pd

data = pd.read_csv("data/raw/predictive_maintenance_sensor_data.csv", sep=',')

print("head",data.head())

print("info",data.info())

print("describe",data.describe())

#variable numérique
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
print("colonnes numériques",numeric_columns)