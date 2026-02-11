import pandas as pd
df = pd.read_csv("featured_data/featured_data.csv")
print(df["failure_within_24h"].value_counts())
print(df["failure_within_24h"].value_counts(normalize=True))