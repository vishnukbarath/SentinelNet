import pandas as pd
import os

folder = "data"
files = [f for f in os.listdir(folder) if f.endswith(".csv")]

for file in files:
    print("\n====", file, "====")
    df = pd.read_csv(folder + "/" + file, nrows=5)
    print(df.columns.tolist())
