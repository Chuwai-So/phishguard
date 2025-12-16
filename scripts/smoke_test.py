from phishguard.data.loader import load_csv, split_df

df = load_csv("data/raw/dataset.csv")
splits = split_df(df)
print(df.head())
print("Splits:", len(splits.train), len(splits.val), len(splits.test))
print("Train labels:\n", splits.train["label"].value_counts())