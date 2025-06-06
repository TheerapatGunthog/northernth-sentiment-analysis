import pandas as pd
from pathlib import Path

PROJECT_PATH = Path("/home/whilebell/Code/Project/Northern-Thailand-Review-Analysis")

# Load the datasets
df0 = pd.read_csv(
    PROJECT_PATH / "data/raw/tripadvisor-chiang-mai/tripadvisor-dataset-chiangmai.csv",
    encoding="ISO-8859-1",
)
df1 = pd.read_csv(
    PROJECT_PATH / "data/raw/tripadvisor-reviews-2023/New_Delhi_reviews.csv"
)
df2 = pd.read_csv(PROJECT_PATH / "data/raw/yelp-reviews-dataset/yelp.csv")

# Show the first few rows of each dataset
print(df0.head())
print(df1.head())
print(df2.head())

# Show the column names of each dataset
print(df0.columns)
print(df1.columns)
print(df2.columns)

# Choose only text_review and rating_review columns
df0 = df0[["text_review", "rating_review"]]
df1 = df1[["text_review", "rating_review"]]
df2 = df2[["text_review", "rating_review"]]
print(df0.head())
print(df1.head())
print(df2.head())


# Merge the datasets
df = pd.concat([df0, df1, df2], ignore_index=True)

# Show the first few rows of the merged dataset and show data shape
print(df.head())
print(df.shape)

# Save dataset to csv
df.to_csv(
    PROJECT_PATH / "data/interim/merged_reviews.csv",
    index=False,
    encoding="utf-8",
)
