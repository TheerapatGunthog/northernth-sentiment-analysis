import pandas as pd
from pathlib import Path

PROJECT_PATH = Path("/home/whilebell/Code/northernth-sentiment-analysis/")

df = pd.read_csv(PROJECT_PATH / "data/analyst/sentiment_analysis_results.csv")

# Show all columns names
print(df.columns)

df[["title", "source"]]

normalized_title = {
    "Bua Thong Waterfalls (Nam Phu Chet Si)": "Buatong Waterfall-Chet Si Fountain National Park",
    "Saturday Night Market  Walking Street - Wua Lai Road": "Wua Lai Walking Street Saturday Market",
    "Grand Canyon Chiang Mai": "Grand Canyon Water Park",
    "Maeklang Elephant Conservation Community": "Thai Elephant Conservation Center",
    "Umbrella Village - Bor Sang": "Bo Sang Umbrella Village",
    "Wat Phra Singh": "Wat Phra Singh Woramahawihan",
    "Lanna Folklife Museum": "Lanna Folklife Centre",
    "Kad Rin Come Night Market": "Rincome night market",
    "Sunday Night Market": "Chiang Mai Night Market",
    "Chang Phuak Gate Night Market": "Chang Phuak Market",
    "Wat Chedi Luang Varavihara": "Watchediluang Varaviharn",
    "Doi Inthanon": "Doi Inthanon National Park",
    "Tiger Kingdom - Chiang Mai": "Tiger Kingdom",
}

df["title"] = df["title"].replace(normalized_title)

titles = df["title"].drop_duplicates()

# Save the updated DataFrame to a new CSV file
df.to_csv(
    PROJECT_PATH / "data/analyst/sentiment_analysis_results_normalized.csv", index=False
)
