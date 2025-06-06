import pandas as pd
import numpy as np
import re
from pathlib import Path

PROJECT_PATH = Path("/home/whilebell/Code/Project/Northern-Thailand-Review-Analysis")

# Load the merged dataset
df = pd.read_csv(PROJECT_PATH / "data/interim/merged_reviews.csv")

# Show example rows and columns
print(df.head())
