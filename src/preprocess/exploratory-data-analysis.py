import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Set style for better plots
plt.style.use("default")
sns.set_palette("husl")

PROJECT_PATH = Path("/home/whilebell/Code/northernth-sentiment-analysis/")

# Load the datasets
df = pd.read_csv(PROJECT_PATH / "data/interim/cleaned/cleaned_reviews_data.csv")
lang_distribution = pd.read_csv(
    PROJECT_PATH / "data/interim/cleaned/language_distribution.csv"
)

print("=" * 60)
print("COMPREHENSIVE DATA ANALYSIS REPORT")
print("=" * 60)

# 1. BASIC DATASET INFORMATION
print("\n1. BASIC DATASET INFORMATION")
print("-" * 40)
print(f"Total number of reviews: {len(df):,}")
print(f"Number of columns: {len(df.columns)}")
print(f"Dataset shape: {df.shape}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Display column information
print("\nColumn names and types:")
for col in df.columns:
    print(f"  - {col}: {df[col].dtype} (non-null: {df[col].notna().sum():,})")

# 2. MISSING DATA ANALYSIS
print("\n\n2. MISSING DATA ANALYSIS")
print("-" * 40)
missing_data = df.isnull().sum()
missing_percentage = (missing_data / len(df)) * 100

missing_summary = pd.DataFrame(
    {"Missing Count": missing_data, "Missing Percentage": missing_percentage}
).round(2)

print(missing_summary[missing_summary["Missing Count"] > 0])

if missing_summary["Missing Count"].sum() == 0:
    print("✅ No missing values found in the dataset!")

# 3. RATING ANALYSIS
print("\n\n3. DETAILED RATING ANALYSIS")
print("-" * 40)

# Basic rating statistics
print("Summary statistics of rating_review:")
rating_stats = df["rating_review"].describe()
print(rating_stats)

# Rating distribution
print("\nRating distribution:")
rating_counts = df["rating_review"].value_counts().sort_index()
print(rating_counts)

# Rating percentages
print("\nRating distribution (percentages):")
rating_percentages = df["rating_review"].value_counts(normalize=True).sort_index() * 100
for rating, percentage in rating_percentages.items():
    print(f"  {rating} stars: {percentage:.1f}%")

# 4. TEXT LENGTH ANALYSIS
print("\n\n4. TEXT LENGTH ANALYSIS")
print("-" * 40)

# Assuming there's a text column - adjust column name as needed
text_columns = ["cleaned_text", "review_text", "text", "content"]
text_col = None

for col in text_columns:
    if col in df.columns:
        text_col = col
        break

if text_col:
    # Calculate text lengths
    df["text_length"] = df[text_col].astype(str).str.len()
    df["word_count"] = df[text_col].astype(str).str.split().str.len()

    print("Text length statistics (characters):")
    print(df["text_length"].describe())

    print("\nWord count statistics:")
    print(df["word_count"].describe())

    # Text length by rating
    print("\nAverage text length by rating:")
    text_by_rating = (
        df.groupby("rating_review")["text_length"]
        .agg(["mean", "median", "std"])
        .round(1)
    )
    print(text_by_rating)

else:
    print("❌ No text column found for analysis")

# 5. LANGUAGE DISTRIBUTION ANALYSIS
print("\n\n5. LANGUAGE DISTRIBUTION ANALYSIS")
print("-" * 40)
print("Language distribution from the dataset:")
print(lang_distribution.head(10))

if "detected_language" in df.columns:
    print("\nEnglish vs Non-English breakdown:")
    english_count = (df["detected_language"] == "en").sum()
    total_count = len(df)
    print(
        f"  English reviews: {english_count:,} ({english_count / total_count * 100:.1f}%)"
    )
    print(
        f"  Non-English reviews: {total_count - english_count:,} ({(total_count - english_count) / total_count * 100:.1f}%)"
    )

# 6. SENTIMENT LABEL CREATION & ANALYSIS
print("\n\n6. SENTIMENT LABEL ANALYSIS")
print("-" * 40)


# Create sentiment labels based on ratings
def create_sentiment_labels(rating):
    if pd.isna(rating):
        return "unknown"
    elif rating == 5:
        return "Very Positive"
    elif rating == 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    elif rating == 2:
        return "Negative"
    elif rating == 1:
        return "Very Negative"
    else:
        return "unknown"


df["sentiment_label"] = df["rating_review"].apply(create_sentiment_labels)

# Sentiment distribution
sentiment_counts = df["sentiment_label"].value_counts()
print("Sentiment distribution:")
for sentiment, count in sentiment_counts.items():
    percentage = count / len(df) * 100
    print(f"  {sentiment.capitalize()}: {count:,} ({percentage:.1f}%)")

# Check for class imbalance
print("\nClass balance analysis:")
max_class = sentiment_counts.max()
min_class = sentiment_counts.min()
imbalance_ratio = max_class / min_class
print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")

if imbalance_ratio > 3:
    print("  ⚠️  WARNING: Significant class imbalance detected!")
    print(
        "  Consider using techniques like SMOTE, class weights, or stratified sampling"
    )
else:
    print("  ✅ Classes are reasonably balanced")

# 7. DATE ANALYSIS (if date column exists)
print("\n\n7. TEMPORAL ANALYSIS")
print("-" * 40)

date_columns = ["date", "review_date", "created_date", "timestamp"]
date_col = None

for col in date_columns:
    if col in df.columns:
        date_col = col
        break

if date_col:
    try:
        df[date_col] = pd.to_datetime(df[date_col])
        print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
        print(f"Time span: {(df[date_col].max() - df[date_col].min()).days} days")

        # Reviews by year (if applicable)
        if df[date_col].dt.year.nunique() > 1:
            yearly_counts = df[date_col].dt.year.value_counts().sort_index()
            print("\nReviews by year:")
            for year, count in yearly_counts.items():
                print(f"  {year}: {count:,}")
    except (ValueError, TypeError):
        print("Date column found but couldn't be parsed")
    except KeyError:
        print("Date column not found in DataFrame")
else:
    print("No date column found")

# 8. DATA QUALITY INDICATORS
print("\n\n8. DATA QUALITY INDICATORS")
print("-" * 40)

quality_indicators = {}

# Duplicate check
duplicates = df.duplicated().sum()
quality_indicators["Duplicate rows"] = (
    f"{duplicates:,} ({duplicates / len(df) * 100:.2f}%)"
)

# Rating validity
invalid_ratings = df[(df["rating_review"] < 1) | (df["rating_review"] > 5)].shape[0]
quality_indicators["Invalid ratings"] = f"{invalid_ratings:,}"

# Text quality (if text column exists)
if text_col:
    very_short_reviews = df[df["text_length"] < 10].shape[0]
    very_long_reviews = df[df["text_length"] > 2000].shape[0]
    quality_indicators["Very short reviews (<10 chars)"] = f"{very_short_reviews:,}"
    quality_indicators["Very long reviews (>2000 chars)"] = f"{very_long_reviews:,}"

for indicator, value in quality_indicators.items():
    print(f"  {indicator}: {value}")

# 9. RECOMMENDATIONS
print("\n\n9. RECOMMENDATIONS FOR MODEL TRAINING")
print("-" * 40)

recommendations = []

# Class imbalance
if imbalance_ratio > 3:
    recommendations.append(
        "• Address class imbalance using SMOTE, class weights, or stratified sampling"
    )

# Text length
if text_col and df["text_length"].std() > df["text_length"].mean():
    recommendations.append(
        "• Consider text length normalization or filtering extreme lengths"
    )

# Missing data
if missing_summary["Missing Count"].sum() > 0:
    recommendations.append("• Handle missing data through imputation or removal")

# Duplicates
if duplicates > 0:
    recommendations.append(f"• Remove {duplicates:,} duplicate entries")

# Data size
if len(df) < 1000:
    recommendations.append("• Dataset size might be small for deep learning models")
elif len(df) > 100000:
    recommendations.append("• Consider sampling strategies for faster training")

# Rating distribution
negative_pct = sentiment_counts.get("negative", 0) / len(df) * 100
if negative_pct < 10:
    recommendations.append(
        "• Low negative sentiment samples may affect model performance"
    )

if recommendations:
    print("\n".join(recommendations))
else:
    print("✅ Dataset appears to be in good condition for model training!")

# 10. SAVE ENHANCED DATASET
print("\n\n10. SAVING ENHANCED DATASET")
print("-" * 40)

# Add computed columns to the dataset
enhanced_df = df.copy()

# Save enhanced dataset
output_path = PROJECT_PATH / "data/interim/enhanced_reviews_with_labels.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
enhanced_df.to_csv(output_path, index=False)

print(f"Enhanced dataset saved to: {output_path}")
print("New columns added: sentiment_label, text_length, word_count")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE!")
print("=" * 60)
