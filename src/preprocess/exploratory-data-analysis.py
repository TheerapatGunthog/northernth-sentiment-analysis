import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from transformers import pipeline  # <-- Added import

warnings.filterwarnings("ignore")

# Before running the code, make sure to install the necessary libraries
# pip install pandas matplotlib seaborn transformers torch

# Set style for better plots
plt.style.use("default")
sns.set_palette("husl")

PROJECT_PATH = Path("/home/whilebell/Code/northernth-sentiment-analysis/")

# Load the datasets
df = pd.read_csv(PROJECT_PATH / "data/interim/cleaned/cleaned_reviews_data.csv")
lang_distribution = pd.read_csv(
    PROJECT_PATH / "data/interim/cleaned/language_distribution.csv"
)

# ====================================================================
# Sections 1-5 remain unchanged (skipping for brevity)
# ... Your code for Sections 1-5 ...
# Assume the code is executed up to this point
# ====================================================================

# Check if there is a text column (from Section 4 of your code)
text_columns = ["cleaned_text", "review_text", "text", "content"]
text_col = None
for col in text_columns:
    if col in df.columns:
        text_col = col
        break

df["text_length"] = df[text_col].astype(str).str.len()
df["word_count"] = df[text_col].astype(str).str.split().str.len()


# 6. SENTIMENT PREDICTION & ANALYSIS (USING NLP MODEL)
print("\n\n6. SENTIMENT PREDICTION & ANALYSIS (USING NLP MODEL)")
print("-" * 50)

# ### MODIFIED SECTION START ###

# 1. Load Sentiment Analysis model
print("Loading NLP model: nlptown/bert-base-multilingual-uncased-sentiment...")
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    # If using GPU, add device=0
)
print("✅ Model loaded successfully.")

# 2. Prepare Text Data for Prediction
# Convert to list and handle non-string values
texts_to_analyze = df[text_col].astype(str).tolist()

# 3. Predict Sentiments using the Model
print(
    f"Predicting sentiment for {len(texts_to_analyze):,} reviews... (This may take a while)"
)
# Use batch_size for faster processing and truncation=True to handle overly long texts
predictions = sentiment_analyzer(texts_to_analyze, batch_size=16, truncation=True)
print("✅ Prediction complete.")

# 4. Extract Results (labels) and Convert to Numeric
# Results will be in the format {'label': '5 stars', 'score': 0.8...}
# Extract '5 stars' and convert to number 5
predicted_ratings_str = [p["label"] for p in predictions]
predicted_ratings_numeric = [int(s.split()[0]) for s in predicted_ratings_str]

# Add a new column to the DataFrame
df["predicted_rating"] = predicted_ratings_numeric


# 5. Create Sentiment Labels from Model Predictions
# (Using your existing function)
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


df["predicted_sentiment_label"] = df["predicted_rating"].apply(create_sentiment_labels)

print("\nSentiment distribution based on model prediction:")
sentiment_counts = df["predicted_sentiment_label"].value_counts()
for sentiment, count in sentiment_counts.items():
    percentage = count / len(df) * 100
    print(f"  {sentiment.capitalize()}: {count:,} ({percentage:.1f}%)")


print("\nClass balance analysis (based on prediction):")
max_class = sentiment_counts.max()
min_class = sentiment_counts.min()
imbalance_ratio = max_class / min_class
print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")

if imbalance_ratio > 3:
    print("  ⚠️  WARNING: Significant class imbalance detected in predictions!")
    print(
        "  Consider using techniques like SMOTE, class weights, or stratified sampling"
    )
else:
    print("  ✅ Predicted classes are reasonably balanced")

# ### MODIFIED SECTION END ###


# ====================================================================
# Sections 7-9 will proceed using the data generated from the model
# ... Your code for Sections 7-9 ...
# ====================================================================


# 10. SAVE ENHANCED DATASET
print("\n\n10. SAVING ENHANCED DATASET")
print("-" * 40)

# Copy the final DataFrame
enhanced_df = df.copy()

# Save enhanced dataset
output_path = (
    PROJECT_PATH / "data/interim/enhanced_reviews_with_predictions.csv"
)  # Change the file name
output_path.parent.mkdir(parents=True, exist_ok=True)
enhanced_df.to_csv(output_path, index=False)

print(f"Enhanced dataset saved to: {output_path}")
print(
    "New columns added: predicted_rating, predicted_sentiment_label, text_length, word_count"
)

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE!")
print("=" * 60)
