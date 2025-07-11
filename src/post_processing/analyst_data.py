import pandas as pd
from pathlib import Path
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# --- Initial Setup ---
PROJECT_PATH = Path("/home/whilebell/Code/northernth-sentiment-analysis/")

# Load datasets
GOOGLE_REVIEW_DATA = pd.read_csv(
    PROJECT_PATH / "data/raw/analyst_data/google_review.csv"
)
TRIPADVISOR_REVIEW_DATA = pd.read_csv(
    PROJECT_PATH / "data/raw/analyst_data/tripadvisor_review.csv"
)


# Select relevant columns
selected_columns = ["title", "text_review"]
GOOGLE_REVIEW_DATA = GOOGLE_REVIEW_DATA[selected_columns]
TRIPADVISOR_REVIEW_DATA = TRIPADVISOR_REVIEW_DATA[selected_columns]

# --- Text Preprocessing ---

# Drop all row with NaN values in 'text_review' column
GOOGLE_REVIEW_DATA = GOOGLE_REVIEW_DATA.dropna(subset=["text_review"])
TRIPADVISOR_REVIEW_DATA = TRIPADVISOR_REVIEW_DATA.dropna(subset=["text_review"])

# Download required NLTK data
import nltk

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt_tab")

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def preprocess_text(text):
    """
    Cleans and preprocesses a single text string.
    """
    if not isinstance(text, str):
        return ""

    # 1. Removing URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

    # 2. Removing Punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # 3. Tokenization
    tokens = word_tokenize(text)

    # 4. Removing Stopwords and Lemmatization
    processed_tokens = [
        lemmatizer.lemmatize(word) for word in tokens if word not in stop_words
    ]

    return " ".join(processed_tokens)


# Apply the preprocessing function to the 'text_review' column
GOOGLE_REVIEW_DATA["cleaned_review"] = GOOGLE_REVIEW_DATA["text_review"].apply(
    preprocess_text
)
TRIPADVISOR_REVIEW_DATA["cleaned_review"] = TRIPADVISOR_REVIEW_DATA[
    "text_review"
].apply(preprocess_text)

# --- Display Results ---
print("Google Review Data with Cleaned Text:")
print(GOOGLE_REVIEW_DATA[["text_review", "cleaned_review"]].head())

print("\nTripAdvisor Review Data with Cleaned Text:")
print(TRIPADVISOR_REVIEW_DATA[["text_review", "cleaned_review"]].head())

# Select column
slected_columns = ["title", "cleaned_review"]
GOOGLE_REVIEW_DATA = GOOGLE_REVIEW_DATA[slected_columns]
TRIPADVISOR_REVIEW_DATA = TRIPADVISOR_REVIEW_DATA[slected_columns]

# Model inference
