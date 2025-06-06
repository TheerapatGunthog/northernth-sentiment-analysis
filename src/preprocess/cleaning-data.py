import pandas as pd
import numpy as np
import re
from pathlib import Path
import unicodedata
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Set seed for consistent language detection
DetectorFactory.seed = 0

PROJECT_PATH = Path("/home/whilebell/Code/Project/Northern-Thailand-Review-Analysis")

# Load the merged dataset
df = pd.read_csv(PROJECT_PATH / "data/interim/merged_reviews.csv")

print("Original dataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nSample data:")
print(df.head())


class TextCleaner:
    def __init__(self):
        # Compile regex patterns for efficiency
        self.html_pattern = re.compile(r"<[^>]+>")
        self.url_pattern = re.compile(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        )
        self.email_pattern = re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        )
        self.phone_pattern = re.compile(
            r"(\+?\d{1,4}[-.\s]?)?\(?\d{1,3}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}"
        )
        self.mention_pattern = re.compile(r"@\w+")
        self.hashtag_pattern = re.compile(r"#\w+")
        self.extra_whitespace = re.compile(r"\s+")
        self.repeated_chars = re.compile(r"(.)\1{2,}")  # 3+ repeated characters
        self.special_chars = re.compile(r'[^\w\s.,!?;:\'"()-]')

    def detect_language(self, text):
        """Detect language of text"""
        try:
            if pd.isna(text) or len(str(text).strip()) < 3:
                return "unknown"
            return detect(str(text))
        except LangDetectException:
            return "unknown"

    def remove_html_tags(self, text):
        """Remove HTML tags"""
        if pd.isna(text):
            return text
        return self.html_pattern.sub(" ", str(text))

    def remove_urls(self, text):
        """Remove URLs"""
        if pd.isna(text):
            return text
        return self.url_pattern.sub(" ", str(text))

    def remove_emails(self, text):
        """Remove email addresses"""
        if pd.isna(text):
            return text
        return self.email_pattern.sub(" ", str(text))

    def remove_phone_numbers(self, text):
        """Remove phone numbers"""
        if pd.isna(text):
            return text
        return self.phone_pattern.sub(" ", str(text))

    def remove_social_media_elements(self, text):
        """Remove @mentions and #hashtags"""
        if pd.isna(text):
            return text
        text = self.mention_pattern.sub(" ", str(text))
        text = self.hashtag_pattern.sub(" ", text)
        return text

    def normalize_unicode(self, text):
        """Normalize unicode characters"""
        if pd.isna(text):
            return text
        # Convert to NFD (decomposed) form and remove accents
        text = unicodedata.normalize("NFD", str(text))
        # Keep only ASCII characters and common punctuation
        text = "".join(c for c in text if unicodedata.category(c) != "Mn")
        return text

    def fix_repeated_characters(self, text):
        """Fix repeated characters (e.g., 'sooooo' -> 'soo')"""
        if pd.isna(text):
            return text
        return self.repeated_chars.sub(r"\1\1", str(text))

    def remove_special_characters(self, text):
        """Remove special characters but keep basic punctuation"""
        if pd.isna(text):
            return text
        return self.special_chars.sub(" ", str(text))

    def normalize_whitespace(self, text):
        """Normalize whitespace"""
        if pd.isna(text):
            return text
        text = self.extra_whitespace.sub(" ", str(text))
        return text.strip()

    def convert_to_lowercase(self, text):
        """Convert to lowercase"""
        if pd.isna(text):
            return text
        return str(text).lower()

    def remove_extra_punctuation(self, text):
        """Remove excessive punctuation"""
        if pd.isna(text):
            return text
        # Replace multiple punctuation with single
        text = re.sub(r"[.]{2,}", ".", str(text))
        text = re.sub(r"[!]{2,}", "!", text)
        text = re.sub(r"[?]{2,}", "?", text)
        return text

    def clean_text(self, text):
        """Apply all cleaning steps"""
        # Chain all cleaning methods
        text = self.remove_html_tags(text)
        text = self.remove_urls(text)
        text = self.remove_emails(text)
        text = self.remove_phone_numbers(text)
        text = self.remove_social_media_elements(text)
        text = self.normalize_unicode(text)
        text = self.remove_special_characters(text)
        text = self.fix_repeated_characters(text)
        text = self.remove_extra_punctuation(text)
        text = self.normalize_whitespace(text)
        text = self.convert_to_lowercase(text)

        return text if text and len(text.strip()) > 0 else np.nan


# Initialize cleaner
cleaner = TextCleaner()

# Assume the review text column is named 'review_text' or 'text'
# Adjust column name based on your dataset
review_column = "text_review"  # Change this to match your column name

# Show examples before cleaning
print("\n" + "=" * 50)
print("BEFORE CLEANING - Sample Reviews:")
print("=" * 50)
for i in range(min(3, len(df))):
    print(f"\nReview {i + 1}:")
    print(f"Original: {df[review_column].iloc[i]}")

# Detect language and filter English reviews
df["detected_language"] = df[review_column].apply(cleaner.detect_language)

# Show language distribution
print(df["detected_language"].value_counts())

# Filter English reviews
english_mask = df["detected_language"] == "en"
df_english = df[english_mask].copy()

print(
    f"\nFiltered to English reviews: {len(df_english)} out of {len(df)} ({len(df_english) / len(df) * 100:.1f}%)"
)

# Clean the text
df_english["cleaned_text"] = df_english[review_column].apply(cleaner.clean_text)

# Remove rows with empty cleaned text
df_english = df_english.dropna(subset=["cleaned_text"])
df_english = df_english[
    df_english["cleaned_text"].str.len() > 10
]  # Keep reviews with at least 10 characters

# Show remaining data after cleaning
print(len(df_english))

# Show examples after cleaning
print("\n" + "=" * 50)
print("AFTER CLEANING - Sample Reviews:")
print("=" * 50)
for i in range(min(3, len(df_english))):
    print(f"\nReview {i + 1}:")
    print(f"Original: {df_english[review_column].iloc[i]}")
    print(f"Cleaned:  {df_english['cleaned_text'].iloc[i]}")
    print(f"Language: {df_english['detected_language'].iloc[i]}")

# Basic statistics
print("\n" + "=" * 50)
print("CLEANING STATISTICS:")
print("=" * 50)
print(f"Original reviews: {len(df):,}")
print(f"English reviews: {len(df_english):,}")
print(f"Successfully cleaned: {len(df_english):,}")
print(
    f"Average review length: {df_english['cleaned_text'].str.len().mean():.1f} characters"
)
print(
    f"Median review length: {df_english['cleaned_text'].str.len().median():.1f} characters"
)

# Select relevant columns for final output
# Choose only rating_review and cleaned_text columns
df_english = df_english[["rating_review", "cleaned_text"]].copy()

# Save cleaned data
output_path = PROJECT_PATH / "data/interim/cleaned/cleaned_reviews_data.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
df_english.to_csv(output_path, index=False)
print(f"\nCleaned data saved to: {output_path}")

# Optional: Save language distribution for analysis
lang_dist = df["detected_language"].value_counts()
lang_dist.to_csv(PROJECT_PATH / "data/interim/cleaned/language_distribution.csv")
print(
    f"Language distribution saved to: {PROJECT_PATH / 'data/interim/cleaned/language_distribution.csv'}"
)
