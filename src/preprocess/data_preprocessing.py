import pandas as pd
from sklearn.utils import resample
from imblearn.over_sampling import (
    SMOTE,
)  # This import is present but the SMOTE function is not called in main()
from pathlib import Path
import re
import matplotlib.pyplot as plt

# Configuration
PROJECT_PATH = Path("/home/whilebell/Code/northernth-sentiment-analysis/")
MAX_TEXT_LENGTH = 1000  # Maximum text length
MIN_TEXT_LENGTH = 20  # Minimum text length


def load_data():
    """Load data produced from EDA step"""
    df = pd.read_csv(PROJECT_PATH / "data/interim/enhanced_reviews_with_labels.csv")
    print(f"Original dataset size: {len(df):,} reviews")
    return df


def remove_duplicates(df):
    """Remove duplicate entries"""
    initial_size = len(df)
    df_clean = df.drop_duplicates(subset=["cleaned_text", "rating_review"])
    removed = initial_size - len(df_clean)
    print(f"Removed {removed:,} duplicate reviews")
    return df_clean


def filter_text_length(df, min_length=MIN_TEXT_LENGTH, max_length=MAX_TEXT_LENGTH):
    """Filter text by length"""
    initial_size = len(df)

    # Filter out texts that are too short or too long
    df_filtered = df[
        (df["text_length"] >= min_length) & (df["text_length"] <= max_length)
    ].copy()

    removed = initial_size - len(df_filtered)
    print(f"Removed {removed:,} reviews due to text length constraints")
    print(
        f"  - Too short (<{min_length} chars): {len(df[df['text_length'] < min_length])}"
    )
    print(
        f"  - Too long (>{max_length} chars): {len(df[df['text_length'] > max_length])}"
    )

    return df_filtered


def clean_text_further(text):
    """Perform additional text cleaning"""
    if pd.isna(text):
        return ""

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

    # Remove emails
    text = re.sub(r"\S+@\S+", "", text)

    # Remove unnecessary numbers (such as phone numbers)
    text = re.sub(r"\b\d{10,}\b", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def apply_additional_cleaning(df):
    """Apply additional text cleaning"""
    print("Applying additional text cleaning...")
    df["cleaned_text"] = df["cleaned_text"].apply(clean_text_further)

    # Recalculate lengths after cleaning
    df["text_length"] = df["cleaned_text"].str.len()
    df["word_count"] = df["cleaned_text"].str.split().str.len()

    # Remove texts that are too short after cleaning
    df = df[df["text_length"] >= MIN_TEXT_LENGTH].copy()

    return df


def analyze_class_distribution(df, title="Class Distribution"):
    """Analyze class distribution"""
    print(f"\n{title}:")
    class_counts = df["sentiment_label"].value_counts()
    total = len(df)

    for sentiment, count in class_counts.items():
        percentage = count / total * 100
        print(f"  {sentiment.capitalize()}: {count:,} ({percentage:.1f}%)")

    return class_counts


def handle_class_imbalance_sampling(df, method="undersample"):
    """Handle class imbalance using sampling methods"""

    # Split data by sentiment
    positive = df[df["sentiment_label"] == "positive"]
    neutral = df[df["sentiment_label"] == "neutral"]
    negative = df[df["sentiment_label"] == "negative"]

    print("\nOriginal distribution:")
    print(f"  Positive: {len(positive):,}")
    print(f"  Neutral: {len(neutral):,}")
    print(f"  Negative: {len(negative):,}")

    if method == "undersample":
        # Undersampling: reduce majority class size
        # Use 3 times the size of the minority class (negative) as a target for positive
        # And 1.5 times (half of positive's target) for neutral.
        # This creates a more balanced but not perfectly equal distribution,
        # which can sometimes be better than full balancing.
        target_negative_size = len(negative)
        target_neutral_size = (
            target_negative_size * 2
        )  # Aim for double negative size for neutral
        target_positive_size = (
            target_negative_size * 3
        )  # Aim for triple negative size for positive

        # Ensure we don't try to sample more than available
        positive_sampled = positive.sample(
            n=min(target_positive_size, len(positive)), random_state=42
        )
        neutral_sampled = neutral.sample(
            n=min(target_neutral_size, len(neutral)), random_state=42
        )
        negative_sampled = negative  # Keep all negative as it is the minority

    elif method == "oversample":
        # Oversampling: increase size of minority class
        # Target size is the size of the majority class (positive)
        target_size = len(positive)

        neutral_sampled = resample(
            neutral, n_samples=target_size, random_state=42, replace=True
        )
        negative_sampled = resample(
            negative, n_samples=target_size, random_state=42, replace=True
        )
        positive_sampled = positive  # Keep all positive as it is the majority

    # Combine data
    balanced_df = pd.concat(
        [positive_sampled, neutral_sampled, negative_sampled], ignore_index=True
    )
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(
        drop=True
    )  # shuffle

    print(f"\nAfter {method}:")
    analyze_class_distribution(balanced_df, "Balanced Distribution")

    return balanced_df


def handle_class_imbalance_smote(df):
    """Use SMOTE to handle class imbalance (for numeric data only)"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    print("Applying SMOTE for class balancing...")

    # Create features from text using TF-IDF
    vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
    X_tfidf = vectorizer.fit_transform(df["cleaned_text"])

    # Reduce dimensionality with SVD for performance
    # It's important to use a dimensionality reduction technique before SMOTE on TF-IDF
    # as SMOTE works better in lower-dimensional spaces and on numerical features.
    svd = TruncatedSVD(n_components=100, random_state=42)
    X_reduced = svd.fit_transform(X_tfidf)

    # Map sentiment labels to numbers
    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    y = df["sentiment_label"].map(label_map)

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_reduced, y)

    print(f"After SMOTE: {len(X_resampled):,} samples")

    # Note: Using SMOTE with text data (even after TF-IDF/SVD) means the new samples
    # are synthetic *numerical vectors*, not actual new text. You would typically
    # proceed to train your model directly on X_resampled, y_resampled.
    # Reconstructing text from these vectors is generally not feasible or meaningful.

    # For now, this function only returns the resampled numerical features and labels.
    # If you want to return a dataframe, you'd need to convert X_resampled back
    # into a DataFrame structure with relevant columns, which is non-trivial for text.
    # For a typical ML pipeline, you'd feed X_resampled and y_resampled directly into model training.
    # The current `main` function does not use the output of this SMOTE function directly
    # to form the `df` that is then split, which is something to consider.
    # If using SMOTE, you'd typically split *before* applying SMOTE to only the training set.

    return X_resampled, y_resampled


def create_train_test_split(df, test_size=0.2, val_size=0.1):
    """Split data into train/validation/test using stratified sampling"""
    from sklearn.model_selection import train_test_split

    # Split into train and temp (test + validation)
    X = df[["cleaned_text", "text_length", "word_count"]]
    y = df["sentiment_label"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), stratify=y, random_state=42
    )

    # Split temp into test and validation
    test_ratio = test_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_ratio, stratify=y_temp, random_state=42
    )

    print("\nDataset split:")
    print(f"  Training: {len(X_train):,} ({len(X_train) / len(df) * 100:.1f}%)")
    print(f"  Validation: {len(X_val):,} ({len(X_val) / len(df) * 100:.1f}%)")
    print(f"  Test: {len(X_test):,} ({len(X_test) / len(df) * 100:.1f}%)")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def save_processed_data(df, splits=None):
    """Save the processed data"""
    output_dir = PROJECT_PATH / "data/processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save processed dataset
    df.to_csv(output_dir / "final_processed_data.csv", index=False)
    print(f"Processed dataset saved: {len(df):,} reviews")

    # Save train/val/test splits (if available)
    if splits:
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = splits

        train_df = X_train.copy()
        train_df["sentiment_label"] = y_train
        train_df.to_csv(output_dir / "train_data.csv", index=False)

        val_df = X_val.copy()
        val_df["sentiment_label"] = y_val
        val_df.to_csv(output_dir / "validation_data.csv", index=False)

        test_df = X_test.copy()
        test_df["sentiment_label"] = y_test
        test_df.to_csv(output_dir / "test_data.csv", index=False)

        print("Train/Val/Test splits saved separately")


def visualize_improvements(original_df, processed_df):
    """Create comparison plots before and after preprocessing"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Class distribution comparison
    original_counts = original_df["sentiment_label"].value_counts()
    processed_counts = processed_df["sentiment_label"].value_counts()

    # Original Class Distribution
    axes[0, 0].bar(
        original_counts.index, original_counts.values, alpha=0.7, label="Original"
    )
    axes[0, 0].set_title("Original Class Distribution")
    axes[0, 0].set_ylabel("Count")
    for i, count in enumerate(original_counts.values):
        axes[0, 0].text(i, count, f"{count}", ha="center", va="bottom")

    # Processed Class Distribution (after sampling)
    axes[0, 1].bar(
        processed_counts.index,
        processed_counts.values,
        alpha=0.7,
        label="Processed",
        color="orange",
    )
    axes[0, 1].set_title("Processed Class Distribution")
    axes[0, 1].set_ylabel("Count")
    for i, count in enumerate(processed_counts.values):
        axes[0, 1].text(i, count, f"{count}", ha="center", va="bottom")

    # Text length distribution
    axes[1, 0].hist(original_df["text_length"], bins=50, alpha=0.7, label="Original")
    axes[1, 0].set_title("Original Text Length Distribution")
    axes[1, 0].set_xlabel("Text Length")
    axes[1, 0].set_ylabel("Frequency")

    axes[1, 1].hist(
        processed_df["text_length"],
        bins=50,
        alpha=0.7,
        color="orange",
        label="Processed",
    )
    axes[1, 1].set_title("Processed Text Length Distribution")
    axes[1, 1].set_xlabel("Text Length")
    axes[1, 1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(
        PROJECT_PATH / "data/processed/data_preprocessing_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def main():
    """Main preprocessing pipeline"""
    print("=" * 60)
    print("DATA PREPROCESSING PIPELINE")
    print("=" * 60)

    # 1. Load data
    df = load_data()
    original_df = df.copy()  # Save for comparison

    # 2. Remove duplicates
    df = remove_duplicates(df)

    # 3. Additional text cleaning
    df = apply_additional_cleaning(df)

    # 4. Filter by text length
    df = filter_text_length(df)

    # 5. Analyze current class distribution
    analyze_class_distribution(df, "After Basic Cleaning")

    # 6. Handle class imbalance
    print("\nChoose class imbalance handling method:")
    print("1. Undersampling (recommended for large datasets)")
    print("2. Oversampling")
    print("3. Keep original distribution")

    choice = input("Enter choice (1-3): ").strip()

    if choice == "1":
        df = handle_class_imbalance_sampling(df, method="undersample")
    elif choice == "2":
        df = handle_class_imbalance_sampling(df, method="oversample")
    # Removed the SMOTE option here because it returns X_resampled, y_resampled (numpy arrays)
    # instead of a DataFrame, which would break the subsequent DataFrame operations.
    # If SMOTE is intended, the pipeline needs to be adjusted to work with numerical features
    # (e.g., train/val/test splits would be on X_resampled and y_resampled directly).
    # elif choice == "3":
    #     X_resampled, y_resampled = handle_class_imbalance_smote(df)
    #     # If you uncomment this, you'll need to reconstruct a DataFrame from X_resampled and y_resampled
    #     # or adjust subsequent steps to work with numpy arrays.
    else:
        print("Keeping original distribution")

    # 7. Create train/validation/test splits
    splits = create_train_test_split(df)

    # 8. Save processed data
    save_processed_data(df, splits)

    # 9. Visualize improvements
    visualize_improvements(original_df, df)

    # 10. Final summary
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)
    print(f"Original dataset: {len(original_df):,} reviews")
    print(f"Final dataset: {len(df):,} reviews")
    print(f"Data reduction: {(1 - len(df) / len(original_df)) * 100:.1f}%")

    print("\nFinal class distribution:")
    analyze_class_distribution(df, "Final")

    print(f"\nFiles saved in: {PROJECT_PATH}/data/processed/")
    print("- final_processed_data.csv")
    print("- train_data.csv")
    print("- validation_data.csv")
    print("- test_data.csv")
    print("- data_preprocessing_comparison.png")


if __name__ == "__main__":
    main()
