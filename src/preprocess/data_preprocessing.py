import pandas as pd
from pathlib import Path
import re

# Configuration
PROJECT_PATH = Path("/home/whilebell/Code/northernth-sentiment-analysis/")
MAX_TEXT_LENGTH = 1000  # Maximum text length
MIN_TEXT_LENGTH = 20  # Minimum text length


def load_data():
    """Load data produced from EDA step"""
    df = pd.read_csv(
        PROJECT_PATH / "data/interim/enhanced_reviews_with_predictions.csv"
    )
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
    df_filtered = df[
        (df["text_length"] >= min_length) & (df["text_length"] <= max_length)
    ].copy()
    removed = initial_size - len(df_filtered)
    print(f"Removed {removed:,} reviews due to text length constraints")
    return df_filtered


def clean_text_further(text):
    """Perform additional text cleaning"""
    if pd.isna(text):
        return ""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"\b\d{10,}\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def apply_additional_cleaning(df):
    """Apply additional text cleaning"""
    print("Applying additional text cleaning...")
    df["cleaned_text"] = df["cleaned_text"].apply(clean_text_further)
    df["text_length"] = df["cleaned_text"].str.len()
    df["word_count"] = df["cleaned_text"].str.split().str.len()
    df = df[df["text_length"] >= MIN_TEXT_LENGTH].copy()
    return df


def analyze_class_distribution(df, title="Class Distribution"):
    """Analyze and print class distribution"""
    print(f"\n{title}:")
    class_counts = df["predicted_sentiment_label"].value_counts()
    total = len(df)
    for sentiment, count in class_counts.items():
        percentage = count / total * 100
        print(f"  {str(sentiment).capitalize()}: {count:,} ({percentage:.1f}%)")
    return class_counts


def create_train_test_split(df, test_size=0.2, val_size=0.1):
    """Split data into train/validation/test using stratified sampling"""
    from sklearn.model_selection import train_test_split

    # Check if there are enough samples for each class to stratify
    class_counts = df["predicted_sentiment_label"].value_counts()
    if (class_counts < 2).any():
        print(
            "\nWARNING: Some classes have fewer than 2 samples. Cannot use stratification."
        )
        print("Splitting data without stratification.")
        stratify_option = None
    else:
        stratify_option = df["predicted_sentiment_label"]

    X = df.drop(columns=["predicted_sentiment_label"])
    y = df["predicted_sentiment_label"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=(test_size + val_size),
        stratify=stratify_option,
        random_state=42,
    )

    # Adjust stratify option for the second split
    if stratify_option is not None:
        stratify_option_temp = y_temp
        if (y_temp.value_counts() < 2).any():
            stratify_option_temp = None
    else:
        stratify_option_temp = None

    test_ratio = test_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=test_ratio,
        stratify=stratify_option_temp,
        random_state=42,
    )

    print("\nDataset split:")
    print(f"  Training: {len(X_train):,}")
    print(f"  Validation: {len(X_val):,}")
    print(f"  Test: {len(X_test):,}")

    return (X_train.join(y_train)), (X_val.join(y_val)), (X_test.join(y_test))


def save_processed_data(splits):
    """Save the processed data splits"""
    output_dir = PROJECT_PATH / "data/processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = splits

    train_df.to_csv(output_dir / "train_data.csv", index=False)
    val_df.to_csv(output_dir / "validation_data.csv", index=False)
    test_df.to_csv(output_dir / "test_data.csv", index=False)
    print("\nTrain/Val/Test splits saved separately.")


def balance_classes(df, label_column):
    """
    Downsample each class in the DataFrame to the size of the minority class to prevent class imbalance.
    Args:
        df (pd.DataFrame): The input DataFrame.
        label_column (str): The column name for class labels.
    Returns:
        pd.DataFrame: A balanced DataFrame with equal number of samples per class.
    """
    class_counts = df[label_column].value_counts()
    min_count = class_counts.min()
    balanced_df = (
        df.groupby(label_column, group_keys=False)
        .apply(lambda x: x.sample(n=min_count, random_state=42))
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    print(
        f"Balanced dataset to {min_count} samples per class (total {len(balanced_df)})"
    )
    return balanced_df


def main():
    """Main preprocessing pipeline"""
    print("=" * 60)
    print("DATA PREPROCESSING PIPELINE")
    print("=" * 60)

    # 1. Load, clean, and filter data
    df = load_data()
    df = remove_duplicates(df)
    df = apply_additional_cleaning(df)
    df = filter_text_length(df)

    # 2. Analyze the final class distribution (without balancing)
    analyze_class_distribution(df, "Final Distribution (Original Imbalance)")

    df = balance_classes(df, "predicted_sentiment_label")
    analyze_class_distribution(df, "Distribution After Balancing")

    # 4. Create train/validation/test splits
    splits = create_train_test_split(df)

    # 5. Save processed data
    save_processed_data(splits)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
