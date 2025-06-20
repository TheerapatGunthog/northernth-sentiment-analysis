import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset  # New import for Hugging Face Dataset objects

# Configuration
PROJECT_PATH = Path("/home/whilebell/Code/northernth-sentiment-analysis/")
# Define the pre-trained RoBERTa model to use
# You can choose other models like 'bert-base-uncased', 'xlm-roberta-base', etc.
ROBERTA_MODEL_NAME = "roberta-base"


def load_processed_data():
    """Load preprocessed data (from previous steps)"""
    data_path = PROJECT_PATH / "data/processed"

    train_df = pd.read_csv(data_path / "train_data.csv")
    val_df = pd.read_csv(data_path / "validation_data.csv")
    test_df = pd.read_csv(data_path / "test_data.csv")

    print("Loaded data:")
    print(f"  Train: {len(train_df):,} reviews")
    print(f"  Validation: {len(val_df):,} reviews")
    print(f"  Test: {len(test_df):,} reviews")

    return train_df, val_df, test_df


def prepare_labels(train_df, val_df, test_df):
    """Prepare labels for training using LabelEncoder"""

    # Create label encoder
    label_encoder = LabelEncoder()

    # Fit on all labels to ensure consistency across datasets
    all_labels = pd.concat(
        [
            train_df["sentiment_label"],
            val_df["sentiment_label"],
            test_df["sentiment_label"],
        ]
    )
    label_encoder.fit(all_labels)

    # Transform labels to numerical format
    y_train = label_encoder.transform(train_df["sentiment_label"])
    y_val = label_encoder.transform(val_df["sentiment_label"])
    y_test = label_encoder.transform(test_df["sentiment_label"])

    print("Label encoding mapping:")
    for i, label in enumerate(label_encoder.classes_):
        print(f"  {label}: {i}")

    return y_train, y_val, y_test, label_encoder


def tokenize_data(train_df, val_df, test_df, tokenizer_name=ROBERTA_MODEL_NAME):
    """Tokenize text data using Hugging Face's AutoTokenizer"""
    print(f"\nInitializing tokenizer for {tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize_function(examples):
        # Tokenize the 'cleaned_text' column, handle truncation and padding
        return tokenizer(
            examples["cleaned_text"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )  # RoBERTa's max length is typically 512

    print("Converting DataFrames to Hugging Face Datasets...")
    # Get numerical labels for direct assignment to Dataset objects
    # Pass all three dataframes to prepare_labels to ensure consistent encoding
    _, _, _, label_encoder = prepare_labels(train_df, val_df, test_df)
    train_labels = label_encoder.transform(train_df["sentiment_label"])
    val_labels = label_encoder.transform(val_df["sentiment_label"])
    test_labels = label_encoder.transform(test_df["sentiment_label"])

    # Create Hugging Face Dataset objects with numerical labels
    train_dataset = Dataset.from_pandas(train_df.assign(labels=train_labels))
    val_dataset = Dataset.from_pandas(val_df.assign(labels=val_labels))
    test_dataset = Dataset.from_pandas(test_df.assign(labels=test_labels))

    print("Tokenizing train, validation, and test datasets...")
    # Apply tokenization using map function with batched processing for speed
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    # Remove original text and length columns as they are no longer needed after tokenization
    # REMOVED '__index_level_0__' FROM HERE
    tokenized_train = tokenized_train.remove_columns(
        ["cleaned_text", "text_length", "word_count", "sentiment_label"]
    )
    tokenized_val = tokenized_val.remove_columns(
        ["cleaned_text", "text_length", "word_count", "sentiment_label"]
    )
    tokenized_test = tokenized_test.remove_columns(
        ["cleaned_text", "text_length", "word_count", "sentiment_label"]
    )

    # Rename 'labels' column to 'label' for compatibility with Hugging Face Trainer API
    tokenized_train = tokenized_train.rename_column("labels", "label")
    tokenized_val = tokenized_val.rename_column("labels", "label")
    tokenized_test = tokenized_test.rename_column("labels", "label")

    print(f"Tokenized train dataset features: {tokenized_train.column_names}")
    print(f"Tokenized validation dataset features: {tokenized_val.column_names}")
    print(f"Tokenized test dataset features: {tokenized_test.column_names}")

    return tokenized_train, tokenized_val, tokenized_test, tokenizer


def save_processed_data_for_roberta(
    tokenized_train, tokenized_val, tokenized_test, tokenizer, label_encoder
):
    """Save tokenized datasets and tokenizer/label_encoder objects"""

    output_dir = PROJECT_PATH / "data/roberta_processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save tokenized datasets to disk (in Arrow format)
    tokenized_train.save_to_disk(output_dir / "train_tokenized_dataset")
    tokenized_val.save_to_disk(output_dir / "val_tokenized_dataset")
    tokenized_test.save_to_disk(output_dir / "test_tokenized_dataset")

    # Save the tokenizer and label encoder for future use (e.g., inference)
    tokenizer.save_pretrained(output_dir / "tokenizer")
    with open(output_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    print(f"Tokenized datasets and related objects saved to: {output_dir}")


def main():
    """Main data preparation pipeline for RoBERTa model training"""
    print("=" * 60)
    print("DATA PREPARATION PIPELINE FOR ROBERTA")
    print("=" * 60)

    # 1. Load processed data (from the previous preprocessing step)
    train_df, val_df, test_df = load_processed_data()

    # 2. Prepare labels (encode string labels to numerical IDs)
    # This also returns the label_encoder object for later decoding
    y_train, y_val, y_test, label_encoder = prepare_labels(train_df, val_df, test_df)

    # 3. Tokenize text data for RoBERTa using its specific tokenizer
    # This step converts raw text into input_ids, attention_mask, etc.
    tokenized_train, tokenized_val, tokenized_test, tokenizer = tokenize_data(
        train_df, val_df, test_df, ROBERTA_MODEL_NAME
    )

    # 4. Save the prepared data (tokenized datasets, tokenizer, label encoder)
    # These files are now ready to be loaded by a Hugging Face Trainer for model training.
    save_processed_data_for_roberta(
        tokenized_train, tokenized_val, tokenized_test, tokenizer, label_encoder
    )

    # No need for manual feature engineering steps like `analyze_feature_importance()` or `create_word_clouds()`
    # at this stage, as RoBERTa handles its own feature extraction internally.
    # Word clouds can still be performed on the original text data for EDA purposes, but not as input preparation.

    # 5. Final summary of the preparation process
    print("\n" + "=" * 60)
    print("ROBERTA DATA PREPARATION COMPLETE!")
    print("=" * 60)

    print("\nDataset sizes (tokenized):")
    print(f"  Training: {len(tokenized_train):,} samples")
    print(f"  Validation: {len(tokenized_val):,} samples")
    print(f"  Test: {len(tokenized_test):,} samples")

    print(f"\nFiles saved in: {PROJECT_PATH}/data/roberta_processed/")
    print("- train_tokenized_dataset/ (folder containing Arrow files)")
    print("- val_tokenized_dataset/ (folder containing Arrow files)")
    print("- test_tokenized_dataset/ (folder containing Arrow files)")
    print("- tokenizer/ (folder containing tokenizer configuration files)")
    print("- label_encoder.pkl (Python pickle file for label mapping)")

    print("\n✅ Ready to train RoBERTa model using Hugging Face Trainer!")


if __name__ == "__main__":
    main()
