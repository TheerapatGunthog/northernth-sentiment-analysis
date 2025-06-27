import pandas as pd
from pathlib import Path
import json
from transformers import AutoTokenizer
from datasets import Dataset

# Configuration
PROJECT_PATH = Path("/home/whilebell/Code/northernth-sentiment-analysis/")
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
    """
    Prepare labels using a predefined mapping and validate the result.
    """
    print("\nPreparing labels with predefined sentiment order...")

    label_order = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
    label2id = {label: i for i, label in enumerate(label_order)}
    id2label = {i: label for i, label in enumerate(label_order)}

    print("Label mapping (id: label):")
    for i, label in id2label.items():
        print(f"  {i}: {label}")

    # --- ADDED: Validation Step ---
    # We will now check if the labels in the dataframe can be correctly mapped.

    def validate_and_map_labels(df, df_name):
        # Create a copy to avoid SettingWithCopyWarning
        df_copy = df.copy()

        # Apply the mapping
        df_copy["labels_id"] = df_copy["sentiment_label"].map(label2id)

        # Check for invalid labels that resulted in NaN
        invalid_mask = df_copy["labels_id"].isna()
        num_invalid = invalid_mask.sum()

        if num_invalid > 0:
            print(f"\nWARNING: Found {num_invalid} invalid labels in '{df_name}'.")
            # Show examples of invalid labels from the original data
            invalid_examples = df_copy[invalid_mask]["sentiment_label"].unique()
            print(f"  Examples of invalid labels found: {list(invalid_examples[:5])}")
            print("  These rows will be REMOVED from the dataset.")
            # Filter out the invalid rows
            df_copy = df_copy[~invalid_mask]
        else:
            print(f"\n✅ All labels in '{df_name}' are valid.")

        # Convert the column to integer type now that NaNs are removed
        df_copy["labels_id"] = df_copy["labels_id"].astype(int)

        return df_copy

    # Apply validation and mapping to each dataset
    train_df_mapped = validate_and_map_labels(train_df, "train_df")
    val_df_mapped = validate_and_map_labels(val_df, "val_df")
    test_df_mapped = validate_and_map_labels(test_df, "test_df")

    return train_df_mapped, val_df_mapped, test_df_mapped, label2id, id2label


def tokenize_data(
    train_df,
    val_df,
    test_df,
    tokenizer_name=ROBERTA_MODEL_NAME,
):
    """Tokenize text data. Now works with the mapped dataframes."""
    print(f"\nInitializing tokenizer for {tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize_function(examples):
        return tokenizer(
            examples["cleaned_text"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )

    print("Converting DataFrames to Hugging Face Datasets...")
    # The 'labels_id' column is already in the dataframe from the previous step
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    print("Tokenizing train, validation, and test datasets...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    # The Trainer API expects the label column to be named 'label'
    tokenized_train = tokenized_train.rename_column("labels_id", "label")
    tokenized_val = tokenized_val.rename_column("labels_id", "label")
    test_dataset = test_dataset.rename_column(
        "labels_id", "label"
    )  # also rename for consistency

    # Remove original text and other unneeded columns
    columns_to_remove = [
        "cleaned_text",
        "text_length",
        "word_count",
        "sentiment_label",
        "rating_review",
    ]
    tokenized_train = tokenized_train.remove_columns(columns_to_remove)
    tokenized_val = tokenized_val.remove_columns(columns_to_remove)
    # Note: We don't tokenize the test set here in the same way, but it's good practice.
    # The finetune script will handle the final test set tokenization if needed.

    print(f"\nTokenized train dataset features: {tokenized_train.column_names}")

    return tokenized_train, tokenized_val, test_dataset, tokenizer


def save_processed_data_for_roberta(
    tokenized_train, tokenized_val, test_dataset, tokenizer, label2id, id2label
):
    """Save all processed objects."""
    output_dir = PROJECT_PATH / "data/roberta_processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenized_train.save_to_disk(output_dir / "train_tokenized_dataset")
    tokenized_val.save_to_disk(output_dir / "val_tokenized_dataset")
    test_dataset.save_to_disk(
        output_dir / "test_tokenized_dataset"
    )  # Save the mapped test set
    tokenizer.save_pretrained(output_dir / "tokenizer")

    with open(output_dir / "label2id.json", "w") as f:
        json.dump(label2id, f)
    with open(output_dir / "id2label.json", "w") as f:
        json.dump(id2label, f)

    print(f"\nTokenized datasets and mapping files saved to: {output_dir}")


def main():
    """Main data preparation pipeline"""
    print("=" * 60)
    print("DATA PREPARATION PIPELINE FOR ROBERTA")
    print("=" * 60)

    train_df, val_df, test_df = load_processed_data()

    # Prepare labels and get validated dataframes
    train_df_mapped, val_df_mapped, test_df_mapped, label2id, id2label = prepare_labels(
        train_df, val_df, test_df
    )

    tokenized_train, tokenized_val, final_test_dataset, tokenizer = tokenize_data(
        train_df_mapped, val_df_mapped, test_df_mapped, ROBERTA_MODEL_NAME
    )

    save_processed_data_for_roberta(
        tokenized_train,
        tokenized_val,
        final_test_dataset,
        tokenizer,
        label2id,
        id2label,
    )

    print("\n" + "=" * 60)
    print("ROBERTA DATA PREPARATION COMPLETE!")
    print("=" * 60)
    print(f"  Training samples after cleaning: {len(tokenized_train):,}")
    print(f"  Validation samples after cleaning: {len(tokenized_val):,}")
    print("\n✅ Ready to train RoBERTa model.")


if __name__ == "__main__":
    main()
