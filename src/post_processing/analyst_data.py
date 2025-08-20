import pandas as pd
import torch
import mysql.connector
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path
from tqdm import tqdm
import os
from dotenv import load_dotenv

# --- 1. CONFIGURATION ---
load_dotenv()

PROJECT_PATH = Path("/home/whilebell/Code/northernth-sentiment-analysis/")
MODEL_PATH = PROJECT_PATH / "models/results/final_model/"
OUTPUT_PATH = PROJECT_PATH / "data/analyst/sentiment_analysis_results.csv"

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def get_existing_reviews_from_db():
    """Fetch existing reviews from the database to avoid duplication."""
    print("\nConnecting to database to fetch existing reviews...")
    existing_reviews = set()
    db_connection = None
    try:
        db_connection = mysql.connector.connect(**DB_CONFIG)
        cursor = db_connection.cursor(dictionary=True)
        # Use text_review and source as a pair to identify unique reviews
        cursor.execute("SELECT text_review, source FROM reviews")
        for row in cursor.fetchall():
            existing_reviews.add((row["text_review"], row["source"]))
        print(f"Found {len(existing_reviews)} existing reviews in the database.")
        return existing_reviews
    except mysql.connector.Error as err:
        print(f"Database Error: {err}")
        return None  # Return None if unable to connect
    finally:
        if db_connection and db_connection.is_connected():
            cursor.close()
            db_connection.close()
            print("Database connection closed.")


def predict_sentiment(text, tokenizer, model):
    """Function to predict sentiment."""
    if not isinstance(text, str) or not text.strip():
        return "No Text"
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=512
    )
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_class_id = torch.argmax(outputs.logits, dim=1).item()
    return model.config.id2label[predicted_class_id]


def main():
    # --- 2. LOAD RAW DATA ---
    try:
        print("\nLoading and combining raw review data...")
        google_reviews = pd.read_csv(
            PROJECT_PATH / "data/raw/analyst_data/google_review.csv"
        )
        tripadvisor_reviews = pd.read_csv(
            PROJECT_PATH / "data/raw/analyst_data/tripadvisor_review.csv"
        )
        google_reviews["source"] = "Google Maps"
        tripadvisor_reviews["source"] = "TripAdvisor"
        all_reviews_df = pd.concat(
            [google_reviews, tripadvisor_reviews], ignore_index=True
        )
        all_reviews_df = all_reviews_df[["title", "text_review", "source"]].copy()
        all_reviews_df.dropna(subset=["text_review"], inplace=True)
        print(f"Successfully loaded {len(all_reviews_df)} total raw reviews.")
    except FileNotFoundError as e:
        print(f"Error: Raw data file not found. {e}")
        exit()

    # --- 3. FILTER FOR NEW REVIEWS ONLY ---
    existing_reviews_set = get_existing_reviews_from_db()
    if existing_reviews_set is None:
        print("Could not connect to DB to check for existing reviews. Exiting.")
        return

    # Create a temporary column for checking
    all_reviews_df["temp_key"] = all_reviews_df.apply(
        lambda row: (row["text_review"], row["source"]), axis=1
    )

    # Filter only reviews not yet in the database
    new_reviews_df = all_reviews_df[
        ~all_reviews_df["temp_key"].isin(existing_reviews_set)
    ].copy()
    new_reviews_df.drop(columns=["temp_key"], inplace=True)

    if new_reviews_df.empty:
        print("\nNo new reviews to analyze. All data is up-to-date. ✅")
        # Even if there are no new reviews, we might still want an empty results file for the migration script to work
        pd.DataFrame(
            columns=["source", "title", "text_review", "predicted_sentiment"]
        ).to_csv(OUTPUT_PATH, index=False)
        print(f"Empty results file created at: {OUTPUT_PATH}")
        return

    print(f"\nFound {len(new_reviews_df)} new reviews to analyze.")

    # --- 4. LOAD MODEL & RUN ANALYSIS (ONLY ON NEW DATA) ---
    try:
        print("Loading model and tokenizer...")
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        model.to(device)
        model.eval()
        print("Model and tokenizer loaded successfully.")
    except OSError:
        print(f"Error: Model or tokenizer not found at '{MODEL_PATH}'.")
        exit()

    new_reviews_df["full_review"] = (
        new_reviews_df["title"].fillna("")
        + " "
        + new_reviews_df["text_review"].fillna("")
    )

    print("\n--- Starting Sentiment Analysis on new reviews ---")
    tqdm.pandas(desc="Analyzing sentiments")
    new_reviews_df["predicted_sentiment"] = new_reviews_df[
        "full_review"
    ].progress_apply(lambda text: predict_sentiment(text, tokenizer, model))

    # --- 5. SAVE RESULTS ---
    print("\n--- Analysis Complete ---")
    selected_columns = ["source", "title", "text_review", "predicted_sentiment"]
    final_df = new_reviews_df[selected_columns]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nAnalysis results for {len(final_df)} new reviews saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
