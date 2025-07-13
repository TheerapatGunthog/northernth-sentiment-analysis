import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path

PROJECT_PATH = Path("/home/whilebell/Code/northernth-sentiment-analysis/")
MODEL_PATH = PROJECT_PATH / "models/results/final_model/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

try:
    print("\nLoading and combining review data...")
    google_reviews = pd.read_csv(
        PROJECT_PATH / "data/raw/analyst_data/google_review.csv"
    )
    tripadvisor_reviews = pd.read_csv(
        PROJECT_PATH / "data/raw/analyst_data/tripadvisor_review.csv"
    )
    google_reviews["source"] = "Google"
    tripadvisor_reviews["source"] = "TripAdvisor"
    all_reviews_df = pd.concat([google_reviews, tripadvisor_reviews], ignore_index=True)
    all_reviews_df = all_reviews_df[["title", "text_review", "source"]].copy()
    all_reviews_df.dropna(subset=["text_review"], inplace=True)
    all_reviews_df["full_review"] = (
        all_reviews_df["title"].fillna("")
        + " "
        + all_reviews_df["text_review"].fillna("")
    )
    print(f"Successfully loaded and combined {len(all_reviews_df)} reviews.")
except FileNotFoundError as e:
    print(f"Error: Data file not found. {e}")
    exit()


def predict_sentiment(text):
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


print("\n--- Starting Sentiment Analysis ---")
all_reviews_df["predicted_sentiment"] = all_reviews_df["full_review"].apply(
    predict_sentiment
)

print("\n--- Analysis Complete ---")
print("Sample results from combined data:")
print(all_reviews_df[["source", "full_review", "predicted_sentiment"]].head(10))

print("\nSample results from Google Reviews:")
print(all_reviews_df[all_reviews_df["source"] == "Google"].head())

print("\nSample results from TripAdvisor Reviews:")
print(all_reviews_df[all_reviews_df["source"] == "TripAdvisor"].head())

selected_columns = ["source", "title", "text_review", "predicted_sentiment"]

all_reviews_df = all_reviews_df[selected_columns]

output_path = PROJECT_PATH / "data/analyst/sentiment_analysis_results.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
all_reviews_df.to_csv(output_path, index=False)
print(f"\nFull analysis results saved to: {output_path}")
