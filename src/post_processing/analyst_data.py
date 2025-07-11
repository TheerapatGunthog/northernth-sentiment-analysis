import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path

# --- Configuration ---
PROJECT_PATH = Path("/home/whilebell/Code/northernth-sentiment-analysis/")
# Path to the folder containing your downloaded model files
MODEL_PATH = PROJECT_PATH / "models/results/final_model/"

# --- Setup ---
# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Load Model and Tokenizer ---
try:
    print("Loading model and tokenizer...")
    # Load your fine-tuned model from the local path
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    # FIX: Load the tokenizer from the original 'roberta-base' checkpoint.
    # This avoids issues with missing local tokenizer files.
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    model.to(device)
    model.eval()  # Set the model to evaluation mode
    print("Model and tokenizer loaded successfully.")
except OSError:
    print(f"Error: Model or tokenizer not found at '{MODEL_PATH}'.")
    print(
        "Please make sure you have downloaded and unzipped the model from Kaggle into the correct directory."
    )
    exit()

# --- Load and Combine Data ---
try:
    print("\nLoading and combining review data...")
    google_reviews = pd.read_csv(
        PROJECT_PATH / "data/raw/analyst_data/google_review.csv"
    )
    tripadvisor_reviews = pd.read_csv(
        PROJECT_PATH / "data/raw/analyst_data/tripadvisor_review.csv"
    )

    # Add a 'source' column to identify the origin of each review
    google_reviews["source"] = "Google"
    tripadvisor_reviews["source"] = "TripAdvisor"

    # Combine both dataframes into one
    all_reviews_df = pd.concat([google_reviews, tripadvisor_reviews], ignore_index=True)

    # Clean up the DataFrame
    all_reviews_df = all_reviews_df[["title", "text_review", "source"]].copy()
    all_reviews_df.dropna(subset=["text_review"], inplace=True)

    # Create a single 'full_review' column for analysis
    # This is the raw text that will be fed to the model
    all_reviews_df["full_review"] = (
        all_reviews_df["title"].fillna("")
        + " "
        + all_reviews_df["text_review"].fillna("")
    )

    print(f"Successfully loaded and combined {len(all_reviews_df)} reviews.")

except FileNotFoundError as e:
    print(f"Error: Data file not found. {e}")
    exit()


# --- Prediction Function ---
def predict_sentiment(text):
    """
    Analyzes a single piece of raw text and returns the predicted sentiment label.
    """
    if not isinstance(text, str) or not text.strip():
        return "No Text"

    # Tokenize the text. The tokenizer handles everything needed for RoBERTa.
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=512
    )

    # Move tokenized inputs to the same device as the model
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Perform prediction
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model(**inputs)

    # Get the label with the highest probability
    predicted_class_id = torch.argmax(outputs.logits, dim=1).item()

    # Convert the label ID back to its string name (e.g., 'Positive')
    return model.config.id2label[predicted_class_id]


# --- Run Analysis ---
print("\n--- Starting Sentiment Analysis ---")
# Apply the prediction function to the 'full_review' column
all_reviews_df["predicted_sentiment"] = all_reviews_df["full_review"].apply(
    predict_sentiment
)

# --- Display Results ---
print("\n--- Analysis Complete ---")
print("Sample results from combined data:")
# Display a sample of the results, showing the source of the review
print(all_reviews_df[["source", "full_review", "predicted_sentiment"]].head(10))

# You can also view results from each source separately
print("\nSample results from Google Reviews:")
print(all_reviews_df[all_reviews_df["source"] == "Google"].head())

print("\nSample results from TripAdvisor Reviews:")
print(all_reviews_df[all_reviews_df["source"] == "TripAdvisor"].head())

# Optionally, save the full results to a new CSV file
output_path = PROJECT_PATH / "data/processed/sentiment_analysis_results.csv"
# Ensure the output directory exists
output_path.parent.mkdir(parents=True, exist_ok=True)
all_reviews_df.to_csv(output_path, index=False)
print(f"\nFull analysis results saved to: {output_path}")
