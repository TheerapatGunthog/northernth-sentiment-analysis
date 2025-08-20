import pandas as pd
import mysql.connector
from pathlib import Path
from tqdm import tqdm
import os
from dotenv import load_dotenv

# --- 1. CONFIGURATION ---
# Load Environment Variables from .env file
load_dotenv()

# Set the project's Path
PROJECT_PATH = Path("/home/whilebell/Code/northernth-sentiment-analysis/")
NORMALIZED_DATA_PATH = PROJECT_PATH / "data/analyst/sentiment_analysis_results.csv"

# Configure database connection from Environment Variables
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
}


def normalize_titles(df: pd.DataFrame) -> pd.DataFrame:
    """Function to normalize attraction names"""
    print("Normalizing attraction titles...")
    normalized_title_map = {
        "Bua Thong Waterfalls (Nam Phu Chet Si)": "Buatong Waterfall-Chet Si Fountain National Park",
        "Saturday Night Market  Walking Street - Wua Lai Road": "Wua Lai Walking Street Saturday Market",
        "Grand Canyon Chiang Mai": "Grand Canyon Water Park",
        "Maeklang Elephant Conservation Community": "Thai Elephant Conservation Center",
        "Umbrella Village - Bor Sang": "Bo Sang Umbrella Village",
        "Wat Phra Singh": "Wat Phra Singh Woramahawihan",
        "Lanna Folklife Museum": "Lanna Folklife Centre",
        "Kad Rin Come Night Market": "Rincome night market",
        "Sunday Night Market": "Chiang Mai Night Market",
        "Chang Phuak Gate Night Market": "Chang Phuak Market",
        "Wat Chedi Luang Varavihara": "Watchediluang Varaviharn",
        "Doi Inthanon": "Doi Inthanon National Park",
        "Tiger Kingdom - Chiang Mai": "Tiger Kingdom",
    }
    df["title"] = df["title"].replace(normalized_title_map)
    print("Title normalization complete.")
    return df


def migrate_data():
    """Main script for migrating data into the database"""
    db_connection = None
    try:
        # --- 2. LOAD & PREPARE DATA ---
        print("Loading analysis results data...")
        df = pd.read_csv(PROJECT_PATH / "data/analyst/sentiment_analysis_results.csv")
        df = normalize_titles(df)

        # --- 3. DATABASE CONNECTION ---
        print("Connecting to the database...")
        db_connection = mysql.connector.connect(**DB_CONFIG)
        cursor = db_connection.cursor()
        print("Database connection successful.")

        # --- 4. PROCESS ATTRACTIONS ---
        print("\n--- Processing Attractions ---")

        # Fetch existing attractions from the DB to check
        cursor.execute("SELECT id, title FROM attractions")
        existing_attractions = {title: id for id, title in cursor.fetchall()}

        # Find new attractions that are not yet in the DB
        unique_titles_from_file = set(df["title"].unique())
        new_titles = unique_titles_from_file - set(existing_attractions.keys())

        if new_titles:
            print(f"Found {len(new_titles)} new attractions to insert.")
            new_attractions_data = [(title,) for title in new_titles]

            # Use executemany to insert all data at once (much faster)
            insert_attraction_query = "INSERT INTO attractions (title) VALUES (%s)"
            cursor.executemany(insert_attraction_query, new_attractions_data)
            db_connection.commit()
            print(f"Successfully inserted {cursor.rowcount} new attractions.")

            # Update the list of attractions after adding new ones
            cursor.execute("SELECT id, title FROM attractions")
            existing_attractions = {title: id for id, title in cursor.fetchall()}
        else:
            print("No new attractions to add.")

        # --- 5. PROCESS REVIEWS ---
        print("\n--- Processing Reviews ---")

        # Prepare all review data for Bulk Insert
        reviews_to_insert = []
        for index, row in tqdm(
            df.iterrows(), total=df.shape[0], desc="Preparing reviews"
        ):
            attraction_id = existing_attractions.get(row["title"])
            if attraction_id:
                reviews_to_insert.append(
                    (
                        attraction_id,
                        row["source"],
                        row["text_review"],
                        row["predicted_sentiment"],
                    )
                )

        if reviews_to_insert:
            # Clear all old reviews to start fresh (TRUNCATE is fast and avoids Safe Mode)
            print("Clearing old reviews...")
            cursor.execute("TRUNCATE TABLE reviews")

            # Bulk Insert all reviews
            print(f"Inserting {len(reviews_to_insert)} new reviews...")
            insert_review_query = """
                INSERT INTO reviews (attraction_id, source, text_review, predicted_sentiment)
                VALUES (%s, %s, %s, %s)
            """
            cursor.executemany(insert_review_query, reviews_to_insert)
            db_connection.commit()
            print(f"Successfully inserted {cursor.rowcount} reviews.")
        else:
            print("No reviews to insert.")

        # --- 6. UPDATE SENTIMENT STATS ---
        print("\n--- Updating Sentiment Statistics ---")

        # Clear all old statistics
        cursor.execute("TRUNCATE TABLE sentiment_stats")

        # Call Stored Procedure to calculate new statistics for all attractions
        attraction_ids = existing_attractions.values()
        for att_id in tqdm(attraction_ids, desc="Updating stats"):
            cursor.callproc("UpdateSentimentStats", [att_id])

        db_connection.commit()
        print("Sentiment statistics updated for all attractions.")

        print("\n✅ Migration process completed successfully!")

    except mysql.connector.Error as err:
        print(f"Database Error: {err}")
    except FileNotFoundError as err:
        print(f"File Error: {err}")
    except Exception as err:
        print(f"An unexpected error occurred: {err}")
    finally:
        if db_connection and db_connection.is_connected():
            cursor.close()
            db_connection.close()
            print("\nDatabase connection closed.")


if __name__ == "__main__":
    migrate_data()
