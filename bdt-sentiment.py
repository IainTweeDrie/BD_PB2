import os
from transformers import pipeline
from google.cloud import storage
import csv


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")


def main():
    # Step 1: Download the source file from Google Cloud Storage
    BUCKET_NAME = 'tweets'
    SOURCE_BLOB_NAME = 'source_file.csv'
    DESTINATION_FILE_NAME = 'local_tweets.csv'
    download_blob(BUCKET_NAME, SOURCE_BLOB_NAME, DESTINATION_FILE_NAME)

    # Step 2: Use Hugging Face pre-trained transformer to get sentiments
    nlp = pipeline("sentiment-analysis")

    correct_predictions = 0
    total_predictions = 0

    # Export to CSV
    with open(DESTINATION_FILE_NAME, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            tweet, ground_truth = row
            result = nlp(tweet)
            predicted_sentiment = result[0]['label']
            if predicted_sentiment == ground_truth:
                correct_predictions += 1
            total_predictions += 1

    # Step 3: Calculate accuracy
    accuracy = (correct_predictions / total_predictions) * 100
    print(f"Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()