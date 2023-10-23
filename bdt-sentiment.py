import os
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, pipeline
from google.cloud import storage
import csv
from io import StringIO
def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")

def get_blob_data(bucket_name, blob_name):
    """Fetches the content of a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    data = blob.download_as_text()
    return data


def main():
    # Step 1: Download the source file from Google Cloud Storage
    BUCKET_NAME = 'tweets'
    BLOB_NAME  = 'source_file.csv'
    data = get_blob_data(BUCKET_NAME, BLOB_NAME)

    # Step 2: Use Hugging Face pre-trained transformer to get sentiments
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    correct_predictions = 0
    total_predictions = 0

    # Read the data from the string directly using StringIO
    file_like_object = StringIO(data)
    reader = csv.reader(file_like_object)
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