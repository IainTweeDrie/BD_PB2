from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from google.cloud import storage

def fetch_tweets_from_gcs(bucket_name, file_name):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = storage.Blob(file_name, bucket)
    content = blob.download_as_text()
    tweets = content.splitlines()
    return tweets


def main():
    BUCKET_NAME = 'GCS-BUCKET-NAME'
    FILE_NAME = 'STORED-TWEETS-FILE-NAME'

    tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    sentiment_analysis_pipeline = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

    tweets_with_ground_truth = fetch_tweets_from_gcs(BUCKET_NAME, FILE_NAME)

    correct_predictions = 0

    for tweet_with_ground_truth in tweets_with_ground_truth:
        tweet, ground_truth = tweet_with_ground_truth.rsplit(',', 1)
        ground_truth = ground_truth.strip()

        result = sentiment_analysis_pipeline(tweet)[0]
        predicted_sentiment = result['label'].lower()

        if predicted_sentiment == ground_truth:
            correct_predictions += 1

    accuracy = correct_predictions / len(tweets_with_ground_truth)
    print(f'Accuracy of the pre-trained model: {accuracy * 100:.2f}%')


if __name__ == "__main__":
    main()