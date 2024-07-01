import nltk
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
def analyze_sentiment_textblob(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'

def analyze_sentiment_vader(text):
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(text)
    if score['compound'] >= 0.05:
        return 'Positive'
    elif score['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'
sample_texts = [
    "The food was amazing and the service was excellent!",
    "I didn't enjoy the meal. It was too salty.",
    "It was an okay experience, nothing special.",
    "Absolutely fantastic! Highly recommended.",
    "Terrible service and bland food."
]

for text in sample_texts:
    print(f"Text: {text}")
    print(f"TextBlob Sentiment: {analyze_sentiment_textblob(text)}")
    print(f"VADER Sentiment: {analyze_sentiment_vader(text)}")
    print("-----")
import pandas as pd

# Load your dataset (assuming you have a CSV file with a column named 'review')
df = pd.read_csv('restaurant_reviews.csv')

# Apply the sentiment analysis functions to each review
df['TextBlob_Sentiment'] = df['review'].apply(analyze_sentiment_textblob)
df['VADER_Sentiment'] = df['review'].apply(analyze_sentiment_vader)

# Display the results
print(df.head())
