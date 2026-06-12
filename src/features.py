import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def engineer_features(df):
    print("Engineering features...")
    analyzer = SentimentIntensityAnalyzer()

    # 1. VADER sentiment score (-1 to +1) on review text
    print("  Running VADER sentiment analysis... (takes 1-2 min)")
    df['sentiment_score'] = df['review_text'].apply(
        lambda x: analyzer.polarity_scores(str(x))['compound']
    )

    # 2. Normalize rating to same -1 to +1 scale as sentiment
    df['rating_normalized'] = (df['rating'] - 3) / 2

    # 3. THE KEY FEATURE: mismatch between sentiment and rating
    # Positive = review text is more negative than star rating suggests
    # Negative = review text is more positive than star rating suggests
    df['mismatch_score'] = df['rating_normalized'] - df['sentiment_score']

    # 4. Review length — longer reviews often signal problems
    df['review_length'] = df['review_text'].apply(lambda x: len(str(x).split()))

    # 5. Has exclamation mark — correlates with fake positivity
    df['has_exclamation'] = df['review_text'].apply(
        lambda x: int('!' in str(x))
    )

    # 6. Recommend vs rating mismatch
    # Someone who says do_recommend=False but gives 4-5 stars is a strong signal
    df['return_risk'] = (
    (df['helpful_votes'] == 0) & 
    (df['rating'] >= 4) & 
    (df['sentiment_score'] < 0.5)
).astype(int)

    # 7. PROXY LABEL: return risk
    # High star rating + zero helpful votes = potentially misleading review
    # This is our best approximation without actual return data
    df['return_risk'] = (
    (df['mismatch_score'] > 0.3) & (df['rating'] >= 4)
).astype(int)

    print(f"  Mismatch score stats:\n{df['mismatch_score'].describe().round(3)}")
    print(f"\n  Return risk label distribution:\n{df['return_risk'].value_counts()}")
    print(f"\n  Features engineered: {df.shape[1]} columns total")

    df.to_csv("data/electronics_features.csv", index=False)
    print("\nSaved to data/electronics_features.csv")

    return df

if __name__ == "__main__":
    df = pd.read_csv("data/electronics_clean.csv")
    df = engineer_features(df)
    print("\nSample of key features:")
    print(df[['rating', 'sentiment_score', 'rating_normalized',
              'mismatch_score', 'review_length', 'return_risk']].head(10).round(3))