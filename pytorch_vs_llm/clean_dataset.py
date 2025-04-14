import pandas as pd
import numpy as np

# Load your CSV file
df = pd.read_csv('amazon_reviews.csv')

# Display information about the dataset
print(f"Dataset has {len(df)} rows and {len(df.columns)} columns")
print("\nColumn names:")
print(df.columns.tolist())

# Check for missing values in reviewText
print(f"\nMissing values in reviewText: {df['reviewText'].isna().sum()}")

# Fill missing values in reviewText with empty string
df['reviewText'] = df['reviewText'].fillna("").astype(str)

# Convert score_pos_neg_diff to sentiment categories
# You might need to adjust these thresholds based on your data
def map_score_to_sentiment(score):
    if score < 0:
        return 0  # Negative
    elif score == 0:
        return 1  # Neutral
    else:
        return 2  # Positive

# Create a new column for sentiment
df['sentiment'] = df['score_pos_neg_diff'].apply(map_score_to_sentiment)

# Check the distribution of sentiments
print("\nSentiment distribution:")
print(df['sentiment'].value_counts())

# If the distribution is very imbalanced, we can balance it
# by subsampling the majority class or oversampling the minority classes

# Get counts
sentiment_counts = df['sentiment'].value_counts()
min_count = sentiment_counts.min()
balanced_df = pd.DataFrame()

# Balance the dataset by taking an equal number of samples from each class
for sentiment in [0, 1, 2]:
    sentiment_data = df[df['sentiment'] == sentiment]
    # If there are more samples than the minimum count, downsample
    if len(sentiment_data) > min_count:
        sentiment_data = sentiment_data.sample(min_count, random_state=42)
    balanced_df = pd.concat([balanced_df, sentiment_data])

# Shuffle the balanced dataset
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nBalanced dataset has {len(balanced_df)} rows")
print("\nBalanced sentiment distribution:")
print(balanced_df['sentiment'].value_counts())

# Save the processed dataset
balanced_df[['reviewText', 'sentiment']].to_csv('balanced_amazon_reviews.csv', index=False)
print("\nBalanced dataset saved as 'balanced_amazon_reviews.csv'")