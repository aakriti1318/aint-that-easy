import pandas as pd
import numpy as np

# # Create sample data
# reviews = [
#     "This product is amazing. I love how well it works!",
#     "Not bad, but could be better. The price is a bit high for what you get.",
#     "Terrible experience. Never buying again. The quality is just awful.",
#     "The quality is good, but the price is high. I'm on the fence about recommending it.",
#     "Absolutely fantastic service and product. Customer support was also excellent.",
#     "Disappointed with the quality. It started falling apart after just a week.",
#     "Average product, nothing special. Does what it says but nothing more.",
#     "So happy with my purchase! Everything about it exceeds expectations.",
#     "Waste of money. Don't recommend. Save your cash for something better.",
#     "It's okay, I guess. Neither great nor terrible, just mediocre.",
#     "I really like this product, it has improved my daily routine significantly.",
#     "The interface is confusing and not intuitive. Hard to figure out how to use it.",
#     "Decent value for the money, but there are better options available.",
#     "This exceeded all my expectations! Will definitely buy from this company again.",
#     "Shipping was fast but the product quality is questionable. Mixed feelings overall.",
#     "Not worth the hype. Pretty basic functionality for the premium price.",
#     "Great customer service when I had an issue. They resolved it quickly.",
#     "The product broke after two months of light use. Very disappointed.",
#     "Perfect for what I needed! Simple to use and works exactly as described.",
#     "The design is sleek but functionality is lacking. Form over function."
# ]

import pandas as pd

# Create a balanced dataset with clear examples
reviews = [
    # Negative examples
    "This product is terrible and broke within a week.",
    "Worst purchase I've ever made. Complete waste of money.",
    "The customer service was rude and unhelpful.",
    "Do not buy this! It's poorly made and overpriced.",
    "I regret purchasing this item. It doesn't work as advertised.",
    # Neutral examples
    "The product is okay. Nothing special but it works.",
    "Average quality for the price. Does the job adequately.",
    "It has some good features and some drawbacks.",
    "It's exactly what I expected, neither impressive nor disappointing.",
    "Decent product but there are better alternatives available.",
    # Positive examples
    "Excellent product! Exceeded all my expectations.",
    "I love this! Best purchase I've made all year.",
    "Great value for money and high quality construction.",
    "Highly recommend this product. Works perfectly.",
    "Outstanding performance and customer service was excellent."
]

# Assign clear sentiment labels: 0=Negative, 1=Neutral, 2=Positive
sentiments = [
    0, 0, 0, 0, 0,  # Negative examples
    1, 1, 1, 1, 1,  # Neutral examples
    2, 2, 2, 2, 2   # Positive examples
]

# Create DataFrame
df = pd.DataFrame({
    'review_text': reviews,
    'sentiment': sentiments
})

# Save to CSV
df.to_csv('balanced_reviews.csv', index=False)
# Assign sentiments: 0 = Negative, 1 = Neutral, 2 = Positive
# sentiments = [
#     2, 1, 0, 1, 2, 
#     0, 1, 2, 0, 1,
#     2, 0, 1, 2, 1,
#     0, 2, 0, 2, 1
# ]

# # Create DataFrame
# df = pd.DataFrame({
#     'review_text': reviews,
#     'sentiment': sentiments
# })

# # Save to CSV
# df.to_csv('sample_reviews.csv', index=False)

# print("Sample dataset created and saved as 'sample_reviews.csv'")
# print(f"Dataset contains {len(df)} reviews with the following sentiment distribution:")
# print(df['sentiment'].value_counts().sort_index())