import pandas as pd
import numpy as np
import re
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

# Data Preprocess
def preprocess_text(text):
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)
    return text

# Model Deployment
MODEL = f"LiYuan/amazon-review-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Maximum sequence length
max_seq_length = tokenizer.model_max_length

def polarity_scores(text):
    processed_text = preprocess_text(text)

    # Tokenize and truncate input
    inputs = tokenizer.encode_plus(
        processed_text,
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        max_length=max_seq_length,
        return_tensors='pt'
    )

    # Perform sentiment analysis
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = softmax(logits, axis=1).tolist()[0]

    scores_dict = {f'{i} star': score for i, score in enumerate(probabilities, start=1)}
    return scores_dict

# Read CSV file
data = pd.read_csv('amazon_review.csv')

# Add sentiment analysis results to each review
results = []
for index, row in data.iterrows():
    review = row['Review']
    scores = polarity_scores(review)
    result = {
        'Review': review,
        'Sentiment Scores': scores
    }
    results.append(result)

# Display results
for result in results:
    print(result)