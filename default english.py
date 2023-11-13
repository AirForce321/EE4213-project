import re
from transformers import pipeline

# Data Input
text = """I purchased this box of batteries last September 2022. Today, April 2, 2023, I tried to use them and none of them work. The mfg claims it will last 10 years in storage! Well, sad to let Energizer know it was NOT the case here. Didn't even last one full year. Product didn't keep up with the written guarantee mentioned under the description that will last 10 years in storage. Don't throw your hard earned dollar away buying in bulk. Unfortunately, I cannot return them to Amazon, as time has passed. And doubt mfg will do anything about it"""

# Data Preprocess
def preprocess_text(text):

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)

    return text

cleaned_text = preprocess_text(text)

# Model Deployment
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
output = classifier(cleaned_text)

# Display result
print(output)
