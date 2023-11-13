import re
from transformers import pipeline

# Data Input
text = """I always head to this brand for the dependability and these arrived fresh and ready to go...

...and...

...still going!

I mean they last for quite some time, I have these in everything from my mice zapper to my under-the-counter lighting. This brand seems to last longer than anything else out there and I'm always satisfied with my purchase of the energizers."""

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
