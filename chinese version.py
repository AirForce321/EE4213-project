import re
from transformers import pipeline

# Data Input
text = """已经吃了很多年了，这颜色还是第一次，眼睛需要的重要营养补给，绝对有好处。"""

# Data Preprocess
def preprocess_text(text):

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)

    return text

cleaned_text = preprocess_text(text)

# Model Deployment
classifier = pipeline("sentiment-analysis", model="Ayazhankad/bert-finetuned-semantic-chinese")
output = classifier(cleaned_text)

# Display result
print(output)