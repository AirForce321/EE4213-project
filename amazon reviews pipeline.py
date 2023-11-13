import re
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

text = """垃圾亚马逊，发票到现在都收不到，中间还少发卡，垃圾。"""
cleaned_text = preprocess_text(text)

# Model Deployment
MODEL = f"LiYuan/amazon-review-sentiment-analysis"
tokenizers = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

encoded_text = tokenizers(cleaned_text, return_tensors='pt')
output = model(**encoded_text)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
scores_dict = {
    '1 star': scores[0],
    '2 stars': scores[1],
    '3 stars': scores[2],
    '4 stars': scores[3],
    '5 stars': scores[4]
}

# Display result
print(scores_dict)