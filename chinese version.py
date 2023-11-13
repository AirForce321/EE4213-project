import re
from transformers import pipeline

# Data Input
text = """从伦敦漂洋过海就一个牛皮纸袋子，里面的望远镜盒子都压扁了，好歹弄个纸箱子呀。"""

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