import re
from transformers import pipeline

# Data Input
text = """マリオシリーズが大大大好きで、新作が出るたびもう何十年もやっていますが、こちらはいつもより難易度が低いように感じました。

私のような大人のやり込み系プレイヤーには難易度の面では少し物足りなさもあるかもしれませんが、子どもがプレイしたり子供と一緒に遊んだり、また、マリオをあまりやったことがない人はとっつきやすく、とても楽しめる作品だと思います！

新しい要素のワンダーシードも楽しかったです。"""

# Data Preprocess
def preprocess_text(text):

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)

    return text

cleaned_text = preprocess_text(text)

# Model Deployment
classifier = pipeline("sentiment-analysis", model="koheiduck/bert-japanese-finetuned-sentiment")
output = classifier(cleaned_text)

# Display result
print(output)