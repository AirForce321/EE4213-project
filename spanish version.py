import re
from transformers import pipeline

# Data Input
text = """El producto está muy bien, pero no duran las pilas más de un mes, aparte a las dos semana cuando se van gastando las pilas, no suena para buscar el mando, por lo que pierde el atractivo que tiene, espero que con el próximo modelo lo solucionen."""

# Data Preprocess
def preprocess_text(text):

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)

    return text

cleaned_text = preprocess_text(text)

# Model Deployment
classifier = pipeline("sentiment-analysis", model="maxpe/bertin-roberta-base-spanish_sem_eval_2018_task_1")
output = classifier(cleaned_text)

# Display result
print(output)