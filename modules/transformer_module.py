from transformers import pipeline

classifier = pipeline("sentiment-analysis")

def transformer_sentiment(text):
    result = classifier(text)
    return result[0]