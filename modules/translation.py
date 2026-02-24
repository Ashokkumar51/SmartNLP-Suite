from transformers import pipeline

translator = pipeline("translation_en_to_fr")

def translate_text(text):
    result = translator(text)
    return result[0]['translation_text']