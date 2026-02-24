from transformers import pipeline

# Load specific summarization model
summarizer = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)

def abstractive_summary(text):
    prompt = "Summarize the following text: " + text
    result = summarizer(prompt, max_length=120, min_length=30, do_sample=False)
    return result[0]["generated_text"]