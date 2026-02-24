import streamlit as st
from modules.metrics import evaluate_classification
from modules.preprocessing import preprocess_text
from modules.pos_ner import pos_tagging, named_entity_recognition
from modules.summarization import abstractive_summary
from modules.qa_chatbot import answer_question
from modules.translation import translate_text
from modules.transformer_module import transformer_sentiment

st.title("Smart NLP Suite")

text = st.text_area("Enter Text")

if st.button("Preprocess"):
    st.write(preprocess_text(text))

if st.button("POS Tagging"):
    st.write(pos_tagging(text))

if st.button("NER"):
    st.write(named_entity_recognition(text))

if st.button("Summarize"):
    st.write(abstractive_summary(text))

if st.button("Translate"):
    st.write(translate_text(text))

if st.button("Sentiment (Transformer)"):
    st.write(transformer_sentiment(text))

    st.subheader("📊 Performance Metrics (Demo Evaluation)")

# Example dummy data for demonstration
true_labels = ["Positive", "Negative", "Positive", "Neutral", "Positive"]
predicted_labels = ["Positive", "Negative", "Neutral", "Neutral", "Positive"]

if st.button("Show Performance Metrics"):
    results = evaluate_classification(true_labels, predicted_labels)
    st.write(results)