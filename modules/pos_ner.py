import spacy
import streamlit as st

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

def pos_tagging(text):
    doc = nlp(text)
    return [(token.text, token.pos_) for token in doc]

def named_entity_recognition(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]