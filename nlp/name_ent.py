from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import streamlit as st

def load_model():
    tokenizer = AutoTokenizer.from_pretrained("NlpHUST/ner-vietnamese-electra-base")
    model = AutoModelForTokenClassification.from_pretrained("NlpHUST/ner-vietnamese-electra-base")
    nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    return nlp

# Initialize model
st.cache_resource
def get_nlp_pipeline():
    return load_model()

def name_entity(query, nlp):
    ner_results = nlp(query)
    return ner_results