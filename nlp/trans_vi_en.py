from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import streamlit as st

def load_model():
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-1.3B")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-1.3B")
    trans = pipeline("translation", model=model, tokenizer=tokenizer, padding=True, truncation=True)
    return trans

# Initialize model
st.cache_resource
def get_trans():
    return load_model()

def trans_qrt(query, trans):
    trans_result = trans(query)
    return trans_result