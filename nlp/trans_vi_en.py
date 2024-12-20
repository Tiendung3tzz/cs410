from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

from transformers import pipeline
import streamlit as st

def load_model():

    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    return model, tokenizer

# Initialize model
st.cache_resource
def get_trans():
    return load_model()

def trans_qrt(query, model, tokenizer):
    return 