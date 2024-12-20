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
    tokenizer.src_lang = "vie_Latn"
    encoded = tokenizer(query, return_tensors="pt")
    generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"])
    translation = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return translation