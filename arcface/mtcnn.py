from facenet_pytorch import MTCNN, InceptionResnetV1
import streamlit as st

def load_model(DEVICE = "cpu"):
    mtcnn = MTCNN(keep_all=True, device=DEVICE)
    return mtcnn
# Initialize model
st.cache_resource
def get_mtcnn(DEVICE):
    return load_model(DEVICE)