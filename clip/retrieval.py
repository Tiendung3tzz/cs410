import os
from glob import glob
from PIL import Image
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import random
import streamlit as st

import pickle

def load_embedding():
#   current_dir = os.path.dirname(os.path.abspath(__file__))
#   npy_folder = os.path.join(current_dir, "../embeding")
#   npy_file_path = os.path.join(npy_folder, "image_embeddings.npy")

  embeddings = np.load("/kaggle/input/bogia-embedding/image_embeddings.npy")
  model = SentenceTransformer('clip-ViT-B-32')
  image_path = "/kaggle/input/bogia-data/avengre"
  image_files = glob (os.path.join(image_path, "*.png"))
  return embeddings, model, image_path,image_files
st.cache_resource
def get_clip():
  return load_embedding()

def fass_index(loaded_embeddings):
  dimension = len(loaded_embeddings[0])
  index = faiss.IndexFlatIP(dimension)
  index = faiss.IndexIDMap(index)

  vectors = np.array(loaded_embeddings).astype('float32')
  index.add_with_ids(vectors, np.array(range(len(loaded_embeddings))))
  return index

def search_image(query, model, index, image_files, top_k=5):
  # Query có thể là ảnh hoặc text
  if query.endswith(".png"):
      query = Image.open(query)

  query_embedding = model.encode(query)
  query_embedding = query_embedding.astype("float32").reshape(1, -1)

  distances, indices = index.search(query_embedding, top_k)

  retrieved_image_files = [image_files[i] for i in indices[0]]
  return distances[0], retrieved_image_files  # Trả về cả khoảng cách và ảnh khớp

# Visualize retrieved_image_files
def visualize_results(query, distances, retrieved_images):
  st.title("Kết quả tìm kiếm")

  # Hiển thị query
  st.subheader("Query:")
  if isinstance(query, Image.Image):  # Nếu là ảnh
      st.image(query, caption="Query Image", use_container_width=True)
  else:  # Nếu là văn bản
      st.markdown(f"**Query Text:** `{query}`")

  # Hiển thị các ảnh khớp kèm khoảng cách
  st.subheader("Kết quả khớp:")
  for i, (img_path, distance) in enumerate(zip(retrieved_images, distances)):
      with st.container():
        st.image(Image.open(img_path), caption=f"Match {i + 1} (Distance: {distance:.2f})", use_column_width=True)

def main_clip(query, loaded_embeddings, model, image_path,image_files):
  index = fass_index(loaded_embeddings)
  distances, retrieved_image_files = search_image(query, model, index, image_files)
  visualize_results(query, distances, retrieved_image_files)
  return
