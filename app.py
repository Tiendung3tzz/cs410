from nlp.name_ent import *
from nlp.replace import replace_named_entities
from nlp.trans_vi_en import *
from clip.retrieval import *
from arcface.arcface_model import *
import streamlit as st

Men = ['Trấn Thành', 'Tuấn Trần']
Woman = ['Thu Trang','Diệu Linh','Ngọc Giàu','Lê Giang', 'Uyển Ân','Hariwon']

st.title("Vietnamese Named Entity Recognition (NER)")
model_trans, tokenizer = get_trans()
nlp = get_nlp_pipeline()
embeddings, model_clip, image_path,image_files = get_clip()
model_arcface, mtcnn = get_arcface()
# Input text from user
query = st.text_input("Nhập câu cần phân tích NER:", "")
threshod = st.slider(
    label="threshod",
    min_value=1,
    max_value=100,
    value=70,  # Giá trị mặc định
    step=1     # Bước tăng/giảm
)
threshod = threshod/100
# Process input and display results
updated_query = ""
if st.button("Phân tích"):
    if query.strip():
        with st.spinner("Đang phân tích..."):
            ent_results = name_entity(query, nlp)
            updated_query = replace_named_entities(query, ent_results, Men, Woman)
            trans_results = trans_qrt(updated_query, model_trans, tokenizer)
        st.success("Phân tích hoàn tất!")
        st.write("Kết quả phân tích:")
        st.write(updated_query)
        st.write(trans_results)
        distances, retrieved_image_files = main_clip(trans_results, embeddings, model_clip, image_path,image_files)
        visualize_results(trans_results, distances, retrieved_image_files)
        max_index, normalized_array,img_final = arcface_run(retrieved_image_files,mtcnn,model_arcface,distances,ent_results,threshod)
        st.write(normalized_array,max_index)
        visualize_results(trans_results, normalized_array[max_index], img_final,1)

    else:
        st.warning("Vui lòng nhập một câu trước khi phân tích.")

