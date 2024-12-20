from nlp.name_ent import *
from nlp.replace import replace_named_entities
from nlp.trans_vi_en import *
import streamlit as st

Men = ['Trấn Thành', 'Bảo Lâm']
Woman = ['Thu Trang','Diệu Linh']

st.title("Vietnamese Named Entity Recognition (NER)")
model, tokenizer = get_trans()
nlp = get_nlp_pipeline()
# Input text from user
query = st.text_input("Nhập câu cần phân tích NER:", "")
# Process input and display results
updated_query = ""
if st.button("Phân tích"):
    if query.strip():
        with st.spinner("Đang phân tích..."):
            ent_results = name_entity(query, nlp)
            updated_query = replace_named_entities(query, ent_results, Men, Woman)
            trans_results = trans_qrt(updated_query, model, tokenizer)
        st.success("Phân tích hoàn tất!")
        st.write("Kết quả phân tích:")
        st.write(updated_query)
        st.write(trans_results)
        # if st.button("dịch"):
        #         st.write(updated_query)                
        #         with st.spinner("Đang dịch..."):
        #             trans_results = trans_qrt(updated_query, model, tokenizer)
        #         st.success("Phân tích hoàn tất!")
        #         st.write("Kết quả phân tích:")
        #         st.write(trans_results)

    else:
        st.warning("Vui lòng nhập một câu trước khi phân tích.")

