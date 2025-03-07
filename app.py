import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
import torch
import base64

# MODEL AND TOKENIZER
checkpoint = "LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float32)

# FILE LOADER AND PREPROCESSING (Only for Text-Based PDFs)
def file_preprocessing(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    final_texts = ""
    for text in texts:
        print(text)
        final_texts = final_texts + text.page_content
    return final_texts

# LM PIPELINE
def llm_pipeline(filepath):
    pipe_sum = pipeline(
        'summarization',
        model=base_model,
        tokenizer=tokenizer,
        max_length=1000,
        min_length=50
    )

    # Extract text from the PDF using PyPDFLoader (for text-based PDFs only)
    input_text = file_preprocessing(filepath)
    result = pipe_sum(input_text)
    result = result[0]['summary_text']
    return result

@st.cache_data
# FUNCTION TO DISPLAY THE PDF OF A GIVEN FILE
def displayPDF(file):
    # OPENING FILE FROM FILE PATH
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    
    # EMBEDDING PDF IN HTML
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    
    # DISPLAY FILE
    st.markdown(pdf_display, unsafe_allow_html=True)

# STREAMLIT CODE
st.set_page_config(layout="wide")

def main():
    st.title("Document Summarization App using Language Model")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        # Save uploaded file to disk
        filepath = "data/" + uploaded_file.name
        with open(filepath, 'wb') as temp_file:
            temp_file.write(uploaded_file.read())

        if st.button("Summarization"):
            col1, col2 = st.columns(2)

            with col1:
                st.info("Uploaded PDF File")
                pdf_viewer = displayPDF(filepath)

            with col2:
                st.info("Summarizing is below")

                # Run the summarization pipeline
                summary = llm_pipeline(filepath)
                st.success(summary)

if __name__ == "__main__":
    main()
