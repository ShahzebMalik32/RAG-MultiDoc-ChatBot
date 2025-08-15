import os
import zipfile
import shutil
import gradio as gr
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from llama_index.core import SimpleDirectoryReader
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient

UPLOAD_DIR = "uploaded_docs"
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
client = InferenceClient(model="deepseek-ai/DeepSeek-V3-0324", token=os.environ["HF_TOKEN"])

# Store embeddings and texts
doc_texts = []
doc_embeddings = []

def extract_and_embed(zip_file):
    global doc_texts, doc_embeddings
    # Clear previous uploads
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # Extract ZIP
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(UPLOAD_DIR)

    # Read documents
    documents = SimpleDirectoryReader(input_dir=UPLOAD_DIR).load_data()
    doc_texts = [doc.text for doc in documents]
    doc_embeddings = [embedder.embed_query(text) for text in doc_texts]

    return f"✅ Uploaded and embedded {len(doc_texts)} documents."

def answer_question(query):
    if not doc_texts:
        return "⚠️ Please upload documents first."

    query_embedding = embedder.embed_query(query)
    cosine_sim = cosine_similarity([query_embedding], doc_embeddings)[0]

    top_k = 2
    top_indices = np.argsort(cosine_sim)[::-1][:top_k]
    top_docs = "\n\n".join([doc_texts[i] for i in top_indices])

    context_prompt = f"Context:\n{top_docs}\n\nQuestion:\n{query}"

    completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer based on the context. Only answer the question."},
            {"role": "user", "content": context_prompt}
        ]
    )

    return completion.choices[0].message["content"]

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 📘 Ask Questions from Your Documents (DeepSeek + Langchain)")
    zip_upload = gr.File(label="Upload a ZIP folder of .txt/.pdf files", file_types=[".zip"])
    upload_output = gr.Textbox(label="Upload Status")
    upload_btn = gr.Button("Process Uploaded Files")

    query_input = gr.Textbox(label="Ask a Question")
    answer_output = gr.Textbox(label="Answer")
    ask_btn = gr.Button("Get Answer")

    upload_btn.click(fn=extract_and_embed, inputs=zip_upload, outputs=upload_output)
    ask_btn.click(fn=answer_question, inputs=query_input, outputs=answer_output)

demo.launch()
