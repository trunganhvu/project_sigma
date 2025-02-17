import ollama
import re
import gradio as gr
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

embedding_model = OllamaEmbeddings(model="deepseek-r1:1.5b")
vector_db_path = "vector_dbs/sigma_doc_db_deepseek"

if os.path.exists(vector_db_path):
    db = FAISS.load_local(
        vector_db_path,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    retriever = db.as_retriever()
else:
    raise Exception("Vector database không tồn tại. Hãy chạy process_files_in_folder trước.")

def retrieve_context(question):
    results = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in results])
    return context

def format_question(question: str) -> str:
    template = (
        "You are an AI assistant. Answer the question strictly based on the provided context.\n"
        "If the answer cannot be found in the context, respond with 'Information not found in the provided sources.'\n"
        "Be concise and provide references to the source documents.\n\n"
        "Question: {question}\n"
        "Context:"
    )
    return template.format(question=question)
def query_deepseek(question, context):
    formatted_prompt = format_question(question)
    
    response = ollama.chat(
        model="deepseek-r1:7b",
        messages=[{'role': 'user', 'content': formatted_prompt}]
    )
    
    response_content = response['message']['content']
    final_answer = re.sub(r'<think>.*?</think>', '', response_content, flags=re.DOTALL).strip()
    return final_answer

def ask_question(question):
    context = retrieve_context(question)
    answer = query_deepseek(question, context)
    return answer

# Thiết lập giao diện Gradio
interface = gr.Interface(
    fn=ask_question,
    inputs="text",
    outputs="text",
    title="RAG Chatbot: Foundations of LLMs",
    description="Ask any question about the Foundations of LLMs book. Powered by DeepSeek-R1."
)

interface.launch()