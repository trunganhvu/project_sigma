import gradio as gr
from langchain_community.llms import CTransformers
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
import os

model_file = "models/deepseek-llm-7b-chat.Q8_0.gguf"

def load_llm():
    return CTransformers(
        model=model_file,
        model_type="deepseek",
        max_new_tokens=2048,
        temperature=0.1,
        context_length=4096
    )

def load_documents():
    loader = DirectoryLoader('omni', glob="**/*.txt")
    documents = loader.load()
    
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return text_splitter.split_documents(documents)

def ask_question(question):
    context = "\n\n".join([doc.page_content for doc in chunks])
    prompt = f"""### System:
Bạn là trợ lý AI. Sử dụng thông tin sau để trả lời câu hỏi:

{context}

### User:
{question}

### Assistant:
"""
    return llm(prompt)

# Khởi tạo hệ thống
llm = load_llm()
chunks = load_documents()

# Tạo giao diện
gr.Interface(
    fn=ask_question,
    inputs="text",
    outputs="text",
    title="Chatbot không dùng Vector DB",
    description="Hỏi đáp trực tiếp trên toàn bộ tài liệu trong thư mục 'omni'"
).launch()