import gradio as gr
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import FAISS
import os

model_file = "models/vinallama-7b-chat.Q8_0.gguf"
vector_db_path = "vector_dbs/omni_doc_db"

def load_llm():
    return CTransformers(
        model=model_file,
        model_type="llama",
        max_new_tokens=1024,
        temperature=0.01
    )

def create_prompt():
    template = """<|im_start|>system\nSử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời\n
    {context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"""
    return PromptTemplate(template=template, input_variables=["context", "question"])

def read_vectors_db():
    if not os.path.exists(vector_db_path):
        print(f"Cơ sở dữ liệu vector tại {vector_db_path} không tồn tại.")
        return None

    try:
        embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-f16.gguf")
        db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        print(f"Lỗi khi tải cơ sở dữ liệu vector: {e}")
        return None

db = read_vectors_db()
llm = load_llm()
prompt = create_prompt()

if db:
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}, max_tokens_limit=1024),
        return_source_documents=False,
        chain_type_kwargs={'prompt': prompt}
    )
else:
    qa_chain = None

def ask_question(question):
    if qa_chain:
        response = qa_chain.invoke({"query": question})
        return response
    else:
        return "Không thể tải cơ sở dữ liệu vector. Vui lòng kiểm tra lại."

interface = gr.Interface(
    fn=ask_question,
    inputs="text",
    outputs="text",
    title="RAG Chatbot: Single Vector Database",
    description="Ask any question about the loaded documents. Powered by a local language model and FAISS."
)

if __name__ == "__main__":
    interface.launch()
