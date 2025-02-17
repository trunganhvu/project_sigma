import gradio as gr
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings  # Changed embedding model
from langchain.vectorstores import FAISS
import os
from langchain_community.embeddings import OllamaEmbeddings

model_file = "models/deepseek-llm-7b-chat.Q8_0.gguf"  # Update with your model path
# vector_db_path = "vector_dbs/omni_doc_db3"
vector_db_path = "vector_dbs/sigma_doc_db_deepseek"

def load_llm():
    return CTransformers(
        model=model_file,
        model_type="deepseek",
        max_new_tokens=2048,
        temperature=0.1,
        context_length=4096
    )

def create_prompt():
    template = """<｜begin▁of▁sentence｜>
### System:
You are an AI assistant. Use the following context to answer the user's question. 
If you don't know the answer, say you don't know. Be detailed and helpful.

Context: {context}

### User:
{question}

### Assistant:
"""
    return PromptTemplate(template=template, input_variables=["context", "question"])

def read_vectors_db():
    if not os.path.exists(vector_db_path):
        print(f"Vector database at {vector_db_path} does not exist.")
        return None

    try:
        embedding_model = OllamaEmbeddings(model="deepseek-r1:1.5b")
        # embedding_model = HuggingFaceEmbeddings(
        #     model_name="sentence-transformers/all-mpnet-base-v2",
        #     model_kwargs={'device': 'cpu'}
        # )
        db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        print(f"Error loading vector DB: {e}")
        return None

db = read_vectors_db()
llm = load_llm()
prompt = create_prompt()

if db:
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=False,
        chain_type_kwargs={'prompt': prompt},
        verbose=True
    )
else:
    qa_chain = None

def ask_question(question):
    if qa_chain:
        response = qa_chain.invoke({"query": question})
        return response['result']
    else:
        return "Vector database not loaded. Please check setup."

interface = gr.Interface(
    fn=ask_question,
    inputs="text",
    outputs="text",
    title="English RAG Chatbot",
    description="Ask questions about your documents. Powered by DeepSeek and FAISS.",
    allow_flagging="never"
)

if __name__ == "__main__":
    interface.launch()