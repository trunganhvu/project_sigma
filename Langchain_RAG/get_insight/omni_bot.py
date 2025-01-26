from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import FAISS
import os

# Cấu hình
model_file = "models/vinallama-7b-chat.Q8_0.gguf"
vector_db_path = "vector_dbs/omni_doc_db"

# Load LLM
def load_llm(model_file):
    llm = CTransformers(
        model=model_file,
        model_type="llama",
        max_new_tokens=1024,
        temperature=0.01
    )
    return llm

# Tạo prompt template
def create_prompt(template):
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return prompt

# Tạo QA chain
def create_qa_chain(prompt, llm, db):
    llm_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}, max_tokens_limit=1024),
        return_source_documents=False,
        chain_type_kwargs={'prompt': prompt}
    )
    return llm_chain

# Đọc từ VectorDB
def read_vectors_db():
    try:
        if not os.path.exists(vector_db_path):
            print(f"Cơ sở dữ liệu vector tại {vector_db_path} không tồn tại.")
            return None

        embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-f16.gguf")
        db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        print(f"Lỗi khi tải cơ sở dữ liệu vector: {e}")
        return None

# Thực thi pipeline QA
def main():
    # Tải vector database
    db = read_vectors_db()
    if not db:
        return

    # Load LLM
    llm = load_llm(model_file)

    # Tạo prompt
    template = """<|im_start|>system\nSử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời\n
    {context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"""
    prompt = create_prompt(template)

    # Tạo QA chain
    qa_chain = create_qa_chain(prompt, llm, db)

    # Thử nghiệm với câu hỏi mẫu
    question = "Điểm ấn tượng của omni là gì?"
    response = qa_chain.invoke({"query": question})
    print("Câu trả lời:", response)

if __name__ == "__main__":
    main()
