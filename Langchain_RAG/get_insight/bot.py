from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings

# Cấu hình
model_file = "models/vinallama-7b-chat.Q8_0.gguf"
omni_vector_db_path = "vector_dbs/omni_doc_db"
sigma_vector_db_path = "vector_dbs/sigma_doc_db"

# Load vector databases
def load_vector_db(vector_db_path):
    try:
        embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-f16.gguf")
        db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        print(f"Lỗi khi tải cơ sở dữ liệu vector từ {vector_db_path}: {e}")
        return None

# Load LLM
def load_llm():
    return CTransformers(
        model=model_file,
        model_type="llama",
        max_new_tokens=1024,
        temperature=0.01
    )

# So sánh thông tin
def compare_companies(question):
    omni_db = load_vector_db(omni_vector_db_path)
    sigma_db = load_vector_db(sigma_vector_db_path)
    if not omni_db or not sigma_db:
        return

    # Tạo retriever
    omni_retriever = omni_db.as_retriever(search_kwargs={"k": 3})
    sigma_retriever = sigma_db.as_retriever(search_kwargs={"k": 3})

    # Lấy thông tin từ cả hai database
    llm = load_llm()
    omni_qa = RetrievalQA.from_chain_type(llm=llm, retriever=omni_retriever)
    sigma_qa = RetrievalQA.from_chain_type(llm=llm, retriever=sigma_retriever)

    # Truy vấn thông tin
    omni_response = omni_qa.invoke({"query": question})
    sigma_response = sigma_qa.invoke({"query": question})

    # Tạo so sánh
    print(f"Thông tin từ Omni:\n{omni_response}\n")
    print(f"Thông tin từ Sigma:\n{sigma_response}\n")

    # Có thể chỉnh sửa prompt để so sánh tự động hơn nếu cần.
    print(f"So sánh ưu nhược điểm từ hai công ty dựa trên thông tin đã thu thập.")
    # Tùy chỉnh logic tại đây để so sánh.

# Gọi hàm so sánh
compare_companies("So sánh điểm mạnh và điểm yếu của công ty Omni và Sigma?")
