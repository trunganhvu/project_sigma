from urllib.parse import urlparse
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import GPT4AllEmbeddings

def sanitize_filename(url):
    """Chuyển đổi URL thành tên tệp hợp lệ bằng cách loại bỏ ký tự không hợp lệ."""
    parsed_url = urlparse(url)
    filename = parsed_url.path.replace("/", "_").strip("_")
    return filename if filename else "index"

def process_files_in_folder(folder, vector_db_path):
    """Đọc từng file TXT trong thư mục, xử lý nội dung và thêm vào vector database."""
    try:
        if not os.path.exists(folder):
            print(f"Thư mục {folder} không tồn tại.")
            return

        files = [f for f in os.listdir(folder) if f.endswith(".txt")]
        if not files:
            print(f"Không tìm thấy file TXT nào trong thư mục {folder}.")
            return

        all_chunks = []
        for file in files:
            file_path = os.path.join(folder, file)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            if content:
                print(f"Đang xử lý file: {file}")

                # Tách nội dung thành các đoạn nhỏ
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
                chunks = text_splitter.split_text(content)
                all_chunks.extend(chunks)

        # Xử lý embedding và thêm vào database
        if all_chunks:
            embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-f16.gguf")
            db = FAISS.from_texts(all_chunks, embedding_model)

            # Lưu cơ sở dữ liệu vector
            db.save_local(vector_db_path)
            print(f"Đã lưu dữ liệu vector vào {vector_db_path}.")

    except Exception as e:
        print(f"Lỗi khi xử lý file trong thư mục {folder}: {e}")

def main():
    input_folder = "sigma"
    vector_db_path = "vector_dbs/sigma_doc_db"

    # Đọc file trong thư mục và nhập dữ liệu vào vector database
    process_files_in_folder(input_folder, vector_db_path)

if __name__ == "__main__":
    main()
