from urllib.parse import urlparse
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings  # Thay đổi ở đây

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
        embedding_model = OllamaEmbeddings(model="deepseek-r1:1.5b")  # Thay đổi ở đây

        for file in files:
            file_path = os.path.join(folder, file)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            if content:
                print(f"Đang xử lý file: {file}")

                # Tách nội dung thành các đoạn nhỏ
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, add_start_index=True)
                chunks = text_splitter.split_text(content)

                if os.path.exists(vector_db_path):
                    print(f"Database {vector_db_path} already exists. Merging with new data...")
                    
                    # Load the existing database
                    existing_db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
                    print('load done')
                    # Merge the new database with the existing one
                    new_db = FAISS.from_texts(chunks, embedding_model)
                    print('create new db')
                    existing_db.merge_from(new_db)  # Merges the new data into the existing one
                    print('merge done')
                    # Save the merged database
                    existing_db.save_local(vector_db_path)
                    print(f"Merged database saved to {vector_db_path}.")
                else:
                    # If the database does not exist, simply create and save a new one
                    db = FAISS.from_texts(chunks, embedding_model)
                    db.save_local(vector_db_path)
                    print(f"File {vector_db_path} done.")


                # db = FAISS.from_texts(chunks, embedding_model)
                # print('db done')

                # # Lưu cơ sở dữ liệu vector
                # db.save_local(vector_db_path)
                # print(f"File {vector_db_path} done.")

                # all_chunks.extend(chunks)

        # Xử lý embedding và thêm vào database - Đã sửa model ở đây
        # if all_chunks:
        #     print(1)
        #     embedding_model = OllamaEmbeddings(model="deepseek-r1:1.5b")  # Thay đổi ở đây
        #     print(len(all_chunks))
        #     # print(all_chunks)
        #     db = FAISS.from_texts(all_chunks, embedding_model)
        #     # db = FAISS.from_documents(all_chunks, embedding_model)
        #     print('db done')

        #     # Lưu cơ sở dữ liệu vector
        #     db.save_local(vector_db_path)
        #     print(f"Đã lưu dữ liệu vector vào {vector_db_path}.")

    except Exception as e:
        print(f"Lỗi khi xử lý file trong thư mục {folder}: {e}")

def main():
    input_folder = "sigma"
    vector_db_path = "vector_dbs/sigma_doc_db_deepseek"

    # Đọc file trong thư mục và nhập dữ liệu vào vector database
    process_files_in_folder(input_folder, vector_db_path)

    # input_folder = "omni"
    # vector_db_path = "vector_dbs/omni_doc_db_deepseek"

    # # Đọc file trong thư mục và nhập dữ liệu vào vector database
    # process_files_in_folder(input_folder, vector_db_path)

if __name__ == "__main__":
    main()