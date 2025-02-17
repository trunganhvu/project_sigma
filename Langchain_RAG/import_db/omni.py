from urllib.parse import urlparse
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings  # Changed embedding model

def sanitize_filename(url):
    """Convert URL to valid filename by removing invalid characters."""
    parsed_url = urlparse(url)
    filename = parsed_url.path.replace("/", "_").strip("_")
    return filename if filename else "index"

def process_files_in_folder(folder, vector_db_path):
    """Process TXT files in folder and add to vector database."""
    try:
        if not os.path.exists(folder):
            print(f"Directory {folder} does not exist.")
            return

        files = [f for f in os.listdir(folder) if f.endswith(".txt")]
        if not files:
            print(f"No TXT files found in {folder}.")
            return

        all_chunks = []
        for file in files:
            file_path = os.path.join(folder, file)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            if content:
                print(f"Processing file: {file}")

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=512,
                    chunk_overlap=50,
                    separators=["\n\n", "\n", ". ", " ", ""]  # Better for English
                )
                chunks = text_splitter.split_text(content)
                all_chunks.extend(chunks)

        if all_chunks:
            # Using high-quality English embeddings
            embedding_model = HuggingFaceEmbeddings(
                # model_name="sentence-transformers/all-mpnet-base-v2",
                model_name="sentence-transformers/all-MiniLM-L6-v1",
                model_kwargs={'device': 'cpu'}
            )
            
            db = FAISS.from_texts(all_chunks, embedding_model)
            db.save_local(vector_db_path)
            print(f"Vector database saved to {vector_db_path}.")

    except Exception as e:
        print(f"Error processing files in {folder}: {e}")

def main():
    input_folder = "omni"
    vector_db_path = "vector_dbs/omni_doc_db3"
    process_files_in_folder(input_folder, vector_db_path)

if __name__ == "__main__":
    main()