from selenium import webdriver
from selenium.webdriver.common.by import By
from urllib.parse import urljoin
import os
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import GPT4AllEmbeddings
import time

def append_to_file_if_not_exists(output_file, link):
    """Kiểm tra và thêm liên kết vào file nếu chưa tồn tại."""
    try:
        # Đọc nội dung file
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as file:
                existing_links = set(file.read().splitlines())
        else:
            existing_links = set()

        # Nếu liên kết chưa tồn tại, thêm vào file
        if link not in existing_links:
            with open(output_file, "a", encoding="utf-8") as file:
                file.write(link + "\n")
    except Exception as e:
        print(f"Lỗi khi kiểm tra/thêm liên kết: {e}")

def extract_sidebar_links(driver, base_url, output_file):
    """Trích xuất tất cả các liên kết từ <div class="Sidebar1t2G1ZJq-vU1 rm-Sidebar hub-sidebar-content"> và lưu vào file."""
    try:
        # Tìm thẻ div với class cụ thể
        sidebar_menu = driver.find_element(By.CSS_SELECTOR, "div.Sidebar1t2G1ZJq-vU1.rm-Sidebar.hub-sidebar-content")
        links = sidebar_menu.find_elements(By.TAG_NAME, "a")

        # Thu thập tất cả các liên kết
        for link in links:
            href = link.get_attribute("href")
            if href:
                append_to_file_if_not_exists(output_file, urljoin(base_url, href))

    except Exception as e:
        print(f"Lỗi khi trích xuất liên kết: {e}")

def get_content_and_import_to_db(driver, url, vector_db_path):
    """Truy cập URL, lấy nội dung từ <main> và nhập vào vector database."""
    try:
        # Truy cập URL
        driver.get(url)

        # Lấy nội dung từ thẻ <main>
        content_element = driver.find_element(By.CSS_SELECTOR, "main.docMainContainer_TBSr")
        content = content_element.text.strip()
        print(f"Content {url} {len(content)}.")

        # Chuẩn bị dữ liệu để import
        if content:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
            chunks = text_splitter.split_text(content)

            embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-f16.gguf")
            db = FAISS.from_texts(chunks, embedding_model)

            # Lưu vector database
            db.save_local(vector_db_path)
            print(f"Đã nhập dữ liệu từ {url} vào vector database.")
            time.sleep(1)

    except Exception as e:
        print(f"Lỗi khi xử lý URL {url}: {e}")

def process_urls_from_file(input_file, vector_db_path):
    """Đọc từng URL từ file, truy cập và nhập dữ liệu vào vector database."""
    driver = webdriver.Chrome()
    try:
        if os.path.exists(input_file):
            with open(input_file, "r", encoding="utf-8") as file:
                urls = file.read().splitlines()

            for url in urls:
                print(f"Đang xử lý: {url}")
                get_content_and_import_to_db(driver, url, vector_db_path)
        else:
            print(f"File {input_file} không tồn tại.")

    except Exception as e:
        print(f"Lỗi trong quá trình xử lý file: {e}")

    finally:
        driver.quit()

def main():
    input_file = "source/omni_doc_links.txt"
    vector_db_path = "vector_dbs/omni_doc_db"

    # Xử lý từng URL từ file và nhập dữ liệu vào vector database
    process_urls_from_file(input_file, vector_db_path)

if __name__ == "__main__":
    main()
