from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings
import requests
from bs4 import BeautifulSoup
import re
import os
from urllib.parse import urlparse
from datetime import datetime

# Khai bao bien
pdf_data_path = "data"
vector_db_path = "vectorstores/db_faiss"

# Ham 1. Tao ra vector DB tu 1 doan text
def create_db_from_text():
    raw_text = """Nhằm đáp ứng nhu cầu và thị hiếu của khách hàng về việc sở hữu số tài khoản đẹp, dễ nhớ, giúp tiết kiệm thời gian, mang đến sự thuận lợi trong giao dịch. Ngân hàng Sài Gòn – Hà Nội (SHB) tiếp tục cho ra mắt tài khoản số đẹp 9 số và 12 số với nhiều ưu đãi hấp dẫn.
    Cụ thể, đối với tài khoản số đẹp 9 số, SHB miễn phí mở tài khoản số đẹp trị giá 880.000đ; giảm tới 80% phí mở tài khoản số đẹp trị giá từ 1,1 triệu đồng; phí mở tài khoản số đẹp siêu VIP chỉ còn 5,5 triệu đồng.
    Đối với tài khoản số đẹp 12 số, SHB miễn 100% phí mở tài khoản số đẹp, khách hàng có thể lựa chọn tối đa toàn bộ dãy số của tài khoản. Đây là một trong những điểm ưu việt của tài khoản số đẹp SHB so với thị trường. Ngoài ra, khách hàng có thể lựa chọn số tài khoản trùng số điện thoại, ngày sinh, ngày đặc biệt, hoặc số phong thủy mang lại tài lộc cho khách hàng trong quá trình sử dụng.
    Hiện nay, SHB đang cung cấp đến khách hàng 3 loại tài khoản số đẹp: 9 số, 10 số và 12 số. Cùng với sự tiện lợi khi giao dịch online mọi lúc mọi nơi qua dịch vụ Ngân hàng số, hạn chế rủi ro khi sử dụng tiền mặt, khách hàng còn được miễn phí chuyển khoản qua mobile App SHB, miễn phí quản lý và số dư tối thiểu khi sử dụng tài khoản số đẹp của SHB.
    Ngoài kênh giao dịch tại quầy, khách hàng cũng dễ dàng mở tài khoản số đẹp trên ứng dụng SHB Mobile mà không cần hồ sơ thủ tục phức tạp.
    Hướng mục tiêu trở thành ngân hàng số 1 về hiệu quả tại Việt Nam, ngân hàng bán lẻ hiện đại nhất và là ngân hàng số được yêu thích nhất tại Việt Nam, SHB sẽ tiếp tục nghiên cứu và cho ra mắt nhiều sản phẩm dịch vụ số ưu việt cùngchương trình ưu đãi hấp dẫn, mang đến cho khách hàng lợi ích và trải nghiệm tuyệt vời nhất.
    Để biết thêm thông tin về chương trình, Quý khách vui lòng liên hệ các điểm giao dịch của SHB trên toàn quốc hoặc Hotline *6688"""

    # Chia nho van ban
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=512,
        chunk_overlap=50,
        length_function=len
    )

    chunks = text_splitter.split_text(raw_text)

    # Embeding
    embedding_model = GPT4AllEmbeddings(model_path = "models/all-MiniLM-L6-v2-f16.gguf")

    # Dua vao Faiss Vector DB
    db = FAISS.from_texts(texts=chunks, embedding=embedding_model)
    db.save_local(vector_db_path)
    return db


def create_db_from_files():
    # Khai bao loader de quet toan bo thu muc dataa
    loader = DirectoryLoader(pdf_data_path, glob="*.pdf", loader_cls = PyPDFLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # Embeding
    embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-f16.gguf")
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(vector_db_path)
    return db

# def create_db_from_website(url, key, value):
#     response = requests.get(url)
#     response.raise_for_status()

#     # Phân tích nội dung HTML
#     soup = BeautifulSoup(response.text, 'html.parser')

#     # Tìm nội dung theo id hoặc class
#     if key == "id":
#         content = soup.find(id=value)
#     elif key == "class":
#         content = soup.find(class_=value)
#     else:
#         raise ValueError("Key phải là 'id' hoặc 'class'")
#     if not content:
#         raise ValueError(f"Không tìm thấy nội dung với {key} = {value}")

#     raw_text = content.get_text(separator="\n", strip=True)
#     raw_text = remove_unnecessary_lines(raw_text)
#     print(raw_text)
#     export_to_txt('omni', url, raw_text)
    # Chia nhỏ văn bản
    # text_splitter = CharacterTextSplitter(
    #     separator="\n",
    #     chunk_size=512,
    #     chunk_overlap=50,
    #     length_function=len
    # )
    # chunks = text_splitter.split_text(raw_text)

    # # Embeding
    # embedding_model = GPT4AllEmbeddings(model_path="models/all-MiniLM-L6-v2-f16.gguf")
    # db = FAISS.from_texts(texts=chunks, embedding=embedding_model)
    # db.save_local(vector_db_path)
    # return db

exported_files = set()
def create_db_from_website(output_dir, url, key, value):
    visited_links = set()  # Bộ lưu trữ các liên kết đã duyệt
    error_file = "result/error.txt"  # Tệp ghi lỗi
    os.makedirs(os.path.dirname(error_file), exist_ok=True)

    def process_page(current_url):
        # Kiểm tra nếu link đã được duyệt
        if current_url in visited_links:
            return
        visited_links.add(current_url)

        try:
            # Gửi yêu cầu GET đến URL
            response = requests.get(current_url, timeout=10)
            response.raise_for_status()

            # Phân tích nội dung HTML
            soup = BeautifulSoup(response.text, 'html.parser')

            # Tìm nội dung theo id hoặc class
            if key == "id":
                content = soup.find(id=value)
            elif key == "class":
                content = soup.find(class_=value)

            # Nếu không tìm thấy nội dung, ghi vào lỗi
            if not content:
                raise ValueError(f"Không tìm thấy nội dung với {key} = {value}")

            # Lấy nội dung văn bản, làm sạch và xuất ra tệp
            raw_text = content.get_text(separator="\n", strip=True)
            raw_text = remove_unnecessary_lines(raw_text)
            export_to_txt(output_dir, current_url, raw_text)

            # Tìm tất cả các liên kết trên trang và duyệt tiếp
            for link in soup.find_all("a", href=True):
                next_url = link["href"]
                # Bỏ qua liên kết không hợp lệ hoặc liên kết bên ngoài
                if not next_url.startswith("http"):
                    next_url = requests.compat.urljoin(current_url, next_url)
                process_page(next_url)

        except Exception as e:
            # Ghi lỗi vào file
            with open(error_file, "a", encoding="utf-8") as file:
                file.write(f"{datetime.now()}: {current_url} - {str(e)}\n")

    # Bắt đầu duyệt từ trang ban đầu
    process_page(url)

def export_to_txt(output_dir, url, content):
    filename = re.sub(r'[^\w]', '_', urlparse(url).path.strip('/')) 
    if filename not in exported_files:
        filename = f"{filename}.txt"

        # Tạo thư mục lưu trữ nếu cần
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)

        # Ghi nội dung vào tệp
        with open(filepath, "w", encoding="utf-8") as file:
            file.write(content)

        exported_files.add(filename)
        print(f"Nội dung đã được xuất ra tệp: {filepath}")
        return filepath

def remove_unnecessary_lines(text):
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        # Loại bỏ icon và ký tự không đọc được (emoji, ký tự đặc biệt, v.v.)
        line = re.sub(r'[^\w\s.,!?]', '', line) 

        # Bỏ qua dòng trống hoặc dòng quá ngắn (dưới 5 ký tự không chứa số/chữ)
        if line.strip() and len(re.sub(r'\W+', '', line)) >= 5:
            cleaned_lines.append(line.strip())
    return "\n".join(cleaned_lines)

url = "https://docs.omni.co/docs/connections/database/bigquery"
create_db_from_website('omni', url, 'class', 'row')
# create_db_from_text()

