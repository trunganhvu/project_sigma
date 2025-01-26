import psycopg2
import csv
import numpy as np
from sentence_transformers import SentenceTransformer  # Sử dụng mô hình để tạo vector

# Kết nối đến cơ sở dữ liệu PostgreSQL
conn = psycopg2.connect(
    dbname="mydatabase",
    user="myuser",
    password="password",
    host="192.168.1.7"
)

# Khởi tạo mô hình SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")  # Thay đổi nếu dùng mô hình khác

# Tạo bảng nếu chưa tồn tại
with conn.cursor() as cur:
    cur.execute("""
        CREATE TABLE IF NOT EXISTS items (
            id SERIAL PRIMARY KEY,
            url TEXT NOT NULL,
            content TEXT NOT NULL,
            vector_content TEXT NOT NULL,
            embedding VECTOR(1536)  -- Kích thước vector 384 cho MiniLM
        );
    """)
    conn.commit()

# Hàm chèn dữ liệu vào PostgreSQL
def insert_into_db(url, content, vector_content, embedding):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO items (url, content, vector_content, embedding)
            VALUES (%s, %s, %s, %s);
        """, (url, content, vector_content, embedding))
        conn.commit()

# Đọc dữ liệu từ file CSV và xử lý
csv_file = "website_data.csv"
try:
    with open(csv_file, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            url = row["URL"]
            content = row["Content"]
            vector_content = row["VectorContent"]

            # Tạo vector từ nội dung
            embedding = model.encode(vector_content)
            embedding = np.array(embedding).tolist()  # Chuyển thành danh sách Python

            # Chèn vào cơ sở dữ liệu
            insert_into_db(url, content, vector_content, embedding)

            print(f"Đã chèn URL: {url}")

except Exception as e:
    print(f"Lỗi: {e}")

finally:
    conn.close()
    print("Đã đóng kết nối cơ sở dữ liệu.")
