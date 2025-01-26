import psycopg2
import numpy as np
from sentence_transformers import SentenceTransformer
import openai

# Kết nối đến cơ sở dữ liệu PostgreSQL
conn = psycopg2.connect(
    dbname="mydatabase",
    user="myuser",
    password="password",
    host="192.168.1.7"
)

# Khởi tạo mô hình SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")  # Thay đổi nếu dùng mô hình khác

# Hàm tính toán khoảng cách cosine giữa hai vector
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Hàm truy vấn vector tương tự từ cơ sở dữ liệu
def query_similar(query_text, top_k=5):
    # Tạo nhúng (embedding) từ câu truy vấn
    query_embedding = model.encode(query_text).tolist()

    # Truy vấn tất cả vector từ cơ sở dữ liệu
    with conn.cursor() as cur:
        cur.execute("SELECT id, url, content, embedding FROM items;")
        rows = cur.fetchall()

    # Tính toán độ tương đồng cosine và sắp xếp
    results = []
    for row in rows:
        item_id, url, content, embedding = row
        embedding = np.array(embedding)  # Chuyển về numpy array
        similarity = cosine_similarity(query_embedding, embedding)
        results.append((similarity, {"id": item_id, "url": url, "content": content}))

    # Lấy top K kết quả tương tự nhất
    results = sorted(results, key=lambda x: x[0], reverse=True)[:top_k]
    return results

# Hàm tạo prompt và gọi API GPT-3 để sinh kết quả
def generate_insights(query_text):
    results = query_similar(query_text)
    context = "\n\n".join([r[1]["content"] for r in results])
    prompt = f"Given the following context:\n{context}\n\nProvide insights comparing the companies."
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=500
    )
    return response["choices"][0]["text"]

# Hàm chính
if __name__ == "__main__":
    # Nhập câu truy vấn từ người dùng
    query_text = input("Enter your query: ")

    # Gọi hàm generate_insights
    try:
        insights = generate_insights(query_text)
        print("\nGenerated Insights:")
        print(insights)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()
