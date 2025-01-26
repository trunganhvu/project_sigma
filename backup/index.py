from selenium import webdriver
from selenium.webdriver.common.by import By
from urllib.parse import urljoin, urlparse
import time
import csv
import os

# Hàm thu thập liên kết
def collect_links(driver, base_url, visited_links):
    links = set()
    for anchor in driver.find_elements(By.TAG_NAME, "a"):
        href = anchor.get_attribute("href")
        if href:
            full_url = urljoin(base_url, href)
            if (
                urlparse(full_url).netloc == urlparse(base_url).netloc  # Cùng domain
                and full_url not in visited_links  # Chưa được duyệt
            ):
                links.add(full_url)
    return links

# Hàm lấy nội dung từ trang
def get_page_content(driver):
    try:
        return driver.find_element(By.TAG_NAME, "body").text.strip()
    except Exception as e:
        print(f"Lỗi khi lấy nội dung: {e}")
        return ""

# Hàm lưu dữ liệu vào CSV
def save_to_csv(file_name, url, content, vector_content):
    # Kiểm tra nếu file chưa tồn tại thì tạo mới và thêm tiêu đề
    if not os.path.exists(file_name):
        with open(file_name, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["URL", "Content", "VectorContent"])

    # Ghi dữ liệu vào file
    with open(file_name, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([url, content, vector_content])

# Hàm chuyển đổi nội dung thành định dạng phù hợp cho vector database
def prepare_vector_content(content):
    # Loại bỏ các ký tự đặc biệt không cần thiết
    lines = content.splitlines()
    vector_friendly = " ".join(line.strip() for line in lines if line.strip())
    return vector_friendly

# Hàm chính để thu thập dữ liệu
def crawl_website(start_url, break_interval=300, break_time=10):
    driver = webdriver.Chrome()
    visited_links = set()
    links_to_visit = {start_url}
    csv_file_name = "website_data.csv"
    start_time = time.time()

    try:
        while links_to_visit:
            current_link = links_to_visit.pop()
            print(f"Đang duyệt: {current_link}")

            # Tải trang và đánh dấu đã duyệt
            driver.get(current_link)
            visited_links.add(current_link)

            # Lấy nội dung trang
            content = get_page_content(driver)
            vector_content = prepare_vector_content(content)

            # Lưu vào file CSV
            save_to_csv(csv_file_name, current_link, content, vector_content)

            # Thu thập liên kết mới
            new_links = collect_links(driver, start_url, visited_links)
            links_to_visit.update(new_links)

            # Kiểm tra thời gian nghỉ
            if time.time() - start_time >= break_interval:
                print(f"Nghỉ {break_time} giây sau {break_interval // 60} phút.")
                time.sleep(break_time)
                start_time = time.time()

    except Exception as e:
        print(f"Lỗi: {e}")
    finally:
        driver.quit()
        print(f"\nDữ liệu đã được lưu vào file CSV: {csv_file_name}")

# URL bắt đầu
start_url = "https://docs.omni.co/docs"  # Thay bằng URL của bạn
crawl_website(start_url, break_interval=300, break_time=10)
