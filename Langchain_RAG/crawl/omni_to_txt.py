from selenium import webdriver
from selenium.webdriver.common.by import By
from urllib.parse import urljoin, urlparse
import os
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

def sanitize_filename(url):
    """Chuyển đổi URL thành tên tệp hợp lệ bằng cách loại bỏ ký tự không hợp lệ."""
    parsed_url = urlparse(url)
    filename = parsed_url.path.replace("/", "_").strip("_")
    return filename if filename else "index"

def save_content_to_file(folder, url, content):
    """Lưu nội dung vào tệp với tên tệp dựa trên URL."""
    try:
        if not os.path.exists(folder):
            os.makedirs(folder)

        filename = sanitize_filename(url) + ".txt"
        filepath = os.path.join(folder, filename)

        with open(filepath, "w", encoding="utf-8") as file:
            file.write(content)
        print(f"Đã lưu nội dung từ {url} vào {filepath}.")

    except Exception as e:
        print(f"Lỗi khi lưu nội dung từ {url}: {e}")

def get_content_and_save_to_file(driver, url, output_folder):
    """Truy cập URL, lấy nội dung từ <main> và lưu vào tệp."""
    try:
        # Truy cập URL
        driver.get(url)

        # Lấy nội dung từ thẻ <main>
        content_element = driver.find_element(By.CSS_SELECTOR, "main.docMainContainer_TBSr")
        content = content_element.text.strip()

        # Lưu nội dung vào tệp
        if content:
            save_content_to_file(output_folder, url, content)

    except Exception as e:
        print(f"Lỗi khi xử lý URL {url}: {e}")

def process_urls_from_file(input_file, output_folder):
    """Đọc từng URL từ file, truy cập và lưu nội dung vào tệp."""
    driver = webdriver.Chrome()
    try:
        if os.path.exists(input_file):
            with open(input_file, "r", encoding="utf-8") as file:
                urls = file.read().splitlines()

            for url in urls:
                print(f"Đang xử lý: {url}")
                get_content_and_save_to_file(driver, url, output_folder)
                time.sleep(1)  # Nghỉ 1 giây giữa các lần xử lý để tránh quá tải
        else:
            print(f"File {input_file} không tồn tại.")

    except Exception as e:
        print(f"Lỗi trong quá trình xử lý file: {e}")

    finally:
        driver.quit()

def main():
    input_file = "source/omni_doc_links.txt"
    output_folder = "omni"

    # Xử lý từng URL từ file và lưu nội dung vào tệp
    process_urls_from_file(input_file, output_folder)

if __name__ == "__main__":
    main()
