from selenium import webdriver
from selenium.webdriver.common.by import By
from urllib.parse import urljoin
import os

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

def main():
    start_url = "https://help.sigmacomputing.com/docs/get-around-in-sigma"  # URL Sigma
    output_file = "source/sigma_doc_crawl.txt"

    # Khởi tạo trình duyệt
    driver = webdriver.Chrome()

    try:
        # Mở URL bắt đầu
        driver.get(start_url)

        # Gọi hàm trích xuất liên kết
        extract_sidebar_links(driver, start_url, output_file)

    except Exception as e:
        print(f"Lỗi trong quá trình chạy: {e}")

    finally:
        # Đóng trình duyệt
        driver.quit()

if __name__ == "__main__":
    main()
