from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.parse import urlparse
import os
import time

# Danh sách URL cần crawl
URLS_TO_CRAWL = [
    "https://omni.co/platform",
    "https://omni.co/business-intelligence",
    "https://omni.co/embedded-analytics",
    "https://omni.co/integrations",
    "https://omni.co/customer-case-studies",
    "https://omni.co/customer-support",
    "https://omni.co/about",
    "https://omni.co/security",
    "https://omni.co/demos",
]

OUTPUT_FOLDER = "omni"  # Thư mục lưu kết quả

# Tạo thư mục nếu chưa tồn tại
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Cấu hình Selenium (Headless mode để chạy nhanh hơn)
chrome_options = Options()
chrome_options.add_argument("--headless")  # Chạy không giao diện
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--window-size=1920x1080")

# Khởi tạo WebDriver
driver = webdriver.Chrome(options=chrome_options)

def sanitize_filename(url):
    """Chuyển đổi URL thành tên file hợp lệ."""
    parsed_url = urlparse(url)
    filename = parsed_url.path.replace("/", "_").strip("_")
    return filename if filename else "index"

def save_content_to_file(url, content):
    """Lưu nội dung từ URL vào file trong thư mục omni."""
    try:
        filename = sanitize_filename(url) + ".txt"
        filepath = os.path.join(OUTPUT_FOLDER, filename)
        with open(filepath, "w", encoding="utf-8") as file:
            file.write(content)
        print(f"✅ Saved: {filepath}")
    except Exception as e:
        print(f"❌ Error saving {url}: {e}")

def crawl_page_content(url):
    """Truy cập URL, lấy nội dung trong thẻ <main> và lưu vào file."""
    try:
        driver.get(url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "main")))

        main_content = driver.find_element(By.TAG_NAME, "main").text.strip()
        if main_content:
            save_content_to_file(url, main_content)
        else:
            print(f"⚠️ No content found in <main> for {url}")

    except Exception as e:
        print(f"❌ Error processing {url}: {e}")

# Chạy crawler trên danh sách URL
for url in URLS_TO_CRAWL:
    print(f"🔍 Crawling: {url}")
    crawl_page_content(url)
    time.sleep(1)  # Nghỉ 1 giây để tránh bị chặn

# Đóng trình duyệt
driver.quit()
print("✅ Done!")
