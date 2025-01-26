from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import csv
import time

# Hàm crawl dữ liệu từ URL
def crawl_website(start_url, visited_urls, writer):
    driver = webdriver.Chrome()
    driver.get(start_url)

    # Tránh crawl lại URL đã truy cập
    if start_url in visited_urls:
        driver.quit()
        return
    visited_urls.add(start_url)

    # Lấy tiêu đề và nội dung chính
    title = driver.title
    body = driver.find_element(By.TAG_NAME, 'body').text

    # Lưu vào CSV
    writer.writerow([start_url, title, body])

    # Tìm tất cả các link dẫn tới trang con
    links = driver.find_elements(By.TAG_NAME, 'a')
    for link in links:
        href = link.get_attribute('href')
        if href and start_url in href:  # Chỉ lấy link trong cùng domain
            crawl_website(href, visited_urls, writer)

    driver.quit()

# Tạo file CSV và bắt đầu crawl
with open("output.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["URL", "Title", "Content"])
    visited_urls = set()
    crawl_website("https://docs.omni.co/docs", visited_urls, writer)
