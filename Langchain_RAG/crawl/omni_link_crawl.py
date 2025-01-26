from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
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
            return False  # Không đạt ngưỡng kết thúc
        return True  # Đã tồn tại
    except Exception as e:
        print(f"Lỗi khi kiểm tra/thêm liên kết: {e}")
        return True

def click_and_expand_menus(driver, output_file):
    """Thực hiện click tất cả các thẻ <a> có class .menu__link--sublist-caret để mở rộng các menu collapse."""
    consecutive_existing_count = 0
    try:
        while True:
            # Tìm tất cả các thẻ <a> có class .menu__link--sublist-caret
            caret_links = driver.find_elements(By.CSS_SELECTOR, "a.menu__link--sublist-caret")

            # Nếu không còn thẻ nào cần click, thoát vòng lặp
            if not caret_links:
                break

            # Thực hiện click từng thẻ
            for link in caret_links:
                try:
                    ActionChains(driver).move_to_element(link).perform()  # Cuộn đến phần tử
                    link.click()

                    # Sau mỗi lần click, kiểm tra và thêm liên kết vào file
                    href = link.get_attribute("href")
                    if href:
                        is_existing = append_to_file_if_not_exists(output_file, href)
                        if is_existing:
                            consecutive_existing_count += 1
                        else:
                            consecutive_existing_count = 0

                        # Nếu vượt quá 200 lần liên tiếp, kết thúc chương trình
                        if consecutive_existing_count > 200:
                            print("Quá 200 lần kiểm tra liên tiếp, kết thúc chương trình.")
                            return

                except Exception as e:
                    print(f"Lỗi khi click: {e}")

    except Exception as e:
        print(f"Lỗi khi mở rộng menu: {e}")

def extract_sidebar_links(driver, base_url, output_file):
    """Trích xuất tất cả các liên kết từ <ul class="theme-doc-sidebar-menu menu__list"> và lưu vào file."""
    try:
        # Tìm thẻ ul với class cụ thể
        sidebar_menu = driver.find_element(By.CSS_SELECTOR, "ul.theme-doc-sidebar-menu.menu__list")
        links = sidebar_menu.find_elements(By.TAG_NAME, "a")

        # Thu thập tất cả các liên kết
        for link in links:
            href = link.get_attribute("href")
            if href:
                append_to_file_if_not_exists(output_file, urljoin(base_url, href))

    except Exception as e:
        print(f"Lỗi khi trích xuất liên kết: {e}")

def main():
    start_url = "https://docs.omni.co/docs"  # Thay bằng URL của bạn
    output_file = "source/omni_doc_links.txt"

    # Khởi tạo trình duyệt
    driver = webdriver.Chrome()

    try:
        # Mở URL bắt đầu
        driver.get(start_url)

        # Thực hiện mở rộng các menu collapse
        click_and_expand_menus(driver, output_file)

        # Gọi hàm trích xuất liên kết
        extract_sidebar_links(driver, start_url, output_file)

    except Exception as e:
        print(f"Lỗi trong quá trình chạy: {e}")

    finally:
        # Đóng trình duyệt
        driver.quit()

if __name__ == "__main__":
    main()
