# 1_scrape_lessons.py

import requests
from bs4 import BeautifulSoup
import json
import os
import time

# --- 配置 ---
BOOK_TO_SCRAPE = 3 # <<< 改动：现在目标是第三册 >>>
TOTAL_LESSONS = 60 # <<< 改动：第三册共60课 >>>
BASE_URL = "http://www.newconceptenglish.com/index.php"
OUTPUT_DIR = os.path.join("raw_data", f"nce_book_{BOOK_TO_SCRAPE}")
MAX_ATTEMPTS = 3 # 每篇课文最多尝试次数
# ------------

def get_text_until_next_h3(start_tag):
    """一个健壮的辅助函数，用于抓取从一个<h3>标签开始，到下一个<h3>标签之前的所有文本内容"""
    if not start_tag: return ""
    
    content_parts = []
    for element in start_tag.next_siblings:
        if element.name == 'h3':
            break
        text = element.get_text(" ", strip=True) 
        if text:
            content_parts.append(text)
    return " ".join(content_parts).strip()

def scrape_nce_lesson(lesson_url):
    """
    核心爬虫函数：为NCE第三册定制，并抓取所有必要字段。
    返回一个包含内容的字典，或在失败时返回None。
    """
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(lesson_url, headers=headers, timeout=20)
        response.raise_for_status()
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # <<< 核心改动：字段名与我们AI脚本所需的一致 >>>
        content = {'english': '', 'chinese': '', 'vocabulary': ''}
        
        # 1. 提取英文课文
        h3_english = soup.find('h3', string='新概念英语－课文')
        content['english'] = get_text_until_next_h3(h3_english)

        # 2. 提取中文翻译 (新增逻辑)
        h3_chinese = soup.find('h3', string='新概念英语－翻译')
        content['chinese'] = get_text_until_next_h3(h3_chinese)
            
        # 3. 提取单词和短语
        h3_vocab = soup.find('h3', string='新概念英语－单词和短语')
        # 第三册的生词格式可能与第二册不同，用 \n 分隔更合适
        content['vocabulary'] = get_text_until_next_h3(h3_vocab).replace(" ", "\n")

        # 只要核心内容不为空，就认为本次爬取是初步成功的
        if content['english'] and content['chinese']:
            return content
        else:
            print(f"    - 警告: 未能完整抓取到 {lesson_url} 的英文或中文内容。")
            return None
    except Exception:
        # 隐藏具体错误，让调用者处理重试
        return None

def process_single_lesson(lesson_num, book_num):
    """
    处理单篇课文的完整流程：爬取 -> 在线校验 -> 失败重试 -> 保存。
    这是脚本的核心“自修正”逻辑。
    """
    output_filepath = os.path.join(OUTPUT_DIR, f"lesson_{lesson_num:03d}.json")
    if os.path.exists(output_filepath):
        print(f"✅ lesson_{lesson_num:03d}.json 已存在，跳过。")
        return True

    lesson_url = f"{BASE_URL}?id=course-{book_num}-{lesson_num:03d}"
    print(f"\n--- 正在处理 Lesson {lesson_num} ---")

    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"  - 第 {attempt}/{MAX_ATTEMPTS} 次尝试...")
        
        # 首次爬取
        main_data = scrape_nce_lesson(lesson_url)
        if not main_data:
            print("    - ❌ 爬取失败，稍后重试...")
            time.sleep(2)
            continue

        # 在线校验 (通过再次爬取进行比对)
        time.sleep(1) 
        verify_data = scrape_nce_lesson(lesson_url)
        if not verify_data:
            print("    - ❌ 无法获取校验数据，重试...")
            time.sleep(2)
            continue
            
        # 内容比对
        if (main_data['english'] == verify_data['english'] and
            main_data['chinese'] == verify_data['chinese'] and
            main_data['vocabulary'] == verify_data['vocabulary']):
            
            print("    - ✅ 数据一致性校验通过！内容完整。")
            try:
                with open(output_filepath, 'w', encoding='utf-8') as f:
                    json.dump(main_data, f, ensure_ascii=False, indent=4)
                print(f"    - 💾 已成功保存到: {output_filepath}")
                return True
            except IOError as e:
                print(f"    - ❌ 文件保存失败: {e}")
                return False

        else:
            print("    - ❌ 数据不一致，可能存在网络不稳定性。正在重试...")
            time.sleep(2)

    print(f"  - ❌ Lesson {lesson_num} 在 {MAX_ATTEMPTS} 次尝试后仍无法稳定抓取，已跳过。")
    return False

def main():
    """主执行函数，负责调度整个爬取流程"""
    print(f"🚀 脚本1：新概念英语第{BOOK_TO_SCRAPE}册 自我修正爬虫 🚀")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    failed_lessons = []
    for lesson_num in range(1, TOTAL_LESSONS + 1):
        success = process_single_lesson(lesson_num, BOOK_TO_SCRAPE)
        if not success:
            failed_lessons.append(lesson_num)
    
    print("\n" + "#"*50)
    print("📊 所有任务已完成。最终报告：")
    if not failed_lessons:
        print(f"✅🎉🎉 完美！所有 {TOTAL_LESSONS} 篇课文都已成功抓取并通过校验！")
    else:
        print(f"❌ 注意：有 {len(failed_lessons)} 篇课文在多次尝试后依然失败。")
        print(f"失败的课文编号: {failed_lessons}")
    print("#"*50 + "\n")


if __name__ == '__main__':
    main()