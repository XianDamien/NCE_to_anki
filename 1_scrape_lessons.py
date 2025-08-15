# 1_scrape_lessons.py

import requests
from bs4 import BeautifulSoup
import json
import os
import time

# --- 配置 ---
BOOK_TO_SCRAPE = 2
TOTAL_LESSONS = 96
BASE_URL = "http://www.newconceptenglish.com/index.php"
OUTPUT_DIR = os.path.join("raw_data", f"nce_book_{BOOK_TO_SCRAPE}")
MAX_ATTEMPTS = 3 # 每篇课文最多尝试次数
# ------------

def scrape_nce_lesson(lesson_url):
    """
    核心爬虫函数：仅负责从URL获取内容，不包含打印逻辑。
    返回一个包含内容的字典，或在失败时返回None。
    """
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(lesson_url, headers=headers, timeout=20)
        response.raise_for_status()
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')
        
        content = {'english': '', 'chinese': '', 'vocabulary': ''}
        h3_english = soup.find('h3', string='新概念英语－课文')
        if h3_english and h3_english.find_next_sibling('p'):
            content['english'] = h3_english.find_next_sibling('p').get_text(strip=True)

        h3_chinese = soup.find('h3', string='新概念英语－翻译')
        if h3_chinese and h3_chinese.find_next_sibling('p'):
            content['chinese'] = h3_chinese.find_next_sibling('p').get_text(strip=True)
            
        h3_vocab = soup.find('h3', string='新概念英语－单词和短语')
        if h3_vocab:
            vocab_parts = []
            for sibling in h3_vocab.next_siblings:
                if sibling.name == 'h3': break
                if isinstance(sibling, str):
                    cleaned_text = sibling.strip()
                    if cleaned_text: vocab_parts.append(cleaned_text)
            content['vocabulary'] = '\n'.join(vocab_parts)
        
        # 只要核心内容不为空，就认为本次爬取是初步成功的
        if content['english'] and content['chinese']:
            return content
        else:
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
        
        # 1. 首次爬取
        print("    - 正在爬取主要数据...")
        main_data = scrape_nce_lesson(lesson_url)
        if not main_data:
            print("    - ❌ 爬取失败，稍后重试...")
            time.sleep(2) # 等待2秒再重试
            continue

        # 2. 在线校验 (通过再次爬取进行比对)
        print("    - 正在爬取校验数据以进行比对...")
        time.sleep(1) # 友好访问
        verify_data = scrape_nce_lesson(lesson_url)
        if not verify_data:
            print("    - ❌ 无法获取校验数据，重试...")
            time.sleep(2)
            continue
            
        # 3. 内容比对
        if (main_data['english'] == verify_data['english'] and
            main_data['chinese'] == verify_data['chinese'] and
            main_data['vocabulary'] == verify_data['vocabulary']):
            
            print("    - ✅ 数据一致性校验通过！内容完整。")
            try:
                with open(output_filepath, 'w', encoding='utf-8') as f:
                    json.dump(main_data, f, ensure_ascii=False, indent=4)
                print(f"    - 💾 已成功保存到: {output_filepath}")
                return True # 成功处理，返回True
            except IOError as e:
                print(f"    - ❌ 文件保存失败: {e}")
                return False # 保存失败，也算失败

        else:
            print("    - ❌ 数据不一致，可能存在网络不稳定性。正在重试...")
            time.sleep(2)

    # 如果所有尝试都失败了
    print(f"  - ❌ Lesson {lesson_num} 在 {MAX_ATTEMPTS} 次尝试后仍无法稳定抓取，已跳过。")
    return False # 所有尝试都失败了，返回False

def main():
    """主执行函数，负责调度整个爬取流程"""
    print("🚀 脚本1：新概念英语自我修正爬虫 🚀")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    failed_lessons = []
    for lesson_num in range(1, TOTAL_LESSONS + 1):
        success = process_single_lesson(lesson_num, BOOK_TO_SCRAPE)
        if not success:
            failed_lessons.append(lesson_num)
    
    # --- 最终报告 ---
    print("\n" + "#"*50)
    print("📊 所有任务已完成。最终报告：")
    if not failed_lessons:
        print("✅🎉🎉 完美！所有 {TOTAL_LESSONS} 篇课文都已成功抓取并通过校验！")
    else:
        print(f"❌ 注意：有 {len(failed_lessons)} 篇课文在多次尝试后依然失败。")
        print(f"失败的课文编号: {failed_lessons}")
        print("建议您检查网络或稍后重新运行脚本，程序会自动尝试这些失败的课文。")
    print("#"*50 + "\n")


if __name__ == '__main__':
    main()