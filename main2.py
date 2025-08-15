import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
import json
import time
import threading
# ================================= CONFIGURATION =================================
# 1. 填入你的Google Gemini API密钥
GOOGLE_API_KEY = "AIzaSyBoDQQLCJGe7Kz6cMdw2p1hGREMTauhwhM"

# 2. AnkiConnect的配置
ANKI_DECK_NAME = "新概念英语2"  # 目标牌组
ANKI_MODEL_NAME = "背课文" # 笔记类型名称
ANKI_CONNECT_URL = "http://127.0.0.1:8765"

# 3. Gemini模型的配置
GENERATION_CONFIG = {
    "temperature": 0.5,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 4096,
}
# ===============================================================================

# ===============================================================================

# --- STAGE 1: WEB SCRAPER (ULTIMATE ROBUST VERSION) ---
def scrape_nce_lesson(lesson_url):
    """从给定的URL爬取新概念课文、翻译和生词（已兼容新概念2册所有已知HTML结构）"""
    print(f"🕸️  正在爬取: {lesson_url}")
    try:
        response = requests.get(lesson_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=20)
        response.raise_for_status()
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')
        content = {'english': '', 'chinese': '', 'vocabulary': ''}

        # --- 通用提取函数 (核心改进) ---
        def extract_content_after_h3(h3_text):
            h3_tag = soup.find('h3', string=h3_text)
            if not h3_tag: return ""
            
            parts = []
            # 遍历h3之后的所有同级节点
            for sibling in h3_tag.find_next_siblings():
                if sibling.name == 'h3': # 遇到下一个标题就停止
                    break
                # 如果是<p>标签，直接获取文本
                if sibling.name == 'p':
                    parts.append(sibling.get_text(strip=True))
                # 如果是div容器，就进入容器查找<p>
                elif sibling.name == 'div':
                    for p_tag in sibling.find_all('p'):
                        parts.append(p_tag.get_text(strip=True))

            # 如果上述方法没找到内容，尝试在h3的父级容器里找
            if not parts and h3_tag.parent.name == 'div':
                 for p_tag in h3_tag.parent.find_all('p'):
                     # 确保只添加h3标题之后的p标签
                     if p_tag.find_previous_sibling('h3') == h3_tag:
                         parts.append(p_tag.get_text(strip=True))

            return '\n\n'.join(parts)

        content['english'] = extract_content_after_h3('新概念英语－课文')
        content['chinese'] = extract_content_after_h3('新概念英语－翻译')

        # --- 提取单词（逻辑保持，因为它比较稳定） ---
        h3_vocab = soup.find('h3', string='新概念英语－单词和短语')
        if h3_vocab:
            parent_container = h3_vocab.parent
            for br in parent_container.find_all('br'): br.replace_with('\n')
            full_text = parent_container.get_text(separator='\n', strip=True)
            vocab_text = full_text.split('新概念英语－单词和短语')[-1].strip()
            content['vocabulary'] = vocab_text.split('新概念英语－翻译')[0].strip()

        if content['english'] and content['chinese']:
            print("✅ 爬取成功。")
            # print(f"   - [调试] 中文内容片段: {content['chinese'][:30]}...") # 取消注释以进行调试
            return content
        else:
            print(f"❌ 爬取失败: 未找到英文课文或中文翻译。")
            print(f"   - [调试] 英文找到: {'是' if content['english'] else '否'}")
            print(f"   - [调试] 中文找到: {'是' if content['chinese'] else '否'}")
            return None
    except Exception as e:
        print(f"❌ 爬取或解析时出错: {e}")
        return None

# --- STAGE 2: AI PROCESSING ---
def process_lesson_with_gemini(lesson_content):
    # ... (此函数无需修改，保持上一版本的即可)
    print("🤖 开始使用Gemini处理内容...")
    model = genai.GenerativeModel(model_name="gemini-1.5-flash", generation_config=GENERATION_CONFIG)
    stop_spinner = False
    def spinner():
        chars = "-\\|/"
        while not stop_spinner:
            for char in chars:
                print(f'\r   - 正在等待Gemini响应 {char}', end='', flush=True)
                time.sleep(0.1)
        print('\r' + ' '*30 + '\r', end='')
    
    prompt_split = f"请将以下英文课文和中文译文，一句对一句地配对起来。\n严格按照 \"英文句子 | 中文句子\" 的格式输出，每一对占一行。\n\n英文课文:\n{lesson_content['english']}\n\n中文译文:\n{lesson_content['chinese']}"
    sentence_pairs = []
    spinner_thread = threading.Thread(target=spinner)
    try:
        print("   - 正在发送【智能分句】请求...")
        spinner_thread.start()
        response = model.generate_content(prompt_split, request_options={"timeout": 120})
        stop_spinner = True
        spinner_thread.join()
        for line in response.text.strip().split('\n'):
            if '|' in line:
                eng, chn = line.split('|', 1)
                sentence_pairs.append((eng.strip(), chn.strip()))
        print(f"   - ✅ 智能分句完成，共 {len(sentence_pairs)} 句。")
    except Exception as e:
        stop_spinner = True
        spinner_thread.join()
        print(f"\n   - ❌ 调用Gemini进行分句时出错 (可能是超时): {e}")
        return None

    all_notes_data = []
    total = len(sentence_pairs)
    for i, (eng, chn) in enumerate(sentence_pairs):
        prompt_note = f"你是一位专业的英语老师，为词汇量约1000-2000的初学者讲解新概念英语。讲解风格需简明扼要、形象生动。\n本课生词: {lesson_content['vocabulary']}\n请为这个句子生成笔记：英文: \"{eng}\" 中文: \"{chn}\"\n笔记必须用\"1. 2. 3.\"分点说明，包含: 1. **句子结构分析** 2. **核心词汇/短语** 3. **语法点睛**\n请直接输出笔记内容。"
        stop_spinner = False
        spinner_thread = threading.Thread(target=spinner)
        try:
            print(f"   - 🧠 正在为第 {i+1}/{total} 句生成笔记...")
            spinner_thread.start()
            note_response = model.generate_content(prompt_note, request_options={"timeout": 120})
            stop_spinner = True
            spinner_thread.join()
            all_notes_data.append({"english": eng, "chinese": chn, "note": note_response.text.strip()})
            print(f"   - ✅ 第 {i+1}/{total} 句笔记生成成功。")
        except Exception as e:
            stop_spinner = True
            spinner_thread.join()
            print(f"\n   - ❌ 为句子生成笔记时出错 (可能是超时): {e}")
            all_notes_data.append({"english": eng, "chinese": chn, "note": "笔记生成失败。"})
        time.sleep(1)
    print("✅ Gemini处理完成。")
    return all_notes_data

# --- STAGE 3: ANKI IMPORT (ENHANCED ERROR REPORTING) ---
def add_notes_to_anki(notes_data):
    """使用AnkiConnect的addNotes动作批量导入卡片（已修复NoneType并增强错误报告）"""
    print("📤 准备通过AnkiConnect批量导入卡片...")
    notes_to_add = [{"deckName": ANKI_DECK_NAME, "modelName": ANKI_MODEL_NAME, "fields": {"中文": note['chinese'], "英文": note['english'], "笔记": note['note'].replace('\n', '<br>')}, "tags": ["NCE_AutoImport", "NCE_Book2"]} for note in notes_data]
    payload = json.dumps({"action": "addNotes", "version": 6, "params": {"notes": notes_to_add}})
    try:
        response = requests.post(ANKI_CONNECT_URL, data=payload.encode('utf-8'))
        response_data = response.json()
        
        if response_data.get('error') is not None:
             print(f"❌ Anki导入失败 (严重错误): {response_data['error']}")
        else:
            result_list = response_data.get('result')
            if result_list is not None:
                success_count = sum(1 for r in result_list if isinstance(r, int))
                errors = [r for r in result_list if not isinstance(r, int)]
                print(f"🎉 Anki导入完成！成功: {success_count}张。")
                if errors:
                    # 只显示第一条详细错误，避免刷屏
                    print(f"   - 出现 {len(errors)} 个问题，例如: '{errors[0]}'")
            else:
                print("   - AnkiConnect未返回有效结果，可能所有卡片都已存在或发生未知错误。")

    except Exception as e:
        print(f"❌ 发送数据到Anki时发生错误: {e}")

# ... (check_ankiconnect 和 main 函数保持不变) ...
def check_ankiconnect():
    try:
        requests.get(ANKI_CONNECT_URL, timeout=3)
        return True
    except requests.exceptions.RequestException:
        return False

def main():
    print("🚀 启动新概念全自动导入工作流 🚀")
    
    BOOK_TO_SCRAPE = 2
    START_LESSON = 1
    END_LESSON = 96
    
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
    except Exception as e:
        print(f"❌ 初始化Gemini失败: {e}")
        return
    if not check_ankiconnect():
        print("❌ AnkiConnect未运行。请先启动Anki桌面版。")
        return

    print(f"📚 准备处理《新概念英语》第 {BOOK_TO_SCRAPE} 册，从第 {START_LESSON} 课到第 {END_LESSON} 课...")
    base_url = "http://www.newconceptenglish.com/index.php?id=course"
    lesson_urls = [f"{base_url}-{BOOK_TO_SCRAPE}-{lesson:03d}" for lesson in range(START_LESSON, END_LESSON + 1)]
    
    for url in lesson_urls:
        print("\n" + "="*60)
        lesson_data = scrape_nce_lesson(url)
        if not lesson_data:
            print(f"跳过此URL，进入下一课...")
            time.sleep(2)
            continue
        
        anki_notes = process_lesson_with_gemini(lesson_data)
        if not anki_notes:
            print(f"Gemini处理失败，跳过此课...")
            continue

        add_notes_to_anki(anki_notes)
        print(f"该课程处理完毕，暂停15秒以防API请求过快...")
        time.sleep(15)

    print("\n🏁 所有指定课程已处理完毕！")


if __name__ == '__main__':
    main()