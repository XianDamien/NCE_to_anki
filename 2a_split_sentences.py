# 2a_split_sentences.py

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import json
import os
import time
from dotenv import load_dotenv
from tqdm import tqdm
import re

# --- 配置 ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# 重试配置
MAX_RETRIES_PER_ITEM = 3
RETRY_DELAY_SECONDS = 5

# 文件路径配置
RAW_DATA_PATH = os.path.join("raw_data", "cet6_raw_data.json")
SPLIT_DATA_PATH = os.path.join("processed_data", "cet6_split_data.json") # 输出文件

# AI模型配置 (分句任务用Flash即可)
MODEL_NAME = "gemini-2.5-flash" 
GENERATION_CONFIG = {"temperature": 0.0, "max_output_tokens": 4096}
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

# --- 专为分句设计的Prompt ---
PROMPT_FOR_SPLITTING = """
你的任务是将一段中文文本和其对应的英文翻译，精准地拆分成一一对应的句子对。

**核心规则:**
1.  **以英文为准**: 严格以英文文本的句子结束符（. ?!）作为切分基准。英文的一个完整句子必须对应一行输出。
2.  **匹配中文**: 将中文文本进行必要的、最小化的切分或调整，使其在语义上与对应的英文句子对齐。
3.  **严格格式**: 你的输出必须严格遵循以下格式，每行代表一个句子对，不得有任何额外解释：
    `句子序号 | 中文内容 | 英文内容`

**示例:**
输入中文: "中国的家庭观念与其文化传统密切相关。家庭和睦被视为头等大事。"
输入英文: "China's concept of family is closely related to its cultural traditions. Family harmony is regarded as a top priority."
输出:
1 | 中国的家庭观念与其文化传统密切相关。 | China's concept of family is closely related to its cultural traditions.
2 | 家庭和睦被视为头等大事。 | Family harmony is regarded as a top priority.

---
**待处理文本:**

**中文段落:**
{text_cn}

**英文段落:**
{text_en}

**请严格按格式输出:**
"""

def split_text_into_sentences_advanced(text_cn, text_en, model):
    """
    使用AI将中英文段落拆分为带索引的句子对。
    如果失败，返回None。
    """
    prompt = PROMPT_FOR_SPLITTING.format(text_cn=text_cn, text_en=text_en)
    for attempt in range(MAX_RETRIES_PER_ITEM):
        try:
            response = model.generate_content(prompt)
            if not response.parts:
                raise ValueError(f"API响应为空, 原因: {response.candidates[0].finish_reason}")
            
            text_output = response.text.strip()
            sentence_pairs = []
            lines = text_output.split('\n')
            
            # 校验输出格式是否基本正确
            if not lines or '|' not in lines[0]:
                 raise ValueError(f"输出格式不正确，不包含'|': {text_output[:100]}")

            for line in lines:
                parts = line.split('|', 2)
                if len(parts) == 3:
                    index_str = parts[0].strip()
                    cn_sent = parts[1].strip()
                    en_sent = parts[2].strip()
                    
                    # 再次校验数据有效性
                    if index_str.isdigit() and cn_sent and en_sent:
                        sentence_pairs.append({
                            "sentence_index": int(index_str),
                            "source_sentence_cn": cn_sent,
                            "reference_translation_en": en_sent
                        })
                    else: # 如果某一行解析失败，则认为整个分句失败
                        raise ValueError(f"解析行失败: {line}")
                else:
                    raise ValueError(f"行不符合'索引|中|英'格式: {line}")
            
            if not sentence_pairs:
                raise ValueError("成功解析后，句子对列表为空。")

            return sentence_pairs # 成功，返回列表

        except Exception as e:
            print(f"   - ⚠️ 分句尝试 {attempt + 1}/{MAX_RETRIES_PER_ITEM} 失败: {e}")
            if attempt < MAX_RETRIES_PER_ITEM - 1:
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                print(f"   - ❌ 已达最大重试次数，分句失败。")
                return None # 最终失败，返回None
    return None

# 2a_split_sentences.py -> main() 函数的修正版

def main():
    """主执行函数"""
    print("🚀 模块 2a: 开始智能分句... 🚀")
    
    # 确保输出目录存在
    os.makedirs("processed_data", exist_ok=True)
    
    genai.configure(api_key=GOOGLE_API_KEY)
    
    try:
        with open(RAW_DATA_PATH, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        print(f"✅ 成功加载 {len(raw_data)} 套翻译真题。")
    except FileNotFoundError:
        print(f"❌ 错误: 找不到原始数据文件 {RAW_DATA_PATH}。请确保文件存在。")
        return
    except json.JSONDecodeError:
        print(f"❌ 错误: 原始数据文件 {RAW_DATA_PATH} 格式不正确。")
        return

    model = genai.GenerativeModel(MODEL_NAME, generation_config=GENERATION_CONFIG, safety_settings=SAFETY_SETTINGS)
    
    final_split_data, skipped_count = [], 0
    
    # --- 修改点: 使用 enumerate 为数据添加索引 ---
    for index, item in enumerate(tqdm(raw_data, desc="处理所有真题")):
        
        # --- 修改点: 安全地获取id和title，如果不存在则使用索引生成 ---
        item_id = item.get('id', f"cet6_{index + 1:03d}") 
        item_title = item.get('title', f"真题_{index + 1}")
        
        print(f"\n📄 正在分句: {item_id} - {item_title}") # 现在这个print语句可以安全执行
        
        sentence_pairs = split_text_into_sentences_advanced(item['source_text_cn'], item['reference_translation_en'], model)
        
        if sentence_pairs:
            final_split_data.append({
                "id": item_id, 
                "title": item_title, 
                "full_context_cn": item['source_text_cn'],
                "full_context_en": item['reference_translation_en'], 
                "sentences": sentence_pairs
            })
        else:
            skipped_count += 1
            print(f"   - ❗ {item_id} - {item_title} 处理失败，已跳过。")
            
    with open(SPLIT_DATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_split_data, f, ensure_ascii=False, indent=2)
        
    print(f"\n🏁 智能分句全部完成！成功: {len(final_split_data)}, 失败跳过: {skipped_count}。结果保存至: {SPLIT_DATA_PATH}")

if __name__ == '__main__':
    main()