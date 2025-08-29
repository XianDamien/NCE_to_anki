import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import json
import os
import time
from dotenv import load_dotenv
import concurrent.futures
from tqdm import tqdm

# --- 配置 ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# 并行处理配置
MAX_THREAD_WORKERS = 10
MAX_RETRIES_PER_TASK = 3
RETRY_DELAY_SECONDS = 5

# --- 文件路径配置 (MODIFIED) ---
# Reads individual files FROM this directory
INPUT_DIR = os.path.join("processed_data", "split_articles")
# Saves individual files TO this new directory
OUTPUT_DIR = os.path.join("processed_data", "final_notes")

# AI模型配置
FLASH_MODEL_NAME = "gemini-2.5-flash"
PRO_MODEL_NAME = "gemini-2.5-pro"
GENERATION_CONFIG_DRAFT = {"temperature": 0.4}
GENERATION_CONFIG_REFINE = {"temperature": 0.5, "response_mime_type": "application/json"}

SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

# --- Prompts (Unchanged) ---
PROMPT_FOR_DRAFT_CET6 = """
你是一名高效的英语助教，任务是为一句六级翻译真题生成一份简明扼要的笔记草稿，为后续的专家精讲做准备。

**要求:**
1.  **核心词汇**: 提炼1-2个关键中文表达及其精准的英文翻译。
2.  **翻译亮点**: 指出该句翻译中的一个核心技巧或结构特点（如：语序调整为前重后轻、分词作伴随状语、抽象名词具体化等）。
3.  **简洁精炼**: 草稿总字数控制在50-80字。

**句子信息:**
- 中文: `{source_sentence_cn}`
- 英文参考: `{reference_translation_en}`

**你的草稿:**
"""

PROMPT_FOR_REFINEMENT_CET6 = """
你是一位顶级的大学英语六级（CET-6）翻译辅导专家，深谙出题规律和高分技巧。你的核心任务是基于我提供的**同一考试主题下的全文语境**、**初步笔记草稿**以及**十大翻译心法**，为每一句话生成一份专家级的深度精讲笔记。

**背景知识 - CET-6出题规律 & 十大翻译心法:**
1.  **出题规律**: 同一次考试的三套翻译题通常围绕一个宏大主题（如：中国科技成就、传统文化、社会发展），因此文章间存在大量的词汇和句式复现。
2.  **十大翻译心法**:
    - **结构类**: 技巧一(三步走), 技巧二(前重后轻), 技巧三(结构三剑客: as, with, -ing)。
    - **词句类**: 技巧四(同义合并), 技巧五(同义替换), 技巧七(副词取舍), 技巧八(动词隐藏)。
    - **思维类**: 技巧六(舍概括取具体), 技巧九(化抽象为具体), 技巧十(形容词立体化)。

**核心指令:**
请严格按照以下步骤，为每个句子生成最终笔记：

1.  **全局关联分析**:
    *   审视“全文语境”，识别并利用本次考试的**宏大主题**。
    *   对比“所有句子的笔记草稿”，找出贯穿全文的**主题词汇**（如“养老”、“文化”、“科技”）和**高频句式**（如“...是...之一”、“随着...”、“不仅...而且...”）。

2.  **单句深度精炼 (应用十大心法)**:
    *   在草稿基础上，将分析融入“十大翻译心法”。例如，看到语序调整，要点明这是**技巧二“前重后轻”**的应用；看到分词短语，要指出这是**技巧三“结构三剑客”**的妙用。
    *   **【关联】是强制要求**: 如果当前句的知识点（词汇/句式/技巧）与本段其他句子有关，必须使用`【关联】`标签明确指出。例如：“【关联】此处的‘make... more accessible’呼应了技巧九‘化抽象为具体’，与第三句处理‘养老服务’的思路一致。”
    *   **严格遵循PRD三段式结构**:
        *   `【核心词汇与表达解析】`: 精讲核心词汇，并关联全文主题词。
        *   `【句法分析与翻译技巧】`: 结合“十大心法”剖析句子结构和翻译策略。
        *   `【可替换表达与拓展】`: 提供高质量的同义替换，体现技巧五“千变万化”。

**输入信息:**

**1. 全文语境 (同一考试主题下的三篇文章之一):**
- 中文全文: `{full_context_cn}`
- 英文全文: `{full_context_en}`

**2. 所有句子的笔记草稿:**{all_drafts_text}

**输出指令:**
你的最终输出**必须**是一个**纯粹的、不含任何其他文本的、格式正确的JSON数组**。数组中的每个对象代表一个句子的最终笔记，要求内容精炼深刻，单条笔记字数在100字以内，格式如下：
`[ {{ "sentence_index": 1, "final_note": "【核心词汇...】...【关联】..." }}, {{ "sentence_index": 2, "final_note": "..." }}]`

请现在开始生成这个JSON数组。
"""

def generate_draft_for_sentence(args):
    """(阶段一) 为单个句子生成笔记草稿"""
    sentence_data, model = args
    prompt = PROMPT_FOR_DRAFT_CET6.format(**sentence_data)
    for attempt in range(MAX_RETRIES_PER_TASK):
        try:
            response = model.generate_content(prompt)
            return {"sentence_index": sentence_data["sentence_index"], "draft_note": response.text.strip()}
        except Exception as e:
            if attempt == MAX_RETRIES_PER_TASK - 1:
                return {"sentence_index": sentence_data["sentence_index"], "draft_note": f"草稿生成失败: {e}"}
            time.sleep(RETRY_DELAY_SECONDS)

def main():
    print("🚀 模块 2b: 开始生成AI笔记 (两阶段)... 🚀")

    if not GOOGLE_API_KEY:
        print("❌ 错误：未在.env文件中找到GOOGLE_API_KEY。")
        return
    genai.configure(api_key=GOOGLE_API_KEY)

    # --- Create output directory ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Get list of files to process ---
    if not os.path.exists(INPUT_DIR):
        print(f"❌ 错误：找不到输入目录 {INPUT_DIR}。请先运行模块 2a。")
        return
    
    files_to_process = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.json')])
    if not files_to_process:
        print(f"🟡 在目录 {INPUT_DIR} 中没有找到要处理的JSON文件。")
        return
        
    print(f"✅ 找到 {len(files_to_process)} 个已分句的文件准备处理。")

    flash_model = genai.GenerativeModel(FLASH_MODEL_NAME, generation_config=GENERATION_CONFIG_DRAFT, safety_settings=SAFETY_SETTINGS)
    pro_model = genai.GenerativeModel(PRO_MODEL_NAME, generation_config=GENERATION_CONFIG_REFINE, safety_settings=SAFETY_SETTINGS)
    
    success_count, skipped_count = 0, 0

    # --- Main loop iterates over FILENAMES ---
    for filename in tqdm(files_to_process, desc="处理所有真题文件"):
        input_filepath = os.path.join(INPUT_DIR, filename)
        output_filepath = os.path.join(OUTPUT_DIR, filename)

        # Skip if already processed
        if os.path.exists(output_filepath):
            continue

        with open(input_filepath, 'r', encoding='utf-8') as f:
            item = json.load(f)
        
        print(f"\n📄 正在处理: {item['id']} - {item['title']}")
        
        # --- 阶段一: 并行生成草稿 ---
        tasks = [(sentence, flash_model) for sentence in item["sentences"]]
        drafts = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREAD_WORKERS) as executor:
            results = list(tqdm(executor.map(generate_draft_for_sentence, tasks), total=len(tasks), desc="  - (1/2) 生成草稿 (Flash)", leave=False))
            for res in results:
                drafts[res["sentence_index"]] = res["draft_note"]
        
        # --- 阶段二: 整体精炼与关联 ---
        print("  - (2/2) 正在精炼与关联 (Pro)...")
        all_drafts_text = "\n\n".join([f"--- 句子 {i} ---\n{drafts.get(i, 'N/A')}" for i in sorted(drafts.keys())])
        
        refinement_prompt_filled = PROMPT_FOR_REFINEMENT_CET6.format(
            full_context_cn=item["full_context_cn"],
            full_context_en=item["full_context_en"],
            all_drafts_text=all_drafts_text
        )

        final_notes_list = None
        for attempt in range(MAX_RETRIES_PER_TASK):
            try:
                response = pro_model.generate_content(refinement_prompt_filled, request_options={"timeout": 300})
                final_notes_list = json.loads(response.text)
                break 
            except Exception as e:
                print(f"    - ⚠️ 精炼尝试 {attempt + 1}/{MAX_RETRIES_PER_TASK} 失败: {e}")
                if attempt < MAX_RETRIES_PER_TASK - 1:
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    print(f"    - ❌ 跳过此题的笔记生成。")
        
        if final_notes_list:
            final_notes_map = {note["sentence_index"]: note["final_note"] for note in final_notes_list}
            for sentence in item["sentences"]:
                sentence["ai_note"] = final_notes_map.get(sentence["sentence_index"], "最终笔记生成失败")
            
            # --- START OF CHANGE ---
            # Create a new dictionary containing only the 'sentences' key and its data
            output_data = {"sentences": item["sentences"]}
            
            # Save the new, simplified data structure to the file
            with open(output_filepath, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            # --- END OF CHANGE ---
            
            success_count += 1
            print("  - ✅ 笔记生成成功。")
        else:
            skipped_count += 1

    print(f"\n🏁 AI笔记生成全部完成！成功处理 {success_count} 个文件, 跳过/失败 {skipped_count} 个。结果已保存至: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()