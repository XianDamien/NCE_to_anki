# 2_process_with_ai.py
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import json
import os
import time
from dotenv import load_dotenv
import concurrent.futures

# --- 配置 ---
load_dotenv() 
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

MAX_PROCESS_WORKERS = 5 
MAX_THREAD_WORKERS_FOR_DRAFTS = 10
MAX_RETRIES_PER_SENTENCE = 3

BOOK_TO_PROCESS = 2
RAW_DATA_DIR = os.path.join("raw_data", f"nce_book_{BOOK_TO_PROCESS}")
PROCESSED_DATA_DIR = os.path.join("processed_data", f"nce_book_{BOOK_TO_PROCESS}")

GENERATION_CONFIG = {"temperature": 0.4, "top_p": 1, "top_k": 1, "max_output_tokens": 8192}
FLASH_MODEL_NAME = "gemini-2.5-flash"
PRO_MODEL_NAME = "gemini-2.5-pro"

# ------------
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

# <<< 新增：为第一阶段（生成草稿）设计的Prompt >>>
prompt_for_draft = (
    "你是一位顶级的英语老师，专门为中文为母语、词汇量在1000-2000的初学者讲解新概念英语。你的讲解必须简明扼要、形象生动、全部使用中文。\n\n"
    "输出格式规则：\n"
    "1. 列表请直接使用 “1.”, “2.” 这样的数字前缀。\n\n"
    "1. 绝对不要使用任何Markdown或HTML标签（例如 `<b>`, `<br>` 等）。所有内容都必须是纯文本。\n"
    "请严格按照以下三部分内容和格式，为提供的句子生成学习笔记：\n"
    "要用到的讲解架构： 句型-词性：所有的简单句只有五种（即英语的五种句型）——由句子中的动词决定/由需要表达的意思决定。动词：表示状态或动作的词汇； 不及物动词vi.，及物动词vt.，系动词（is, seem, prove，表示状态），双宾语动词，需要宾补的动词等。S + V (主谓)**: The sun rises.S + V + O (主谓宾)**: I love English.S + V + P (主谓表)**: The story **is** true. (系动词)S + V + o + O (主谓双宾)**: My mom bought **me** a **book**.S + V + O + C (主谓宾补)**: I painted the wall **blue**."
    "1. 词汇与短语\n"
    "解释重点单词或词组的含义、词性(vt./vi./adj./adv.等)、发音难点和实用搭配。注意，2-5个重点即可，否则会给学习者带来不必要的文字和负担\n\n"
    "2. 句子结构分析\n"
    "这是最重要的部分，把所有的简单句都分析一下，过于简单的句子可以省略。请严格参考以下示例的思路和格式进行分析：\n"
    "--- 示例开始 ---\n"
    "句子: If you park your car in the wrong place, a traffic policeman will soon find it.\n"
    "分析:\n"
    "这是一个条件句，告诉我们一个动作（停车）会带来一个后果（被发现）。\n"
    "1. 这个句子的“主角”是后半部分 \"a traffic policeman will soon find it\" (警察会发现它)，这是一个“主谓宾”句型，清楚地说明了警察会做什么事。\n"
    "2. 前半部分 \"If you park your car...\" (如果你把车停在错误的地方) 是一个条件，像个“如果...就...”的开关，告诉我们“主角”事件在什么情况下会发生。\n"
    "3. 时态：主句用了将来时 (will find)，表示对未来的预测；条件部分用了现在时 (park) 代替将来时，这是if条件句的固定规则。\n"
    "--- 示例结束 ---\n"
    "3. 举一反三\n"
    "请提供1-2个与原句核心词汇或句型相似的、简单的例句，并附上中文翻译。\n"
    "------------------------------------------------------------------\n\n"
    "本课生词参考: {vocabulary}\n"
    "需要分析的句子:\n"
    "- 英文: \"{eng}\"\n"
    "- 中文: \"{chn}\"\n\n"
)

# <<< 新增：为第二阶段（精炼笔记）设计的Prompt >>>
prompt_for_refinement = (
    "你是一位顶级的英语教学编辑，你的任务是“精炼并关联”一份笔记草稿。\n\n"
    "核心原则：\n"
    "1. **精炼简洁**: 严格遵守“少即是多”，删除草稿中所有非必要的、重复的或过于基础的信息。最终笔记要简明扼要，不要给初学者造成负担。\n"
    "2. **深度关联**: 这是你的核心价值。请仔细阅读下面提供的“本课所有其他句子的笔记草稿”，如果当前句子的知识点（词汇/句型）与它们有关联，请用“这和我们之前遇到的...类似”或“注意区分...”等方式点明，帮助学生建立联系。\n"
    "3. **优化表达**: 用更生动、更易于理解的方式重写草稿，确保最终版本清晰、流畅。\n\n"
    "输出格式：纯文本，用数字前缀分点。绝对不要使用markdown格式！！！！！也不要使用\n的格式！！！！！注意精炼之后每一个句子卡片的解析的字数不能超过100字！！！！关联开始的时候只需要标注【关联点】即可\n"
    "------------------------------------------------------------------\n"
    "当前要精炼的句子:\n"
    "- 英文: \"{eng}\"\n"
    "- 中文: \"{chn}\"\n\n"
    "这份句子的笔记草稿（你需要修改和精炼它）:\n"
    "--- 草稿开始 ---\n"
    "{draft_note}\n"
    "--- 草稿结束 ---\n\n"
    "【参考1】本课所有内容的草稿 (供你通览全局，寻找远距离关联):\n"
    "{full_context}\n\n"
    "【参考2】本课已按顺序精炼过的笔记 (供你回顾近期内容，避免重复):\n"
    "{refined_context}\n"
    "--------------------------------\n"
    "--- 全文背景开始 ---\n"
    "{full_context}\n"
    "--- 全文背景结束 ---\n\n"
    "--- 已精炼笔记开始 ---\n"
    "{refined_context}\n"
    "--- 已精炼笔记结束 ---\n\n"
    "现在，请输出你对上面那份“笔记草稿”进行精炼和关联后的最终版本:输出示例：【核心词汇】 **have (vt.)**: “拥有，有”。这是英语中最基础的动词之一，表示所属关系。 【关联点】: 在本课中，我们还会遇到表示“属于”的 `belong to`。`have` 强调主动拥有，如 `We have an instrument.` (我们拥有一件乐器)。而 `It has belonged to our family for a long time.` (它属于我们家很久了) 则强调被动地“属于”某个群体或个人，并持续了一段时间。注意它们在表达“所属”时的不同侧重【句型解析】：主谓宾 (S + V + O)    `We` (主语 S) + `have` (谓语 V) + `an old musical instrument` (宾语 O)。 这是一个典型的“主谓宾”句型，直接说明“谁拥有什么”。`old` 和 `musical` 是形容词，它们修饰名词 `instrument`，具体描述了乐器的特征。【举一反三】：I have a new car. (我有一辆新车。) She has a big house. (她有一栋大房子。)\n"
)


def process_lesson_with_gemini(lesson_data):
    flash_model = genai.GenerativeModel(model_name=FLASH_MODEL_NAME, generation_config=GENERATION_CONFIG, safety_settings=safety_settings)
    pro_model = genai.GenerativeModel(model_name=PRO_MODEL_NAME, generation_config=GENERATION_CONFIG, safety_settings=safety_settings)
    
    # ... 分句逻辑 ...
    prompt_split = f"你的任务是将一段英文和其对应的中文翻译，一句对一句地精准配对。请严格按照“英文句子 | 中文句子”的格式输出...\n\n英文课文:\n{lesson_data['english']}\n\n中文译文:\n{lesson_data['chinese']}"
    try:
        response = flash_model.generate_content(prompt_split)
        if not response.parts: raise ValueError(f"分句API响应为空, 原因: {response.candidates[0].finish_reason}")
        sentence_pairs = [(p[0].strip(), p[1].strip()) for line in response.text.strip().split('\n') if '|' in line and len(p := line.split('|', 1)) == 2]
    except Exception as e:
        print(f"   - ❌ 调用Gemini分句出错，已中止: {e}"); return None
    if not sentence_pairs:
        print("   - ⚠️ 未能成功配对句子，处理中断。"); return None

    # --- 阶段1: 并行生成草稿 ---
    print(f"   - ✅ 智能分句完成，共 {len(sentence_pairs)} 句。开始并行生成草稿...")
    draft_notes = {}
    def generate_single_draft(sentence_pair):
        eng, chn = sentence_pair
        try:
            draft_prompt_filled = prompt_for_draft.format(eng=eng, chn=chn, vocabulary=lesson_data.get('vocabulary', ''))
            response = flash_model.generate_content(draft_prompt_filled)
            if not response.parts: raise ValueError(f"API响应为空, 原因: {response.candidates[0].finish_reason}")
            return eng, response.text.strip()
        except Exception as e:
            return eng, "草稿生成失败。"
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREAD_WORKERS_FOR_DRAFTS) as executor:
        results = executor.map(generate_single_draft, sentence_pairs)
        for eng, draft_note in results:
            draft_notes[eng] = draft_note

    # --- 阶段2: 串行精炼笔记 (使用双重上下文) ---
    print("   - ✅ 所有草稿生成完毕。开始使用Pro模型进行双重上下文精炼...")
    final_notes_data = []
    for i, (eng, chn) in enumerate(sentence_pairs):
        final_note = None
        for attempt in range(MAX_RETRIES_PER_SENTENCE):
            try:
                print(f"  - 正在精炼第 {i+1}/{len(sentence_pairs)} 句 (Pro模型, 尝试 {attempt + 1}/{MAX_RETRIES_PER_SENTENCE})...")
                draft_note = draft_notes.get(eng, "")
                
                # <<< 核心改动：同时构建两种上下文 >>>
                # 上下文1: 全文草稿 (用于全局视野)
                other_drafts = [f"- {o_eng}\n  草稿: {o_note}" for o_eng, o_note in draft_notes.items() if o_eng != eng]
                full_context = "\n".join(other_drafts)

                # 上下文2: 已精炼笔记 (用于滚动记忆)
                if not final_notes_data:
                    refined_context = "（这是本课第一句，尚无已精炼的笔记）"
                else:
                    context_lines = [f"- {item['english']}\n  最终笔记: {item['note']}" for item in final_notes_data]
                    refined_context = "\n".join(context_lines)

                # 将两种上下文都填充到Prompt中
                refinement_prompt_filled = prompt_for_refinement.format(
                    eng=eng, chn=chn, draft_note=draft_note, 
                    full_context=full_context, 
                    refined_context=refined_context
                )
                
                response = pro_model.generate_content(refinement_prompt_filled, request_options={"timeout": 300})
                if not response.parts: raise ValueError(f"API响应为空, 原因: {response.candidates[0].finish_reason}")
                
                final_note = response.text.strip()
                break
            except Exception as e:
                print(f"    - ⚠️ 精炼尝试失败: {e}")
                if attempt < MAX_RETRIES_PER_SENTENCE - 1:
                    print("    - 正在等待5秒后重试..."); time.sleep(5)
                else:
                    print(f"    - ❌ 已达最大重试次数，精炼失败。")
        
        if final_note is not None:
            final_notes_data.append({"english": eng, "chinese": chn, "note": final_note})
        else:
            print(f"   - ❌ 由于句子“{eng[:20]}...”精炼失败，本课({lesson_data['filename']})处理中止。")
            return None
        
        time.sleep(1)

    print(f"   - ✅ 整篇课文({lesson_data['filename']})所有句子精炼成功！")
    return final_notes_data

# ... process_single_file 和 main 函数保持不变 ...
def process_single_file(filename):
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
    else:
        return f"❌ 错误：子进程无法找到GOOGLE_API_KEY。"
    raw_filepath = os.path.join(RAW_DATA_DIR, filename)
    processed_filepath = os.path.join(PROCESSED_DATA_DIR, filename)
    if os.path.exists(processed_filepath):
        return f"🟡 {filename} 已成功处理，跳过。"
    print(f"🚀 开始处理文件: {filename}")
    with open(raw_filepath, 'r', encoding='utf-8') as f:
        lesson_data = json.load(f)
    lesson_data['filename'] = filename 
    anki_notes = process_lesson_with_gemini(lesson_data)
    if anki_notes:
        with open(processed_filepath, 'w', encoding='utf-8') as f:
            json.dump(anki_notes, f, ensure_ascii=False, indent=4)
        return f"✅ 文件 {filename} 处理成功，已保存。"
    else:
        return f"❌ 文件 {filename} 处理失败，将在下次运行时重试。"

def main():
    print(f"🚀 脚本2：并行处理启动 (使用 {MAX_PROCESS_WORKERS} 个进程) 🚀")
    if not GOOGLE_API_KEY:
        print("❌ 错误：未在.env文件中找到GOOGLE_API_KEY。"); return
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    raw_files = sorted([f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.json') and not f.startswith('.')])
    if not raw_files:
        print("🟡 在raw_data文件夹中没有找到要处理的文件。"); return
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_PROCESS_WORKERS) as executor:
        results = executor.map(process_single_file, raw_files)
        for result in results:
            print(result)
    print("\n🏁 所有文件处理完毕！请检查日志中是否有处理失败的文件。")

if __name__ == '__main__':
    main()