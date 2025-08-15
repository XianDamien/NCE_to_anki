# 3_import_to_anki.py
import requests
import json
import os
import time
import base64
from google.cloud import texttospeech
from dotenv import load_dotenv
from google.oauth2 import service_account

# --- 配置 ---
load_dotenv()
GOOGLE_KEY_FILE_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
BOOK_TO_IMPORT = 2
ANKI_DECK_NAME = "新概念英语2"
ANKI_MODEL_NAME = "背课文"
ANKI_AUDIO_FIELD_NAME = "发音"
ANKI_CONNECT_URL = "http://127.0.0.1:8765"
PROCESSED_DIR = os.path.join("processed_data", f"nce_book_{BOOK_TO_IMPORT}")
IMPORTED_DIR = os.path.join(PROCESSED_DIR, "imported")
TTS_LANGUAGE_CODE = "en-US"
TTS_VOICE_NAME = "en-US-Standard-J"
# ------------

def ankiconnect_invoke(action, **params):
    """一个通用的AnkiConnect调用函数"""
    payload = json.dumps({"action": action, "version": 6, "params": params})
    try:
        response = requests.post(ANKI_CONNECT_URL, data=payload.encode('utf-8'))
        response.raise_for_status()
        response_data = response.json()
        if response_data.get('error'):
            raise Exception(response_data['error'])
        return response_data.get('result')
    except Exception as e:
        print(f"❌ AnkiConnect调用 '{action}' 失败: {e}")
        return None

# <<< 新增：查找已存在卡片的函数 >>>
def find_existing_notes(deck_name, english_sentences):
    """在Anki中查找指定牌组中已存在的卡片"""
    print("🔍 正在检查Anki中是否已存在卡片...")
    query = f'"deck:{deck_name}" ('
    query += ' or '.join([f'"英文:{sentence}"' for sentence in english_sentences])
    query += ')'
    
    note_ids = ankiconnect_invoke('findNotes', query=query)
    if note_ids is None: return set() # 查询失败

    if not note_ids:
        print("   - 未发现重复卡片。")
        return set()

    notes_info = ankiconnect_invoke('notesInfo', notes=note_ids)
    if not notes_info: return set()

    existing_sentences = {info['fields']['英文']['value'] for info in notes_info if '英文' in info['fields']}
    print(f"   - 发现 {len(existing_sentences)} 张已存在的卡片，将跳过它们。")
    return existing_sentences

# <<< 核心改动：增加了重试逻辑 >>>
def generate_and_upload_tts(client, text_to_speak, lesson_num, note_index):
    """使用已有的TTS客户端生成音频并上传，失败时会重试"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"    🔊 正在为句子 '{text_to_speak[:20]}...' 生成音频 (尝试 {attempt + 1}/{max_retries})...")
            synthesis_input = texttospeech.SynthesisInput(text=text_to_speak)
            voice = texttospeech.VoiceSelectionParams(language_code=TTS_LANGUAGE_CODE, name=TTS_VOICE_NAME)
            audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
            response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

            audio_content_b64 = base64.b64encode(response.audio_content).decode('utf-8')
            unique_filename = f"nce_tts_{BOOK_TO_IMPORT}_{lesson_num}_{note_index}.mp3"
            
            result = ankiconnect_invoke("storeMediaFile", filename=unique_filename, data=audio_content_b64)
            if result:
                print(f"    ✅ 音频成功上传到Anki: {result}")
                return result
            else:
                # storeMediaFile 失败通常问题不大，但还是记录一下
                print(f"    ❌ 上传音频到Anki失败。")
                return None
        except Exception as e:
            # 只对特定的“服务已禁用”的临时错误进行重试
            if "SERVICE_DISABLED" in str(e):
                print(f"    - ⚠️ TTS服务暂时不可用 (传播延迟)，将在5秒后重试...")
                time.sleep(5)
                continue # 进入下一次重试
            else:
                print(f"    ❌ 生成或上传TTS时发生不可恢复的错误: {e}")
                return None # 其他错误直接失败
    print(f"    ❌ 在 {max_retries} 次尝试后，TTS生成仍然失败。")
    return None

def add_notes_to_anki(notes_data, lesson_num):
    print(f"📤 正在处理 {len(notes_data)} 条笔记...")
    
    try:
        credentials = service_account.Credentials.from_service_account_file(GOOGLE_KEY_FILE_PATH)
        tts_client = texttospeech.TextToSpeechClient(credentials=credentials)
    except Exception as e:
        print(f"    ❌ 创建TTS客户端时发生严重错误: {e}"); return False

    # <<< 核心改动：先找出所有已存在的卡片 >>>
    all_english_in_lesson = [note['english'] for note in notes_data]
    existing_sentences = find_existing_notes(ANKI_DECK_NAME, all_english_in_lesson)

    notes_to_add = []
    for i, note in enumerate(notes_data):
        eng_sentence = note['english']

        # 如果句子已存在，则跳过
        if eng_sentence in existing_sentences:
            print(f"    ↪️ 已跳过重复卡片: '{eng_sentence[:20]}...'")
            continue

        anki_audio_filename = generate_and_upload_tts(tts_client, eng_sentence, lesson_num, i + 1)
        if anki_audio_filename is None:
            print(f"    ❌ 由于TTS生成失败，文件 {lesson_num} 的导入已中止。")
            return False

        audio_field_content = f"[sound:{anki_audio_filename}]"
        lesson_tag = f"NCE{BOOK_TO_IMPORT}-{lesson_num}"
        note_payload = {
            "deckName": ANKI_DECK_NAME, "modelName": ANKI_MODEL_NAME,
            "fields": { "英文": eng_sentence, "中文": note['chinese'], "笔记": note['note'].replace('\n', '<br>'), ANKI_AUDIO_FIELD_NAME: audio_field_content },
            "tags": [f"NCE_Book{BOOK_TO_IMPORT}", lesson_tag]
        }
        notes_to_add.append(note_payload)
    
    if not notes_to_add:
        print("✅ 本课所有卡片均已存在，无需新增。")
        return True # 认为这是一个“成功”的操作

    print(f"📤 准备批量导入 {len(notes_to_add)} 张新卡片...")
    result = ankiconnect_invoke("addNotes", notes=notes_to_add)
    if result is not None:
        success_count = sum(1 for r in result if r is not None)
        print(f"🎉 Anki导入完成！成功: {success_count}张。")
        return True
    return False

def main():
    print("🚀 脚本3：将处理好的数据导入Anki (增量更新与智能重试版) 🚀")
    if not GOOGLE_KEY_FILE_PATH:
        print("❌ 错误：未在.env文件中找到GOOGLE_APPLICATION_CREDENTIALS。"); return
    if not ankiconnect_invoke('deckNames'):
        print("❌ AnkiConnect未运行或连接失败。"); return
    if ANKI_DECK_NAME not in ankiconnect_invoke('deckNames'):
        print(f"ℹ️ 牌组 '{ANKI_DECK_NAME}' 不存在，正在创建...")
        ankiconnect_invoke('createDeck', deck=ANKI_DECK_NAME)
    else:
        print(f"✅ 牌组 '{ANKI_DECK_NAME}' 已确认存在。")

    os.makedirs(IMPORTED_DIR, exist_ok=True)
    processed_files = sorted([f for f in os.listdir(PROCESSED_DIR) if f.endswith('.json') and not f.startswith('.')])
    if not processed_files: print("🟡 在processed_data文件夹中没有找到要导入的文件。"); return
    
    all_successful = True
    for filename in processed_files:
        filepath = os.path.join(PROCESSED_DIR, filename)
        print("\n" + "="*50)
        print(f"📄 正在处理文件: {filename}")
        try:
            lesson_num = int(filename.split('_')[1].split('.')[0])
        except (IndexError, ValueError):
            lesson_num = "Unknown"
        with open(filepath, 'r', encoding='utf-8') as f:
            anki_notes = json.load(f)
        
        if add_notes_to_anki(anki_notes, lesson_num):
            try:
                os.rename(filepath, os.path.join(IMPORTED_DIR, filename))
                print(f"✅ 已将 {filename} 移动到 imported 文件夹。")
            except Exception as e:
                print(f"   - 移动文件时出错: {e}")
        else:
            all_successful = False
            print(f"⚠️ 文件 {filename} 处理失败或中止，已跳过。")

    print("\n🏁 所有已处理文件导入完毕！")
    if not all_successful:
        print("   - 注意: 部分文件在处理过程中失败，请检查上面的日志。")

if __name__ == '__main__':
    main()