import os
import certifi
import requests
import time
import sounddevice as sd
import soundfile as sf
import numpy as np
import asyncio
import edge_tts
import pygame
import json
import threading
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from openai import OpenAI
from openwakeword.model import Model 
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
import queue
import re 
import glob
import csv 

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# ==========================================
# 1. 基础配置与网页推送服务
# ==========================================
os.environ["SSL_CERT_FILE"] = certifi.where()
API_KEY = "" # 这里填入你的API key
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1" 
MODEL_NAME = "qwen-omni-turbo" 

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
app = FastAPI()
connected_websockets = []
server_loop = None
last_ui_state = ""

# 会议与实验室模式全局变量
MEETING_MODE = False
meeting_content = []
MASTER_VOICEPRINT = None

LAB_MODE = False
LAB_FILE_NAME = ""

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_websockets.append(websocket)
    print("✅ 网页显示面板已连接")
    try:
        while True: await websocket.receive_text()
    except WebSocketDisconnect:
        connected_websockets.remove(websocket)

def notify_ui(state, text=""):
    global server_loop, last_ui_state
    if not server_loop: return
    if state == last_ui_state and state in ["idle", "listening", "thinking"]:
        return
        
    last_ui_state = state
    message = json.dumps({"state": state, "text": text})
    for ws in connected_websockets:
        asyncio.run_coroutine_threadsafe(ws.send_text(message), server_loop)

def start_web_server():
    global server_loop
    server_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(server_loop)
    config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="error")
    server = uvicorn.Server(config)
    server_loop.run_until_complete(server.serve())

threading.Thread(target=start_web_server, daemon=True).start()

# ==========================================
# 2. 模型初始化
# ==========================================
pygame.mixer.init()
print("正在加载 AI 核心模型 (含声纹识别)...")
sense_model = AutoModel(
    model="iic/SenseVoiceSmall", trust_remote_code=True, remote_code="./model.py",
    vad_model="fsmn-vad", vad_kwargs={"max_single_segment_time": 30000}, device="cpu", 
)
WAKE_WORD = "alexa" 
oww_model = Model(wakeword_models=[WAKE_WORD], inference_framework="onnx")

sv_pipeline = pipeline(Tasks.speaker_verification, model='damo/speech_campplus_sv_zh-cn_16k-common')

def get_ip_location():
    try:
        print("🌍 正在侦测所在城市...")
        res = requests.get(
            "http://ip-api.com/json/?lang=zh-CN", 
            timeout=10, 
            proxies={"http": None, "https": None}
        ).json()
        city = res.get("city", "未知城市")
        print(f"📍 定位成功：{city}")
        return city
    except Exception as e:
        print(f"⚠️ IP定位失败，原因：{e}")
        return "未知城市"

chat_history = []
MAX_HISTORY = 10 
CURRENT_CITY = get_ip_location()
print(f"✅ 系统就绪！当前城市：{CURRENT_CITY}")

# ==========================================
# 独立发声引擎 (双线程预加载流水线)
# ==========================================
text_queue = queue.Queue()      # 管道1：装文字
playback_queue = queue.Queue()  # 管道2：装下载好的 MP3
stop_speaking_flag = False
chunk_counter = 0

def tts_downloader_worker():
    global stop_speaking_flag, chunk_counter
    import re
    while True:
        item = text_queue.get()
        if item is None: continue
        
        text, current_voice = item 
        if stop_speaking_flag:
            text_queue.task_done()
            continue
            
        clean_text = text.replace("-", "").replace("*", "").replace("#", "").strip()
        if not clean_text:
            text_queue.task_done()
            continue
            
        chunk_file = f"chunk_{chunk_counter}.mp3"
        chunk_counter += 1
        
        try:
            communicate = edge_tts.Communicate(text, voice=current_voice)
            asyncio.run(communicate.save(chunk_file))
            if not stop_speaking_flag:
                playback_queue.put(chunk_file)
        except Exception as e:
            print(f"⚠️ 下载语音跳过: {e}")
            
        text_queue.task_done()

def tts_player_worker(mic_stream):
    global stop_speaking_flag
    while True:
        chunk_file = playback_queue.get()
        if chunk_file is None: continue
        
        if stop_speaking_flag:
            try: os.remove(chunk_file)
            except: pass
            playback_queue.task_done()
            continue
            
        try:
            pygame.mixer.music.load(chunk_file)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                audio_chunk, _ = mic_stream.read(1280)
                oww_model.predict(audio_chunk.flatten())
                if any(list(oww_model.prediction_buffer[m])[-1] > 0.5 for m in oww_model.prediction_buffer):
                    pygame.mixer.music.stop()
                    oww_model.reset()
                    stop_speaking_flag = True
                    break
                    
            pygame.mixer.music.stop()
            try: pygame.mixer.music.unload()
            except: pass
            
            try: os.remove(chunk_file)
            except: pass
        except Exception as e:
            pass
            
        playback_queue.task_done()

# ==========================================
# 3. 工具与录音逻辑
# ==========================================
def dynamic_record(stream, filename="temp_voice.wav", samplerate=16000):
    SILENCE_THRESHOLD = 400  
    SILENCE_DURATION = 1.5   
    WAIT_TIMEOUT = 5.0       
    MAX_RECORD_TIME = 20.0   
    
    frames = []
    chunk_size = 1280
    chunks_per_second = samplerate / chunk_size
    
    max_silent_chunks = int(SILENCE_DURATION * chunks_per_second)
    max_wait_chunks = int(WAIT_TIMEOUT * chunks_per_second)
    max_total_chunks = int(MAX_RECORD_TIME * chunks_per_second)
    
    silent_chunks = 0
    total_chunks = 0
    has_spoken = False
    
    notify_ui("listening")
    print("\n🎤 [录音中] 开始聆听...")
    
    while True:
        audio_chunk, _ = stream.read(chunk_size)
        frames.append(audio_chunk)
        total_chunks += 1
        volume = np.max(np.abs(audio_chunk))
        
        if volume > SILENCE_THRESHOLD:
            has_spoken = True
            silent_chunks = 0  
        else:
            silent_chunks += 1
            
        if not has_spoken and total_chunks > max_wait_chunks:
            print("💤 [已取消] 未检测到声音")
            notify_ui("idle")
            return False
            
        if has_spoken and silent_chunks > max_silent_chunks:
            print("⏹️ [已结束] 侦测到说话完毕")
            break
            
        if total_chunks > max_total_chunks:
            print("⚠️ [已截断] 达到最长录音限制")
            break
            
    audio_data = np.concatenate(frames, axis=0)
    sf.write(filename, audio_data, samplerate)
    return True

# ==========================================
# 4. 会议模式引擎
# ==========================================
def process_meeting_chunk(audio_file):
    global meeting_content
    
    res = sense_model.generate(input=audio_file, cache={}, language="auto", use_itn=True)
    text = rich_transcription_postprocess(res[0]["text"]).strip()
    
    if not text: return False
    
    exit_keywords = ["结束会议", "关闭会议", "停止会议", "退出会议", "开完会了"]
    if any(keyword in text for keyword in exit_keywords):
        finalize_meeting()
        return True

    speaker = "参会者" 
    highest_score = 0.0
    
    for name, voice_path in VOICE_DB.items():
        try:
            result = sv_pipeline([voice_path, audio_file])
            score = result["scores"][0] if isinstance(result.get("scores"), list) else result.get("score", 0.0)
            
            if score > highest_score and score > 0.45:
                highest_score = score
                speaker = name
        except Exception as e:
            pass
    
    entry = f"[{speaker}]: {text}"
    meeting_content.append(entry)
    
    print(f"📝 实时记录 -> {entry} (匹配度: {highest_score:.2f})")
    notify_ui("user_said", entry)
    
    return False

def finalize_meeting():
    global MEETING_MODE, meeting_content
    MEETING_MODE = False
    full_text = "\n".join(meeting_content)
    
    print("\n🧠 Alexa 正在总结会议纪要...")
    notify_ui("thinking")
    
    sys_prompt = "你是一个专业的会议记录秘书。请将以下的语音转写记录整理成一份结构清晰的会议纪要，必须包含：1.会议时间 2.核心议题总结 3.后续待办事项。直接输出排版好的纯文本即可。"
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME, 
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": full_text}]
        )
        final_note = response.choices[0].message.content
        
        filename = f"会议纪要_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(final_note)
            
        print(f"\n✅ 会议纪要已成功保存至本地: {filename}\n")
        text_queue.put(("会议模式已结束，我已经为您整理好了会议纪要，并生成了本地记事本文件。", "zh-CN-XiaoxiaoNeural"))
    except Exception as e:
        print(f"❌ 生成会议纪要失败: {e}")
        text_queue.put(("抱歉，整理会议纪要时发生了网络错误。", "zh-CN-XiaoxiaoNeural"))
    
    meeting_content = []

# ==========================================
# 5. 实验室模式引擎
# ==========================================

def manage_csv_data(action, file_name, headers=None, data=None):
    # 🌟 【核心修复】：不再乱猜桌面路径，直接在当前代码目录下建一个专属文件夹
    base_dir = os.path.join(os.getcwd(), "实验记录表格")
    os.makedirs(base_dir, exist_ok=True) # 如果文件夹不存在，自动创建
    file_path = os.path.join(base_dir, f"{file_name}.csv")
    
    try:
        if action == "create":
            # encoding='utf-8-sig' 保证用 Excel 打开绝对不会乱码
            with open(file_path, mode='w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                if headers: writer.writerow(headers)
                if data: writer.writerows(data)
            return True
        elif action == "append":
            with open(file_path, mode='a', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                if data: writer.writerows(data)
            return True
    except Exception as e:
        print(f"CSV写入异常: {e}")
        return False

def process_lab_chunk(audio_file):
    global LAB_MODE, LAB_FILE_NAME, LAB_HEADERS
    
    res = sense_model.generate(input=audio_file, cache={}, language="zh", use_itn=True)
    text = rich_transcription_postprocess(res[0]["text"]).strip()
    if not text: return False

    if any(k in text for k in ["退出实验", "结束实验", "停止记录", "做完了"]):
        LAB_MODE = False
        text_queue.put(("实验室模式已结束。所有数据已安全存入本地表格。", "zh-CN-XiaoxiaoNeural"))
        return True

    print(f"🔬 实验室监听中: {text}")
    notify_ui("user_said", f"[口述记录]: {text}")

    prompt = (
        f"你是一个实验数据提取专家。当前正在记录的文件是: {LAB_FILE_NAME}。\n"
        f"当前的表头结构是: {LAB_HEADERS} (注意：第一列通常是时间，系统会自动补充，你只需要提取后面的数据！)。\n"
        f"用户刚说了一句话: '{text}'。\n"
        f"请判断其中是否包含实验数据。如果有，请按表头顺序提取，并严格输出为纯 JSON 数组格式，绝对不要加任何 markdown 标记。\n"
        f"如果没有数据，请回复'NULL'。\n"
        f"【示例】：严格回复 [\"2欧\", \"1A\", \"2V\"]"
    )
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME, 
            messages=[{"role": "user", "content": prompt}]
        )
        extraction = response.choices[0].message.content.strip()
        
        # 🌟 核心修复 1：强力清洗大模型喜欢乱加的 markdown 符号
        clean_extraction = extraction.replace("```json", "").replace("```", "").strip()
        
        if "[" in clean_extraction and "]" in clean_extraction:
            import json
            # 使用标准的 json 解析，比 eval 安全且稳定一百倍
            data_row = json.loads(clean_extraction)
            
            if len(data_row) < len(LAB_HEADERS):
                final_row = [time.strftime('%H:%M:%S')] + data_row
            else:
                final_row = data_row
                
            manage_csv_data("append", LAB_FILE_NAME, data=[final_row])
            print(f"📊 成功写入表格: {final_row}")
            
    except Exception as e:
        # 🌟 核心修复 2：绝不静默报错！把错误原因和大模型的原生输出打印出来
        print(f"⚠️ 数据提取或写入失败: {e}")
        print(f"🤖 大模型原始输出内容是: {extraction}")
        
    return False

# ==========================================
# 6. 核心对话链路
# ==========================================
def chat_pipeline(audio_file, stream):
    # 🌟 【核心修复】：把 LAB_HEADERS 的全局声明统一放到函数最顶端！
    global chat_history, stop_speaking_flag, MEETING_MODE, LAB_MODE, LAB_FILE_NAME, LAB_HEADERS
    stop_speaking_flag = False 
    
    notify_ui("thinking")
    res = sense_model.generate(input=audio_file, cache={}, language="auto", use_itn=True)
    raw_text = res[0]["text"]
    
    user_text = rich_transcription_postprocess(raw_text).strip()
    if not user_text: return False
    
    # 【拦截器 1】：会议模式
    if "开启会议模式" in user_text or "开启会议" in user_text:
        MEETING_MODE = True
        print("\n🚀 [系统状态切换] -> 已进入会议记录模式")
        text_queue.put(("好的，会议模式已开启，我将全程进行声纹分离并记录接下来的对话。", "zh-CN-XiaoxiaoNeural"))
        text_queue.join() 
        return False

    # 【拦截器 2】：实验室模式 (智能提取表名和表头)
    if "开启实验室模式" in user_text or "开始实验" in user_text:
        LAB_MODE = True
        print("\n🚀 [系统状态切换] -> 正在解析实验建表需求...")
        
        setup_prompt = (
            f"用户刚才说：'{user_text}'。\n"
            f"请提取用户想要的“表格名称”和“表头数组”。\n"
            f"要求：1.必须且只能返回纯 JSON 格式。2.如果没有指定表名，默认为'实验数据'。3.如果没有指定表头，默认为 [\"时间\", \"记录内容\"]。\n"
            f"示例：{{\"file_name\": \"光电效应\", \"headers\": [\"时间\", \"电流\", \"电压\"]}}"
        )
        try:
            setup_res = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": setup_prompt}]
            )
            import json
            json_str = setup_res.choices[0].message.content.replace("```json", "").replace("```", "").strip()
            setup_info = json.loads(json_str)
            
            base_name = setup_info.get("file_name", "实验数据")
            LAB_HEADERS = setup_info.get("headers", ["时间", "记录内容"])
            
            if base_name in ["实验数据", "实验记录"]:
                LAB_FILE_NAME = f"{base_name}_{time.strftime('%m%d_%H%M')}"
            else:
                LAB_FILE_NAME = base_name
                
        except Exception as e:
            print(f"⚠️ 解析表头失败，使用默认配置: {e}")
            LAB_FILE_NAME = f"实验数据_{time.strftime('%m%d_%H%M')}"
            LAB_HEADERS = ["时间", "记录内容"]

        # 创建表格
        manage_csv_data("create", LAB_FILE_NAME, headers=LAB_HEADERS)
        
        reply_text = f"好的，实验室模式已启动。已为您创建表格：{LAB_FILE_NAME}。表头包含：{', '.join(LAB_HEADERS)}。请随时口述数据。"
        text_queue.put((reply_text, "zh-CN-XiaoxiaoNeural"))
        text_queue.join() 
        return False
    
    # 提取情绪与语言
    emotion = "平静"
    if "<|HAPPY|>" in raw_text: emotion = "开心"
    elif "<|ANGRY|>" in raw_text: emotion = "生气"
    elif "<|SAD|>" in raw_text: emotion = "难过"

    lang_name = "中文"
    voice_model = "zh-CN-XiaoxiaoNeural" 
    
    if "<|jp|>" in raw_text or "<|ja|>" in raw_text:
        lang_name, voice_model = "日语", "ja-JP-NanamiNeural"
    elif "<|ko|>" in raw_text:
        lang_name, voice_model = "韩语", "ko-KR-SunHiNeural"
    elif "<|en|>" in raw_text:
        lang_name, voice_model = "英语", "en-US-AriaNeural"
    elif "<|yue|>" in raw_text:
        lang_name, voice_model = "粤语", "zh-HK-HiuMaanNeural"

    print(f"🗣️ 用户 ({lang_name} | {emotion}): {user_text}")
    notify_ui("user_said", user_text) 

    current_time_str = time.strftime('%Y年%m月%d日 %H:%M:%S')
    sys_prompt = f"你是极客专属的AI女管家Alexa。当前时间：{current_time_str}。当前城市：{CURRENT_CITY}。系统检测到主人刚才使用了【{lang_name}】，且情绪是【{emotion}】。请务必使用流利的【{lang_name}】给予最自然、贴心的简短回应！"
    
    messages = [{"role": "system", "content": sys_prompt}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_text})

    try:
        notify_ui("typing_start", "")
        response = client.chat.completions.create(model=MODEL_NAME, messages=messages, stream=True)
        
        sentence_buffer = ""
        full_reply = ""
        
        for chunk in response:
            if stop_speaking_flag: break
                
            delta = chunk.choices[0].delta.content
            if delta:
                sentence_buffer += delta
                full_reply += delta
                notify_ui("typing_word", delta)
                
                if any(punct in delta for punct in ['。', '！', '？', '；', '\n', '.', '!', '?']):
                    if len(sentence_buffer.strip()) > 1:
                        import re 
                        current_voice = voice_model 
                        
                        if re.search(r'[\u3040-\u30ff\u31f0-\u31ff]', sentence_buffer): 
                            current_voice = "ja-JP-NanamiNeural"
                        elif re.search(r'[\uac00-\ud7af]', sentence_buffer): 
                            current_voice = "ko-KR-SunHiNeural"  
                        elif re.search(r'[a-zA-Z]', sentence_buffer) and not re.search(r'[\u4e00-\u9fa5]', sentence_buffer):
                            current_voice = "en-US-AriaNeural"   
                        
                        text_queue.put((sentence_buffer, current_voice)) 
                        sentence_buffer = "" 
                        
        if sentence_buffer.strip() and not stop_speaking_flag:
            import re
            current_voice = voice_model
            if re.search(r'[\u3040-\u30ff\u31f0-\u31ff]', sentence_buffer): 
                current_voice = "ja-JP-NanamiNeural"
            elif re.search(r'[\uac00-\ud7af]', sentence_buffer): 
                current_voice = "ko-KR-SunHiNeural"
            elif re.search(r'[a-zA-Z]', sentence_buffer) and not re.search(r'[\u4e00-\u9fa5]', sentence_buffer):
                current_voice = "en-US-AriaNeural"
                
            text_queue.put((sentence_buffer, current_voice))
            
        notify_ui("typing_end", "")
        
        chat_history.append({"role": "user", "content": user_text})
        chat_history.append({"role": "assistant", "content": full_reply})
        if len(chat_history) > 10: chat_history = chat_history[-10:]
            
        text_queue.join()
        playback_queue.join()
        
        if stop_speaking_flag:
            print("\n💥 [系统] 侦测到主人打断！光速切入聆听状态...")
            return True  
        else:
            return False 

    except Exception as e:
        print(f"Error: {e}")
        return False
    
# ==========================================
# 7. 主循环
# ==========================================
if __name__ == "__main__":
    stream = sd.InputStream(samplerate=16000, blocksize=1280, channels=1, dtype='int16')
    stream.start()

    threading.Thread(target=tts_downloader_worker, daemon=True).start()
    threading.Thread(target=tts_player_worker, args=(stream,), daemon=True).start()

    print("\n" + "="*40)
    print("🚀 Alexa 终极融合架构已启动")
    print("="*40)
    
    VOICE_DB = {}
    print("\n🔍 正在扫描本地声纹特征...")
    for wav_file in glob.glob("voice_*.wav"):
        name = wav_file.replace("voice_", "").replace(".wav", "")
        VOICE_DB[name] = wav_file
        print(f"✅ 成功载入身份: {name}")

    if not VOICE_DB:
        print("⚠️ 未检测到声纹库！所有人都将被识别为【参会者】。请运行 register_voice.py 录入。")
        
    MASTER_VOICEPRINT = "master_voice.wav"
    print("✅ 身份识别已就绪。")

    temp_file = "temp_voice.wav"
    need_record = False
    
    try:
        while True:
            # ----------------------------------------
            # 模式分支 A：会议记录模式
            # ----------------------------------------
            if MEETING_MODE:
                notify_ui("listening")
                chunk_file = "meeting_chunk.wav"
                audio_chunk, _ = stream.read(16000 * 10)
                sf.write(chunk_file, audio_chunk, 16000)
                process_meeting_chunk(chunk_file)
                continue  

           # ----------------------------------------
            # 模式分支 B：实验室记录模式 (智能声控截断)
            # ----------------------------------------
            if LAB_MODE:
                # 实验室模式下，不需要唤醒词，直接调用我们强大的智能录音机
                # 只要你说话，它就录；你停顿了1.5秒，它就瞬间截断送去提取！
                success = dynamic_record(stream, filename="lab_chunk.wav")
                if success:
                    process_lab_chunk("lab_chunk.wav")
                continue

            # ----------------------------------------
            # 模式分支 C：日常语音唤醒待命模式
            # ----------------------------------------
            if not need_record:
                notify_ui("idle")
                audio_chunk, _ = stream.read(1280)
                oww_model.predict(audio_chunk.flatten())
                if any(list(oww_model.prediction_buffer[m])[-1] > 0.5 for m in oww_model.prediction_buffer):
                    need_record = True
                    oww_model.reset()
            
            if need_record:
                success = dynamic_record(stream, filename=temp_file)
                if success:
                    need_record = chat_pipeline(temp_file, stream)
                else:
                    need_record = False
                    
    except KeyboardInterrupt: 
        print("\n👋 退出系统")
    finally: 
        stream.stop()