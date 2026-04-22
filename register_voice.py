import sounddevice as sd
import soundfile as sf
import time

print("="*40)
print("🎙️ Alexa 专属声纹录入工具")
print("="*40)

name = input("👉 请输入要注册的人名（例如：李涵、张晓琴）：")
print(f"\n💡 请【{name}】准备，倒数 3 秒后开始录音...")
time.sleep(1)
print("3...")
time.sleep(1)
print("2...")
time.sleep(1)
print("1...")

print("\n🔴 录音开始！请连续说话 5 秒钟...")
fs = 16000
duration = 5
# 开始录音
recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
sd.wait()

# 统一命名格式：voice_名字.wav
filename = f"voice_{name}.wav"
sf.write(filename, recording, fs)

print(f"\n✅ 录音完成！已保存为核心声纹文件：{filename}")
print("下次启动 Alexa 系统时，她将自动认识这个人！")