# Alexa 智能机器听觉系统：桌面级具身智能管家
![Architecture](https://img.shields.io/badge/Architecture-Streaming_Full--Duplex-007ACC?style=flat-square)
![LLM](https://img.shields.io/badge/LLM-Qwen--Omni--Turbo-6151A6?style=flat-square)
![ASR](https://img.shields.io/badge/ASR-SenseVoice-FF6B6B?style=flat-square)
![Voiceprint](https://img.shields.io/badge/Voiceprint-speech__campplus-FF9800?style=flat-square)
![Wake Word](https://img.shields.io/badge/Wake_Word-openwakeword-E91E63?style=flat-square)
![Framework](https://img.shields.io/badge/Framework-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)

本项目是一个基于流式全双工架构的桌面级智能语音助手。通过集成高精度 ASR、动态 LLM 路由以及多线程 TTS 预加载技术，系统实现了无缝的自然语言交互，并针对“科研数据编录”与“智能会议协作”两大高频专业场景进行了深度适配与落地。

## 🌟 核心技术亮点

* **全双工流式交互**：采用多线程异步流水线架构，打破传统串行处理瓶颈，实现“边思考、边合成、边播报”的极速响应体验。
* **语义级实时打断**：在语音播报期间保持麦克风持续监听，支持通过唤醒词“Alexa”毫秒级中断当前播报并切换至聆听状态。
* **语种感知**：基于 SenseVoiceSmall 模型，系统可支持多语种（中、英、日、韩、粤）的闭环识别与回复。
* **参数幻觉深度治理**：针对 LLM 在工具调用中的参数幻觉（如地理位置偏见、数据虚构），通过在工具描述中注入“强规则约束”与“代码级后处理清洗”，确保了数据的绝对准确性。

## 🛠️ 三大共生模式

### 1. 普通问答模式 (Normal Mode)
* **滑动窗口记忆**：通过 `MAX_HISTORY=10` 的滑动窗口机制，系统具备稳定的短期对话记忆，能够处理复杂的指代消解与话题切换。
* **实时信息查询**：内置地理位置侦测，支持查询实时天气、时间等实用信息。

### 2. 实验记录模式 (Lab Mode)
* **无接触式数据编录**：专门针对实验室科研场景，支持通过口令一键建表。利用 Function Calling 技术，将语音指令精准映射为结构化数据并存入 CSV。
* **智能静音截断 (VAD)**：基于 1.5s 静音阈值实现自动切分，研究人员无需重复唤醒即可连续报数录入。

### 3. 会议协作模式 (Meeting Mode)
* **实时声纹分离**：基于阿里达摩院 `speech_campplus` 声纹提取模型，实现 1:N 动态比对，自动标记不同发言人身份。
* **智能纪要提炼**：会议结束后，调用 Qwen-omni-turbo 自动提炼核心议题与待办事项，生成结构化 TXT 纪要文件。

## 🏗️ 技术架构

系统采用四层拓扑架构设计，确保了从底层硬件采集到顶层业务应用的全链路打通：
1.  **业务应用层**：承载实验记录、会议总结等具体业务逻辑。
2.  **流式控制层**：负责多线程调度、双向缓冲管理与状态校验。
3.  **核心模型层**：包含 FunASR (ASR)、Qwen (LLM) 与 Edge-TTS (TTS) 等中枢能力。
4.  **音频外设层**：负责环境收音、VAD 检测与扬声器播放。

## 📊 性能表现

| 维度 | 实测结论 | 达标评价 |
| :--- | :--- | :--- |
| **响应延时** | 首字回复延迟 < 1s，交互丝滑无断续 | 优秀 |
| **ASR 准确性** | 支持多语种与复杂环境下高准确率转写 | 达标 |
| **运行稳定性** | 支持超过 30 分钟的连续稳定在线交互 | 优秀 |
| **数据安全性** | 采用 `utf-8-sig` 编码与物理拦截逻辑，防乱码防幻觉 | 优异 |

## 🚀 快速开始

### 环境依赖
* Python 3.10+
* 相关库：`funasr`, `openai`, `openwakeword`, `edge-tts`, `pygame`, `sounddevice`

### 安装与运行
1.  **声纹录入**：
    ```bash
    python register_voice.py
    ```
2.  **启动系统**：
    ```bash
    python main.py
    ```
3.  **UI 界面**：
    在浏览器中打开 `alexa_ui.html` 即可看到实时状态与对话气泡。
