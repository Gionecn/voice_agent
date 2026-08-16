# Voice Agent Project

本项目是一个基于「三明治架构」（STT → Agent → TTS 流式管道）的实时语音对话助手。用户对着浏览器说话，系统实时完成语音识别、大模型推理、语音合成，并把音频流式回放给用户，实现「开口即聊」的自然对话体验。

- STT：阿里云百炼 `fun-asr-realtime`
- Agent：DeepSeek `deepseek-v4-flash`（通过 `langchain-openai`）
- TTS：阿里云百炼 `qwen3-tts-flash-realtime`
- 后端：FastAPI + WebSocket
- 前端：Vue 3 + Vite + JavaScript

联网搜索使用智谱 Web Search。在 `.env` 中增加以下私密配置即可启用：

```env
ZHIPU_API_KEY=
```

## 一键启动

在项目根目录执行：

```powershell
./.venv/bin/python ./start-all.py
```

脚本会同时启动前端和后端。按 `Ctrl+C` 可以同时停止两个服务。

## 启动后端

在项目根目录填写 `.env` 后执行：

```powershell
./.venv/bin/python -m uvicorn agent.main:app --reload --host 127.0.0.1 --port 8000
```

## 启动前端

```powershell
cd ui
corepack yarn install
corepack yarn dev
```

浏览器打开 `http://127.0.0.1:5173`，允许麦克风权限后即可开始对话。

浏览器向后端发送 16 kHz、单声道、PCM16 小端序音频；后端以二进制 WebSocket 帧返回 24 kHz PCM16 音频。
