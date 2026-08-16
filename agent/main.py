"""VoiceAgent 的 FastAPI 应用入口。

本模块只负责网络边界：创建应用、配置开发环境 CORS、暴露健康检查，并把每条
``/ws/voice`` 连接交给独立的 ``VoiceSession``。ASR、Agent、TTS 的业务时序
全部留在 Session 层，避免 WebSocket 路由演变成难以测试的巨型函数。

WebSocket 使用两种帧：浏览器发送和接收的 PCM 音频使用二进制帧；状态、转写、
回答文本及中断命令使用 JSON 文本帧。
"""

import json
import logging
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from agent.config import get_settings
from agent.session import VoiceSession


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ERROR_LOG_PATH = PROJECT_ROOT / "logs" / "voice-agent-error.log"


def configure_logging() -> None:
    """配置控制台日志和本地错误日志。

    Uvicorn 可能已经为根 Logger 安装 Handler，因此这里不能只依赖
    ``basicConfig(filename=...)``。函数会显式检查目标 FileHandler，不存在时
    才添加，避免模块重复加载后同一错误被写入多次。

    错误文件固定为 ``logs/voice-agent-error.log``，只接收 ERROR 及以上级别；
    普通启动信息仍由控制台 Handler 处理。
    """

    logging.basicConfig(level=logging.INFO)
    ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    target = str(ERROR_LOG_PATH.resolve()).casefold()
    root_logger = logging.getLogger()
    # 防止开发环境重复导入模块时为同一文件添加多个 Handler。
    for handler in root_logger.handlers:
        if not isinstance(handler, logging.FileHandler):
            continue
        if str(Path(handler.baseFilename).resolve()).casefold() == target:
            return

    # 文件只记录错误，日常 INFO 日志仍输出到启动终端。
    error_handler = logging.FileHandler(ERROR_LOG_PATH, encoding="utf-8")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
    )
    root_logger.addHandler(error_handler)


configure_logging()

app = FastAPI(title="Voice Agent API", version="0.1.0")
# 开发阶段只允许本机 Vite 页面跨域访问后端。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> dict[str, str]:
    """返回最小健康状态，供启动脚本或存活探针检查。

    当前检查只证明 FastAPI 进程能够响应，不主动连接三个外部模型服务，因而
    不代表 ASR、DeepSeek 和 TTS 都可用。
    """

    return {"status": "ok"}


@app.websocket("/ws/voice")
async def voice_socket(websocket: WebSocket) -> None:
    """维护一个浏览器连接对应的一条完整语音会话。

    Args:
        websocket: FastAPI 已完成 HTTP Upgrade 的 WebSocket 对象。

    协议处理规则：
        * 二进制帧：16 kHz、单声道、PCM16 小端音频；
        * ``interrupt``：取消当前 Agent/TTS 回答并清空前端播放；
        * ``session.stop``：结束循环，进入统一资源清理。

    未识别的文本控制消息目前会被忽略。任何非断线异常都会写入错误日志，并在
    连接仍可用时尽量发送 ``source=server`` 的错误事件。
    """

    await websocket.accept()
    session: VoiceSession | None = None
    try:
        session = VoiceSession(websocket, get_settings())
        await session.start()
        while True:
            message = await websocket.receive() # 前端传入PCM二进制的音频数据
            # 二进制帧始终视为麦克风 PCM 音频，直接交给实时 ASR。
            if audio := message.get("bytes"):
                await session.receive_audio(audio)
                continue
            text = message.get("text")
            if not text:
                continue
            # 文本帧仅承载中断、停止等控制协议。
            control = json.loads(text)
            if control.get("type") == "interrupt":
                await session.interrupt()
            elif control.get("type") == "session.stop":
                break
    except WebSocketDisconnect:
        # 浏览器正常关闭连接时无需记录错误。
        pass
    except Exception as exc:
        logging.exception("Voice WebSocket failed")
        try:
            await websocket.send_json(
                {"type": "error", "source": "server", "message": str(exc)}
            )
        except RuntimeError:
            pass
    finally:
        # 无论正常断开还是异常退出，都释放三个模型连接和后台任务。
        if session:
            await session.close()
