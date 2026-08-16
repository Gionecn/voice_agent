"""VoiceAgent 内部事件与 WebSocket 线协议辅助结构。

供应商 SDK 的原始事件格式各不相同。Provider 层先把它们转换成本模块中的
小型数据类，``VoiceSession`` 因而只依赖统一事件，而无需理解 DashScope 的
原始回调字典。发送给浏览器的 JSON 事件则统一由 :func:`wire_event` 创建，
确保每条消息都包含 ``type`` 字段。
"""

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(slots=True)
class TranscriptEvent:
    """ASR 产生的一条临时或最终转写事件。

    Attributes:
        text: 当前识别出的非空文本。
        final: ``False`` 表示仍可能变化的 partial；``True`` 表示服务端已经
            判定用户当前句子结束，可以触发新一轮 Agent 推理。
    """

    text: str
    final: bool


@dataclass(slots=True)
class ProviderError:
    """ASR 或 TTS 供应商返回的标准化错误。

    Attributes:
        source: 错误来源，用于前端区分识别与合成故障。
        message: 已转换为字符串的供应商错误说明。
    """

    source: Literal["asr", "tts"]
    message: str


@dataclass(slots=True)
class TtsEvent:
    """TTS 生命周期或音频分片事件。

    ``started`` 表示服务端开始生成本轮语音，``audio`` 携带一块 PCM 字节，
    ``finished`` 表示本轮语音已经全部生成。只有 ``audio`` 类型需要设置
    :attr:`audio`，其他类型保持 ``None``。
    """

    type: Literal["started", "audio", "finished"]
    audio: bytes | None = None


def wire_event(event_type: str, **payload: Any) -> dict[str, Any]:
    """生成前后端 WebSocket 约定的 JSON 事件。

    Args:
        event_type: 协议事件名，例如 ``stt.final`` 或 ``state.changed``。
        **payload: 当前事件需要携带的其他 JSON 可序列化字段。

    Returns:
        以 ``type`` 为首要字段的普通字典，可直接交给 ``send_json``。
    """

    return {"type": event_type, **payload}
