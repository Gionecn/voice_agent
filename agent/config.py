"""VoiceAgent 的集中配置模块。

配置被有意分成两类：

1. 模型名、采样率、音色、服务地址等非敏感运行参数直接保存在本文件中，
   便于代码审查、环境对比和版本追踪。
2. API Key、Workspace ID 等凭据只从项目根目录的 ``.env`` 读取，避免将
   私密信息写入源码。

其他后端模块应通过 :func:`get_settings` 获取私密配置，并从本模块导入普通
常量，不要在各个 Provider 中重复解析环境变量。
"""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


# 模型名称、采样率等非敏感参数随代码进入版本控制，不放入 .env。
#
# ASR_SAMPLE_RATE 的单位是 Hz。前端 AudioWorklet 必须把麦克风音频重采样成
# 相同的 16 kHz、单声道、有符号 16 位小端 PCM；否则语速和识别准确率会异常。
ASR_MODEL = "fun-asr-realtime"
ASR_SAMPLE_RATE = 16_000

# TTS_SAMPLE_RATE 的单位同样是 Hz。该值要和 Provider 请求的 AudioFormat、
# 前端 PcmPlayer 的播放采样率保持一致。Cherry 是当前产品默认中文音色。
TTS_MODEL = "qwen3-tts-flash-realtime"
TTS_VOICE = "Cherry"
TTS_SAMPLE_RATE = 24_000

# DeepSeek 通过 OpenAI 兼容协议接入 LangChain。最大输出长度是单轮生成上限，
# 并不代表语音回答一定会生成这么长；SYSTEM_PROMPT 仍要求回答尽量简短。
DEEPSEEK_MODEL = "deepseek-v4-flash"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_V4_MAX_TOKENS = 8_192


class Settings(BaseSettings):
    """从项目根目录的 ``.env`` 加载私密配置。

    ``SecretStr`` 会在对象打印和验证错误中隐藏真实值，降低密钥意外进入日志
    的风险。真正调用供应商 SDK 时，调用方才通过 ``get_secret_value()`` 取值。

    Attributes:
        dashscope_api_key: Fun-ASR 和 Qwen TTS 共用的阿里云百炼 API Key。
        dashscope_workspace_id: 可选的百炼 Workspace ID；为空时使用账号默认空间。
        deepseek_api_key: DeepSeek OpenAI 兼容接口的鉴权密钥。
        zhipu_api_key: 智谱网络搜索密钥；仅在 Agent 实际调用搜索工具时使用。
    """

    # env_file_encoding 明确为 UTF-8，避免 Windows 默认编码导致中文或特殊字符
    # 解析错误。extra="ignore" 允许 .env 暂存其他环境变量而不阻止服务启动。
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    dashscope_api_key: SecretStr = Field(alias="DASHSCOPE_API_KEY")
    dashscope_workspace_id: str | None = Field(
        default=None,
        alias="DASHSCOPE_WORKSPACE_ID",
    )
    deepseek_api_key: SecretStr = Field(alias="DEEPSEEK_API_KEY")
    zhipu_api_key: SecretStr | None = Field(default=None, alias="ZHIPU_API_KEY")


@lru_cache
def get_settings() -> Settings:
    """创建并缓存全局配置对象。

    首次调用时 Pydantic 会读取并验证 ``.env``；缺少必填密钥会在这里抛出
    ValidationError，使问题尽早暴露。后续调用返回同一对象，避免每条
    WebSocket 消息都重新读取磁盘。

    Returns:
        已完成环境变量解析和类型验证的 :class:`Settings` 实例。

    Note:
        进程启动后修改 ``.env`` 不会自动刷新缓存，需要重启后端服务。
    """

    return Settings()
