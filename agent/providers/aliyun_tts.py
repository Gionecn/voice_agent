"""Qwen 双向实时语音合成适配器。

该 Provider 同时处理两个方向的数据：上行持续接收 DeepSeek 产生的文本增量，
下行持续接收 Qwen 返回的 Base64 音频分片。DashScope SDK 使用同步调用和工作
线程回调，本模块将阻塞方法放入 ``asyncio.to_thread``，并通过线程安全队列把
回调事件交回 ``VoiceSession``。

输出统一为 24 kHz、单声道、16 位 PCM 字节，不包含 WAV 文件头，可直接通过
WebSocket 二进制帧发送给前端流式播放器。
"""

import asyncio
import base64
from collections.abc import AsyncIterator

import dashscope
from dashscope.audio.qwen_tts_realtime import (
    AudioFormat,
    QwenTtsRealtime,
    QwenTtsRealtimeCallback,
)

from agent.events import ProviderError, TtsEvent


# None 是队列结束哨兵，表示供应商会话已经关闭。
TtsQueueItem = TtsEvent | ProviderError | None


class _Callback(QwenTtsRealtimeCallback):
    """接收 TTS SDK 工作线程回调，并转换为项目内部事件。

    Callback 不直接访问 WebSocket，确保供应商线程不会和 FastAPI 事件循环并发
    写同一连接。所有输出先进入队列，再由 Session 中唯一的消费者处理。
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue[TtsQueueItem],
    ) -> None:
        """保存事件循环与 TTS 事件队列。

        Args:
            loop: 当前 VoiceSession 所在的 asyncio 事件循环。
            queue: 连接 SDK 回调生产者和 Session 消费者的内部队列。
        """

        self._loop = loop
        self._queue = queue

    def _put(self, item: TtsQueueItem) -> None:
        """在主事件循环线程中安排无阻塞队列写入。

        不能直接从 SDK 线程调用 ``asyncio.Queue.put_nowait``；
        ``call_soon_threadsafe`` 负责唤醒主事件循环并保持线程安全。
        """

        self._loop.call_soon_threadsafe(self._queue.put_nowait, item)

    def on_event(self, message: dict) -> None:
        """把供应商协议事件转换为标准 TTS 事件。

        Args:
            message: DashScope Qwen TTS WebSocket 事件字典。

        事件映射：
            * ``response.created`` → ``started``；
            * ``response.audio.delta`` → 解码后的 ``audio``；
            * ``response.audio.done``/``response.done`` → ``finished``；
            * ``error`` → ``ProviderError(source="tts")``；
            * ``session.finished`` → 忽略（server_commit 模式下供应商会在
              ``cancel_response`` 后回发此事件作协议确认，但 WebSocket 连接
              仍可复用，真正的会话结束由 ``on_close`` 上报）。
        """

        event_type = message.get("type")
        if event_type == "response.created":
            self._put(TtsEvent(type="started"))
        elif event_type == "response.audio.delta":
            encoded = message.get("delta")
            if encoded:
                # Base64 是供应商 JSON 协议的传输编码，并非音频格式。解码后的
                # 字节已经是 update_session 中约定的 24 kHz PCM16。
                self._put(TtsEvent(type="audio", audio=base64.b64decode(encoded)))
        elif event_type in {"response.audio.done", "response.done"}:
            self._put(TtsEvent(type="finished"))
        elif event_type == "error":
            error = message.get("error") or message
            self._put(ProviderError(source="tts", message=str(error)))
        elif event_type == "session.finished":
            # 不要把 session.finished 当作队列结束哨兵：用户打断（cancel_response）
            # 就会触发该事件，若据此终止消费，_consume_tts 将永久退出，之后所有
            # 轮次的语音都无声。连接复用期间该事件只是协议确认，连接真正关闭时
            # 由 on_close 负责结束事件流。
            pass

    def on_close(self, close_status_code, close_msg) -> None:
        """底层 WebSocket 关闭时发送哨兵，使消费协程自然结束。

        ``close_status_code`` 和 ``close_msg`` 由 SDK 提供；当前错误详情会优先通过
        ``error`` 事件上报，因此关闭回调只负责终止事件流。
        """

        self._put(None)


class AliyunRealtimeTts:
    """Qwen 双向实时 TTS 会话的异步外观。

    ``append_text`` 可以在 LLM 尚未完成回答时连续调用；``commit`` 表示本轮文本
    已结束；``cancel`` 用于用户打断。一个实例在整个 VoiceSession 生命周期内
    复用同一条供应商连接，避免每一轮对话重新握手。
    """

    def __init__(
        self,
        *,
        api_key: str,
        workspace_id: str | None,
        model: str,
        voice: str,
    ) -> None:
        """创建 TTS 客户端，实际网络连接由 ``connect`` 延迟建立。

        Args:
            api_key: 阿里云百炼 API Key。
            workspace_id: 可选 Workspace ID。
            model: Qwen 实时 TTS 模型名。
            voice: 当前会话使用的声音角色。

        Note:
            本类必须在运行中的 asyncio 事件循环里创建，以便 Callback 保存正确
            的主循环引用。
        """

        # DashScope SDK 使用进程级 api_key，初始化时统一设置。
        dashscope.api_key = api_key
        self._queue: asyncio.Queue[TtsQueueItem] = asyncio.Queue()
        self._callback = _Callback(asyncio.get_running_loop(), self._queue)
        self._client = QwenTtsRealtime(
            model=model,
            callback=self._callback,
            workspace=workspace_id,
        )
        self._voice = voice
        self._connected = False

    async def connect(self) -> None:
        """建立连接并配置中文、音色及 24 kHz PCM 输出。

        ``server_commit`` 允许服务端结合流式文本判断合成时机，适合边接收 LLM
        文本边输出声音。方法成功后才设置 ``_connected=True``，避免部分初始化
        失败时其他方法误认为连接可用。
        """

        if self._connected:
            return
        # SDK 方法是同步阻塞调用，放入工作线程避免卡住 FastAPI 事件循环。
        await asyncio.to_thread(self._client.connect)
        await asyncio.to_thread(
            self._client.update_session,
            voice=self._voice,
            response_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
            mode="server_commit",
            language_type="Chinese",
        )
        self._connected = True

    async def append_text(self, text: str) -> None:
        """向双向会话持续追加 LLM 生成的文本增量。

        空文本或未连接状态会被忽略。SDK 调用进入线程池，避免网络写操作阻塞
        Session 的 WebSocket 发送和 ASR 事件消费。
        """

        if text and self._connected:
            await asyncio.to_thread(self._client.append_text, text)

    async def commit(self) -> None:
        """通知服务端当前一轮文本输入已经完成。

        commit 不关闭会话，只结束本轮响应；后续用户问题仍可在同一连接上继续
        append。供应商随后会通过 Callback 发送完成事件。
        """

        if self._connected:
            await asyncio.to_thread(self._client.commit)

    async def cancel(self) -> None:
        """取消当前语音响应。

        用户打断时只需发送 ``cancel_response`` 停止当前音频生成。会话工作在
        ``server_commit`` 模式，服务端自行管理输入文本缓冲，因此不能发送
        ``clear_appended_text``（即 ``input_text_buffer.clear``，仅 client commit
        模式支持），否则会收到 invalid_request_error。
        """

        if not self._connected:
            return
        await asyncio.to_thread(self._client.cancel_response)

    async def events(self) -> AsyncIterator[TtsEvent | ProviderError]:
        """按回调到达顺序产出 TTS 生命周期、音频或错误事件。

        Yields:
            :class:`TtsEvent` 或 :class:`ProviderError`。

        收到 ``None`` 后结束。该队列只应由 ``VoiceSession._consume_tts`` 消费，
        以保持 PCM 分片顺序。
        """

        while True:
            event = await self._queue.get()
            if event is None:
                return
            yield event

    async def close(self) -> None:
        """结束 TTS 会话并关闭底层 WebSocket 连接。

        方法幂等。``finish`` 用于向服务端完成会话协议，``close`` 负责实际释放
        底层连接；即使 finish 抛出异常，finally 仍保证 close 被执行。
        """

        if not self._connected:
            return
        self._connected = False
        try:
            await asyncio.to_thread(self._client.finish)
        finally:
            # 即使 finish 失败也必须执行 close，避免残留网络线程。
            await asyncio.to_thread(self._client.close)
