"""Fun-ASR 实时语音识别适配器。

DashScope ``Recognition`` 使用同步方法启动连接，并在 SDK 自己的工作线程中
触发回调；FastAPI/VoiceSession 则运行在 asyncio 事件循环中。直接在回调线程
操作 ``asyncio.Queue`` 并不安全，因此本模块通过
``loop.call_soon_threadsafe(queue.put_nowait, event)`` 跨越线程边界。

上游输入必须是 16 kHz、单声道、PCM16 小端音频。下游输出统一为
``TranscriptEvent`` 或 ``ProviderError``，Session 不需要理解供应商原始对象。
"""

import asyncio
from collections.abc import AsyncIterator

import dashscope
from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult

from agent.events import ProviderError, TranscriptEvent


# None 是队列结束哨兵，用于终止 ASR 事件消费协程。
AsrQueueItem = TranscriptEvent | ProviderError | None


class _Callback(RecognitionCallback):
    """把 ASR SDK 工作线程中的识别回调安全传入主事件循环。

    该类只做协议转换和跨线程投递，不承担业务状态切换。是否用最终转写触发
    Agent、是否过滤重复句子，都由 ``VoiceSession`` 决定。
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue[AsrQueueItem],
    ) -> None:
        """保存主事件循环和由 Session 消费的异步队列。

        Args:
            loop: 创建 Provider 时正在运行的 FastAPI 事件循环。
            queue: ASR 内部事件队列，生产者是 SDK 回调线程，消费者是 Session。
        """

        self._loop = loop
        self._queue = queue

    def _put(self, item: AsrQueueItem) -> None:
        """从任意 SDK 线程安全地安排一次队列写入。

        ``asyncio.Queue.put_nowait`` 最终仍在所属事件循环线程执行，因此不会违反
        asyncio 对线程亲和性的要求。当前队列无容量上限，不会触发 QueueFull。
        """

        self._loop.call_soon_threadsafe(self._queue.put_nowait, item)

    def on_event(self, result: RecognitionResult) -> None:
        """解析一次识别回调并产生标准转写事件。

        Fun-ASR 会多次回调同一句话：用户说话过程中通常是 ``final=False`` 的
        partial，服务端断句后产生 ``final=True``。空文本和非字典结构没有业务
        价值，会被直接忽略。

        Args:
            result: DashScope SDK 提供的 ``RecognitionResult``。
        """

        sentence = result.get_sentence()
        if not isinstance(sentence, dict):
            return
        text = str(sentence.get("text") or "").strip()
        if text:
            self._put(
                TranscriptEvent(
                    text=text,
                    final=RecognitionResult.is_sentence_end(sentence),
                )
            )

    def on_error(self, result: RecognitionResult) -> None:
        """将 SDK 错误统一转换成 ``source=asr`` 的 ProviderError。

        优先使用 SDK 的 ``message`` 属性；若当前版本没有该属性，则退回整个
        对象的字符串表示，保证前端和日志至少能看到诊断信息。
        """

        message = getattr(result, "message", None) or str(result)
        self._put(ProviderError(source="asr", message=message))

    def on_complete(self) -> None:
        """识别会话完成后发送 ``None`` 哨兵终止异步迭代器。"""

        self._put(None)

    def on_close(self) -> None:
        """实现 SDK 要求的关闭回调。

        正常关闭已经由 ``on_complete`` 投递结束哨兵，因此这里不再重复投递，
        避免消费者在未来复用队列时读到多个结束标记。
        """

        pass


class FunAsrStream:
    """DashScope 回调/线程式 Recognition API 的异步桥接器。

    对 Session 暴露四个简单异步操作：``start``、``send_audio``、``events`` 和
    ``close``。同步阻塞的启动/停止动作放入线程池；高频音频发送保留 SDK 的
    直接调用，避免每个约 100 ms 音频块都产生一次线程池调度开销。
    """

    def __init__(
        self,
        *,
        api_key: str,
        workspace_id: str | None,
        model: str,
        sample_rate: int,
    ) -> None:
        """配置实时中文识别客户端，暂不建立网络连接。

        Args:
            api_key: 阿里云百炼 API Key。
            workspace_id: 可选 Workspace ID。
            model: 实时 ASR 模型名，当前为 ``fun-asr-realtime``。
            sample_rate: 输入 PCM 的采样率，必须与前端实际发送值一致。

        Note:
            ``asyncio.get_running_loop`` 要求本类必须在异步上下文中创建；项目中
            它由 WebSocket 路由创建的 ``VoiceSession`` 初始化。
        """

        dashscope.api_key = api_key
        self._queue: asyncio.Queue[AsrQueueItem] = asyncio.Queue()
        self._callback = _Callback(asyncio.get_running_loop(), self._queue)
        # 参数说明：
        # - format="pcm"：浏览器发送无 WAV 文件头的原始 PCM；
        # - semantic_punctuation_enabled=False：不启用语义标点增强；
        # - max_sentence_silence=800：约 800 ms 静音后服务端确认一句结束；
        # - punctuation_prediction_enabled=True：最终转写补充中文标点；
        # - heartbeat=True：空闲阶段维持长连接，降低重新建连成本。
        self._recognition = Recognition(
            model=model,
            callback=self._callback,
            format="pcm",
            sample_rate=sample_rate,
            workspace=workspace_id,
            semantic_punctuation_enabled=False,
            max_sentence_silence=800,
            punctuation_prediction_enabled=True,
            language_hints=["zh"],
            heartbeat=True,
        )
        self._started = False

    async def start(self) -> None:
        """在线程池中启动阻塞式 ASR 会话。

        该方法是幂等的；重复调用不会建立第二条识别连接。连接失败或鉴权失败
        会直接抛出异常，由 Session 启动流程继续向 WebSocket 路由传播。
        """

        if self._started:
            return
        await asyncio.to_thread(self._recognition.start)
        self._started = True

    async def send_audio(self, chunk: bytes) -> None:
        """发送一块 16 kHz 单声道 PCM16 麦克风音频。

        Args:
            chunk: 浏览器 AudioWorklet 产生的原始二进制音频块。

        Raises:
            RuntimeError: 在 ``start`` 成功前收到音频，说明会话生命周期错误。
        """

        if not self._started:
            raise RuntimeError("Fun-ASR stream has not started")
        self._recognition.send_audio_frame(chunk)

    async def events(self) -> AsyncIterator[TranscriptEvent | ProviderError]:
        """持续产出临时转写、最终转写和供应商错误。

        Yields:
            :class:`TranscriptEvent` 或 :class:`ProviderError`。

        收到 ``None`` 哨兵时正常返回。该迭代器应只有一个消费者，否则多个协程
        会竞争同一队列，破坏识别事件顺序。
        """

        while True:
            event = await self._queue.get()
            if event is None:
                return
            yield event

    async def close(self) -> None:
        """在线程池中停止识别会话。

        先将 ``_started`` 置为 ``False``，阻止关闭过程中继续接受音频；重复调用
        直接返回，保证 WebSocket 异常路径和正常 finally 可以安全共享清理逻辑。
        """

        if not self._started:
            return
        self._started = False
        await asyncio.to_thread(self._recognition.stop)
