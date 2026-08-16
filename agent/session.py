"""单个 VoiceAgent WebSocket 会话的核心编排器。

``VoiceSession`` 把三明治架构的三个阶段组织成可并发工作的流式流水线：

* 浏览器二进制音频持续送入 Fun-ASR；
* ASR final 文本触发 LangChain/DeepSeek Agent；
* Agent 文本增量同时发往浏览器和 Qwen TTS；
* TTS PCM 音频分片重新通过同一 WebSocket 返回浏览器。

Session 还负责跨阶段的控制语义：新一句话到达时取消旧轮次、用户主动打断时清空
TTS 与前端播放队列、连接断开时按依赖顺序释放模型和后台任务。Provider 只关心
单一模型协议，Session 才是整个实时对话的状态机。
"""

import asyncio
import contextlib
import logging
import time
from typing import Any

from fastapi import WebSocket

from agent.config import (
    ASR_MODEL,
    ASR_SAMPLE_RATE,
    TTS_MODEL,
    TTS_VOICE,
    Settings,
)
from agent.events import ProviderError, TranscriptEvent, TtsEvent, wire_event
from agent.llm_agent import DeepSeekVoiceAgent
from agent.providers.aliyun_tts import AliyunRealtimeTts
from agent.providers.fun_asr import FunAsrStream


logger = logging.getLogger("voice_agent.session")

# DashScope 实时 ASR 大约 23 秒收不到任何音频数据就会关闭识别连接（服务端
# 行为，与客户端 SDK 内部定时器无关）；浏览器首次授权麦克风可能耗时更久。
# 因此会话空闲时由 _keepalive_loop 定期补发一个 50 ms 静音 PCM 帧，阻止服务端
# 超时，同时持续重置 SDK 自身的静音定时器。真实音频持续到达时不会触发补发。
_ASR_KEEPALIVE_INTERVAL_S = 5.0
_ASR_KEEPALIVE_FRAME = b"\x00" * (ASR_SAMPLE_RATE * 2 // 20)


class VoiceSession:
    """管理一条用户连接及其完整的实时语音处理生命周期。

    常驻并发任务共三个：WebSocket 发送循环、ASR 事件消费、TTS 事件消费。
    ``_turn_task`` 则是随每个 final 转写创建和取消的短生命周期 Agent 任务。
    这种拆分确保上一轮回答可以被取消，而实时识别和模型长连接继续复用。
    """

    def __init__(self, websocket: WebSocket, settings: Settings) -> None:
        """为当前连接创建三个模型客户端和内部异步状态。

        Args:
            websocket: 已由 FastAPI 接受的浏览器 WebSocket。
            settings: 已验证的私密配置对象。

        初始化阶段只构造对象，不执行完整启动流程；ASR/TTS 网络连接在
        :meth:`start` 中并发建立。DeepSeek 客户端也按 LangChain 的延迟请求
        模式工作，直到第一轮文本进入才访问模型接口。
        """

        self._websocket = websocket
        self._settings = settings
        # ASR 与 TTS 属于同一 DashScope 账号，因此复用一个解密后的 Key；Key
        # 只保存在 Provider 客户端内部，不会写入 WebSocket 事件。
        dashscope_key = settings.dashscope_api_key.get_secret_value()
        self._asr = FunAsrStream(
            api_key=dashscope_key,
            workspace_id=settings.dashscope_workspace_id,
            model=ASR_MODEL,
            sample_rate=ASR_SAMPLE_RATE,
        )
        self._tts = AliyunRealtimeTts(
            api_key=dashscope_key,
            workspace_id=settings.dashscope_workspace_id,
            model=TTS_MODEL,
            voice=TTS_VOICE,
        )
        self._agent = DeepSeekVoiceAgent(
            api_key=settings.deepseek_api_key.get_secret_value(),
        )
        # 所有 JSON 事件和二进制音频先进入同一队列，再由唯一发送协程写入
        # WebSocket。这避免 _consume_asr、_run_turn、_consume_tts 三个协程
        # 同时写连接，并保留诸如 tts.started → audio → tts.finished 的顺序。
        self._outgoing: asyncio.Queue[dict[str, Any] | bytes | None] = (
            asyncio.Queue()
        )
        # _tasks[0] 固定为发送循环，后续元素是 ASR/TTS 消费者。close 会依赖这个
        # 位置约定，让发送循环最后停止并排空此前已经入队的事件。
        self._tasks: list[asyncio.Task] = []
        self._turn_task: asyncio.Task | None = None
        # 标记是否已向 TTS 追加当前轮次文本。只有 True 时才需要发送 cancel，
        # 避免无回答状态下调用供应商取消接口。
        self._tts_pending = False
        # 部分 ASR SDK 会重复回调同一条 final；保存最近文本防止重复触发 Agent。
        self._last_final = ""
        # close 可能同时从 WebSocket finally 和异常清理路径调用，使用此标记幂等。
        self._closed = False
        # 记录最近一次真实音频到达时间；浏览器麦克风尚未出流时保活任务据此
        # 判断是否补发静音帧，receive_audio 每次收到音频都会刷新该时间戳。
        self._last_audio_at = 0.0

    async def start(self) -> None:
        """启动发送循环、模型连接以及 ASR/TTS 消费任务。

        启动顺序经过刻意安排：

        1. 先创建发送循环，使后续任何状态事件都有消费者；
        2. 告诉前端本会话期望的输入和输出音频格式；
        3. 并发连接 ASR 与 TTS，缩短用户等待时间；
        4. 连接成功后创建两个 Provider 消费任务；
        5. 最后切换为 ``listening``。

        Raises:
            Exception: 任一模型连接失败时向上传播，由 WebSocket 路由记录并关闭
                已成功创建的其他资源。
        """

        # 必须先启动发送循环，后续产生的 ready 等事件才有消费者。
        self._tasks.append(asyncio.create_task(self._send_loop()))
        # session.ready 中的格式也是前后端协议契约。修改采样率时必须同步修改
        # config.py、Provider AudioFormat 和前端音频处理代码。
        await self._send(
            wire_event(
                "session.ready",
                input_audio={"encoding": "pcm_s16le", "sample_rate": 16000},
                output_audio={"encoding": "pcm_s16le", "sample_rate": 24000},
            )
        )
        # ASR 和 TTS 相互独立，并发建连可缩短会话初始化时间。
        await asyncio.gather(self._asr.start(), self._tts.connect())
        self._tasks.extend(
            [
                asyncio.create_task(self._keepalive_loop()),
                asyncio.create_task(self._consume_asr()),
                asyncio.create_task(self._consume_tts()),
            ]
        )
        await self._send(wire_event("state.changed", state="listening"))

    async def receive_audio(self, chunk: bytes) -> None:
        """将浏览器上传的 PCM 音频转发到实时 ASR。（STT）

        Args:
            chunk: 16 kHz、单声道、PCM16 小端二进制分片，通常约 100 ms。

        空字节不会发送。该方法不缓存和重编码音频，减少实时链路中的复制与
        延迟；格式正确性由浏览器 AudioWorklet 和协议共同保证。
        """

        if chunk:
            self._last_audio_at = time.monotonic()
            await self._asr.send_audio(chunk)

    async def _keepalive_loop(self) -> None:
        """在浏览器音频长时间中断时向 ASR 补发静音帧，维持供应商长连接。

        DashScope 实时 ASR 约 23 秒收不到音频就会关闭连接；浏览器首次授权
        麦克风或采集链路恢复可能超过该时长。因此距上次真实音频超过 5 秒就
        补发一个 50 ms 静音 PCM 帧。真实音频持续到达时不会触发补发。
        """

        while True:
            await asyncio.sleep(_ASR_KEEPALIVE_INTERVAL_S)
            if (
                self._closed
                or time.monotonic() - self._last_audio_at
                < _ASR_KEEPALIVE_INTERVAL_S
            ):
                continue
            # 补发失败说明 ASR 已停止（例如连接已断开），静默忽略即可。
            with contextlib.suppress(Exception):
                await self._asr.send_audio(_ASR_KEEPALIVE_FRAME)

    async def interrupt(self) -> None:
        """取消当前回答、停止 TTS，并清空浏览器播放队列。

        中断可能来自用户控制按钮，也可能来自新 final 转写开始新一轮回答。
        操作顺序非常重要：先取消并等待 Agent 任务，阻止继续追加文本；再取消
        TTS 服务端输出；最后通知浏览器丢弃已经收到但尚未播放的 PCM。

        方法在当前没有回答时同样安全，仍会发送 ``playback.clear`` 和
        ``state=listening``，使前后端状态重新对齐。
        """

        task = self._turn_task
        self._turn_task = None
        if task and not task.done():
            # 等待取消真正完成，避免旧轮次继续向 TTS 追加迟到文本。
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if self._tts_pending:
            # TTS cancel 只能阻止后续服务端音频；已经发到浏览器的分片必须依靠
            # playback.clear 清理，两端操作缺一不可。
            await self._tts.cancel()
            self._tts_pending = False
        await self._send(wire_event("playback.clear"))
        await self._send(wire_event("state.changed", state="listening"))

    async def _consume_asr(self) -> None:
        """消费 ASR 事件，并在最终转写到达时创建新回答任务。

        partial 只用于前端实时字幕，不触发模型推理。final 会先展示为正式用户
        消息，再中断旧轮次并创建新的 ``_run_turn`` Task。这里使用 create_task
        而非直接 await，使本协程能够继续消费 ASR，用户再次开口时才有机会
        及时打断正在回答的 Agent。
        """

        async for event in self._asr.events():
            if isinstance(event, ProviderError):
                await self._provider_error(event)
                continue
            if event.final:
                # 某些服务回调可能重复发送 final，避免同一句触发两轮回答。
                if event.text == self._last_final:
                    continue
                self._last_final = event.text
                await self._send(wire_event("stt.final", text=event.text))
                # 新问题到达前先清理仍在进行的旧回答，实现轮次级打断。
                await self.interrupt()
                self._turn_task = asyncio.create_task(self._run_turn(event.text))
            else:
                await self._send(wire_event("stt.partial", text=event.text))

    async def _run_turn(self, transcript: str) -> None:
        """运行一轮文本 Agent，并把增量同步送往前端与 TTS。

        Args:
            transcript: Fun-ASR 已确认的最终用户文本。

        ``response_text`` 用于累积完整回答并在 ``agent.finished`` 事件中返回；
        与此同时，每个增量会立刻发送 ``agent.delta`` 并调用 ``append_text``，
        因而文本展示、LLM 后续生成和语音合成能够重叠执行。

        正常结束时调用 TTS commit。取消异常需要向上重新抛出，以便 interrupt
        确认任务已经结束；其他异常则记录完整堆栈，并向前端报告简短错误。
        """

        response_text = ""
        try:
            await self._send(wire_event("agent.started"))
            await self._send(wire_event("state.changed", state="thinking"))
            async for text in self._agent.stream(transcript): # 采用流水输出的方式来执行Agent
                response_text += text
                await self._send(wire_event("agent.delta", text=text))
                # LLM 不必生成完整回答，TTS 可立即消费当前文本分片。
                self._tts_pending = True
                await self._tts.append_text(text)
            # commit 表示这一轮不再有新文本，但 TTS 音频可能仍在生成；真正的
            # speaking → listening 切换由 _consume_tts 收到 finished 后完成。
            await self._tts.commit()
            await self._send(wire_event("agent.finished", text=response_text))
        except asyncio.CancelledError:
            # 取消属于正常控制流程，必须继续向上传播而不是当作错误吞掉。
            raise
        except Exception as exc:
            logger.exception("Agent turn failed")
            await self._send(wire_event("error", source="agent", message=str(exc)))
            await self._send(wire_event("state.changed", state="listening"))

    async def _consume_tts(self) -> None:
        """消费 TTS 状态与音频，并转换为前端协议。

        状态映射如下：

        * ``started``：发送 ``tts.started``，会话进入 speaking；
        * ``audio``：直接发送 24 kHz PCM16 二进制帧；
        * ``finished``：清除 pending 标记，会话重新进入 listening；
        * ``ProviderError``：统一交给 ``_provider_error``。

        音频不会经过 JSON/Base64 二次包装，因此前端必须根据 WebSocket 帧类型
        区分事件和音频。
        """

        async for event in self._tts.events():
            if isinstance(event, ProviderError):
                await self._provider_error(event)
            elif isinstance(event, TtsEvent):
                if event.type == "started":
                    await self._send(wire_event("tts.started"))
                    await self._send(wire_event("state.changed", state="speaking"))
                elif event.type == "audio" and event.audio:
                    # PCM 音频保持二进制发送，避免 Base64 增加带宽和解码开销。
                    await self._send(event.audio)
                elif event.type == "finished":
                    self._tts_pending = False
                    await self._send(wire_event("tts.finished"))
                    await self._send(wire_event("state.changed", state="listening"))

    async def _provider_error(self, error: ProviderError) -> None:
        """统一向浏览器报告 ASR/TTS 供应商错误。

        Args:
            error: 已携带 ``asr`` 或 ``tts`` 来源的标准错误。

        Provider 故障会把会话状态设为 ``error``，与 Agent 单轮错误恢复为
        ``listening`` 的策略不同：ASR/TTS 长连接故障通常意味着当前语音会话
        已无法可靠继续。
        """

        await self._send(
            wire_event("error", source=error.source, message=error.message)
        )
        await self._send(wire_event("state.changed", state="error"))

    async def _send(self, item: dict[str, Any] | bytes) -> None:
        """将待发送内容入队，由唯一发送协程串行写入 WebSocket。

        Args:
            item: JSON 可序列化事件字典，或无需编码的 PCM 音频字节。

        本方法本身不进行网络 I/O，因此模型消费协程不会被浏览器短暂的发送延迟
        直接阻塞；背压目前依赖无界队列，生产环境应增加容量和慢客户端策略。
        """

        await self._outgoing.put(item)

    async def _send_loop(self) -> None:
        """按入队顺序发送 JSON 事件和 PCM 音频。

        ``bytes`` 使用 ``send_bytes``，普通字典使用 ``send_json``。``None`` 是
        仅供内部关闭使用的哨兵，不会发送到浏览器。这个协程是唯一 WebSocket
        写入者，从结构上避免并发 send 导致协议错误或消息交错。
        """

        while True:
            item = await self._outgoing.get()
            if item is None:
                return
            if isinstance(item, bytes):
                await self._websocket.send_bytes(item)
            else:
                await self._websocket.send_json(item)

    async def close(self) -> None:
        """幂等关闭模型连接、后台任务和发送循环。

        清理顺序：

        1. 标记会话关闭并取消当前 Agent 轮次；
        2. 取消 ASR/TTS 消费任务，防止继续生产前端事件；
        3. 并发关闭 ASR 与 TTS 外部连接，且单方失败不阻止另一方释放；
        4. 等待已取消的消费任务退出；
        5. 向发送队列写入 ``None``，让发送循环处理完已有消息后退出；
        6. 最后等待发送任务完成。

        ``return_exceptions=True`` 和 ``contextlib.suppress`` 用于清理阶段的最佳
        努力释放；不能让一个资源的关闭异常阻止其他资源被回收。
        """

        if self._closed:
            return
        self._closed = True
        if self._turn_task and not self._turn_task.done():
            self._turn_task.cancel()
        # 第一个常驻任务是发送循环，需要等模型消费任务结束后再停。
        for task in self._tasks[1:]:
            task.cancel()
        await asyncio.gather(
            self._asr.close(),
            self._tts.close(),
            return_exceptions=True,
        )
        for task in self._tasks[1:]:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        # 最后放入发送哨兵，确保此前排队的事件能够按序处理。
        await self._outgoing.put(None)
        if self._tasks:
            with contextlib.suppress(asyncio.CancelledError, RuntimeError):
                await self._tasks[0]
