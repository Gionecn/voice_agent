"""供 LangChain Agent 调用的智谱网络搜索工具。

当用户问题涉及天气、新闻、价格、赛事结果等时效信息时，系统提示词要求
DeepSeek 先调用本工具。返回模型前强制限制结果数量。

搜索网页属于不可信外部内容。该工具只把标题、链接和摘要作为事实材料返回，
不会执行搜索结果中的任何指令。异常信息在返回前会移除 API Key，并由主应用
的 ERROR FileHandler 写入 ``logs/voice-agent-error.log``。
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from langchain_core.tools import tool

from agent.config import get_settings


logger = logging.getLogger("voice_agent.web_search")


@lru_cache(maxsize=1)
def _get_zhipu_client() -> Any:
    """首次调用搜索工具时创建客户端，后续在进程内复用。

    Returns:
        已使用 ``ZHIPU_API_KEY`` 初始化的 ``ZhipuAiClient``。

    Raises:
        RuntimeError: 未配置非空 API Key，或当前环境没有安装 ``zai-sdk``。

    ``lru_cache(maxsize=1)`` 避免每次搜索重复创建 HTTP 客户端。由于配置对象也
    有缓存，运行中修改 ``.env`` 后必须重启后端才能使用新 Key。
    """

    api_key = get_settings().zhipu_api_key
    if api_key is None or not api_key.get_secret_value().strip():
        raise RuntimeError("未配置 ZHIPU_API_KEY，无法使用联网搜索。")

    try:
        # 延迟导入使未使用联网搜索的场景不必提前加载可选 SDK。
        from zai import ZhipuAiClient
    except ImportError as exc:
        raise RuntimeError(
            "缺少智谱搜索 SDK，请先安装依赖：pip install zai-sdk"
        ) from exc
    return ZhipuAiClient(api_key=api_key.get_secret_value())


def _value(item: Any, name: str) -> str:
    """安全读取并清洗搜索结果中的单个字段。

    Args:
        item: 智谱 SDK 返回的结果对象，或测试中使用的普通字典。
        name: 需要读取的属性/键名，如 ``title``、``link``、``content``。

    Returns:
        去除首尾空白的字符串；字段缺失或值为空时返回空字符串。

    兼容对象和字典可降低 SDK 小版本调整带来的耦合，也便于单元测试使用简单
    ``SimpleNamespace`` 或 dict 构造响应。
    """

    if isinstance(item, dict):
        value = item.get(name)
    else:
        value = getattr(item, name, None)
    return str(value or "").strip()


def _safe_error(exc: Exception) -> str:
    """生成可记录、可返回的脱敏错误文本。

    Args:
        exc: 搜索客户端初始化或请求阶段捕获的异常。

    Returns:
        将当前智谱 API Key 替换为 ``***`` 后的错误字符串。

    该保护针对错误信息中直接出现密钥的情况；生产系统还应在集中日志平台配置
    二次脱敏规则，避免其他请求头、账号或用户隐私进入日志。
    """

    message = str(exc)
    api_key = get_settings().zhipu_api_key
    if api_key is not None:
        secret = api_key.get_secret_value()
        if secret:
            message = message.replace(secret, "***")
    return message


@tool("web_search", parse_docstring=True)
def web_search(query: str) -> str:
    """使用智谱 Web Search API 搜索公开网络资料。

    当问题涉及新闻、天气、价格、赛事结果、软件版本、产品信息或其他可能变化的
    事实时使用。不要搜索 API Key、Token、私有仓库内容或其他敏感信息。

    处理步骤：

    1. 把换行和连续空白压缩成单个空格；
    2. 使用 ``search_pro`` 发起不限时间范围的查询；
    3. 即使服务端返回超过 count，也只遍历前三条；
    4. 跳过完全为空的结果，并输出标题、链接和摘要；
    5. 发生异常时记录 ERROR，同时把可读错误作为工具结果交回 Agent。

    Args:
        query: DeepSeek 生成的简洁搜索词或自然语言问题。

    Returns:
        最多三条结果组成的纯文本。失败不会向 Agent 抛异常，而是返回以
        ``搜索失败`` 开头的文本，让模型能够明确告知用户而不是编造答案。
    """

    # split 后不传参数会按所有 Unicode 空白拆分，因此可以同时处理空格、Tab
    # 和换行；再用单空格拼接，减少无意义 Token 并让日志保持单行。
    normalized_query = " ".join((query or "").split())
    if not normalized_query:
        return "搜索失败：query 不能为空"

    try:
        # count 是服务端期望值；下面仍会在本地强制截断，防止接口返回超量。
        response = _get_zhipu_client().web_search.web_search(
            search_engine="search_pro",
            search_query=normalized_query,
            count=3,
            search_recency_filter="noLimit",
        )
        # SDK 响应对象缺少 search_result 或返回 None 时统一按空列表处理。
        results = getattr(response, "search_result", None) or []
        formatted: list[str] = []
        # 某些响应会超过请求数量，本地强制只给 Agent 传递前三条结果。
        for index, item in enumerate(results[:3], start=1):
            title = _value(item, "title")
            link = _value(item, "link") or _value(item, "url")
            content = _value(item, "content")
            # 三个核心字段全部为空的结果没有提供任何事实信息，直接跳过。
            if not any((title, link, content)):
                continue
            parts = [f"结果 {index}"]
            if title:
                parts.append(f"标题：{title}")
            if link:
                parts.append(f"链接：{link}")
            if content:
                parts.append(f"摘要：{content}")
            formatted.append("\n".join(parts))
        # 结果之间使用空行分隔，帮助 LLM 识别不同来源边界。
        return "\n\n".join(formatted) if formatted else "没有搜索到任何内容。"
    except Exception as exc:
        error = _safe_error(exc)
        # ERROR 级别会由主应用写入 logs/voice-agent-error.log。
        logger.error("联网搜索失败：query=%s error=%s", normalized_query, error)
        return f"搜索失败：{error}"
