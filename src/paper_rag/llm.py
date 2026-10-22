# -*- coding: utf-8 -*-
"""LLM 调用层。

只做一件事：把 messages 发出去，把原始响应 dict 拿回来。业务层的解析、校验、
降级都不放这里。

两个实现，接口一致：
    OpenAISDKChatClient  装了 openai 包时用，和你现有代码写法一致
    UrllibChatClient     没装包时的零依赖兜底（纯标准库）

选择在 build_client() 里做。

关于 thinking 的一个实测结论（deepseek-flash）：
    thinking 开着时，tool_choice 不允许指定具体函数，会返回
    400 "Thinking mode does not support this tool_choice"。
    所以用 tool 的场景必须 thinking=False。
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Protocol


class ChatClient(Protocol):
    """最小接口：发 messages，收原始响应。"""

    def complete(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        tool_choice: dict | None = None,
        thinking: bool = False,
        max_tokens: int = 1024,
    ) -> dict[str, Any]: ...


class UrllibChatClient:
    """零依赖实现：直接 POST {base_url}/chat/completions。"""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 120):
        if not base_url:
            raise ValueError("缺少 base_url：请在 .env 里配置 OPENAI_BASE_URL")
        if not api_key:
            raise ValueError("缺少 api_key：请在 .env 里配置 OPENAI_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def complete(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        tool_choice: dict | None = None,
        thinking: bool = False,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "top_p": 0.95,
            "max_completion_tokens": max_tokens,
            "stream": False,
            "thinking": {"type": "enabled" if thinking else "disabled"},
        }
        if tools:
            body["tools"] = tools
            if tool_choice:
                body["tool_choice"] = tool_choice

        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            raise RuntimeError(f"LLM 返回 HTTP {e.code}：{detail[:500]}") from e


class OpenAISDKChatClient:
    """openai 包实现，和你现有代码一致（含 extra_body 里的 thinking）。"""

    def __init__(self, base_url: str, api_key: str, model: str):
        from openai import OpenAI  # 延迟导入，没装包时不会炸

        self.model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        tool_choice: dict | None = None,
        thinking: bool = False,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "top_p": 0.95,
            "max_completion_tokens": max_tokens,
            "stream": False,
            "extra_body": {"thinking": {"type": "enabled" if thinking else "disabled"}},
        }
        if tools:
            kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice

        resp = self._client.chat.completions.create(**kwargs)
        return resp.model_dump()


def build_client(base_url: str, api_key: str, model: str) -> ChatClient:
    """优先用 openai 包；没装就用 urllib 兜底。"""
    try:
        import openai  # noqa: F401
    except ImportError:
        return UrllibChatClient(base_url=base_url, api_key=api_key, model=model)
    return OpenAISDKChatClient(base_url=base_url, api_key=api_key, model=model)
