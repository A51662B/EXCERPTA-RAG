# -*- coding: utf-8 -*-
"""重排模型客户端（DashScope 原生 rerank 接口）。

和 OpenAI 兼容的 /embeddings 不同，rerank 走的是 DashScope 自己的路子：

    POST https://.../api/v1/services/rerank/text-rerank/text-rerank
    {
      "model": "qwen3.7-text-rerank",
      "input": {"query": "...", "documents": ["...", "..."]},
      "parameters": {"top_n": 5, "return_documents": false}
    }
    → {"output": {"results": [{"index": 0, "relevance_score": 0.92}, ...]}}

为什么只在子块上重排
--------------------
rerank 模型有输入窗口（通常 512 token）。子块约 300 token，塞得下；父块可能
3400 token，塞不下。所以顺序必须是：**先在子块上重排，再按父块聚合**。
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from typing import Protocol


class Reranker(Protocol):
    def score(self, query: str, documents: list[str]) -> list[float]: ...


class DashScopeReranker:
    def __init__(self, url: str, api_key: str, model: str, timeout: int = 60, max_retries: int = 3):
        if not url:
            raise ValueError("缺少 rerank url")
        if not api_key:
            raise ValueError("缺少 rerank api key：请在 .env 里配置 DASHSCOPE_API_KEY 或 RERANK_API_KEY")
        self.url = url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    def _post(self, body: dict) -> dict:
        last: Exception | None = None
        for attempt in range(self.max_retries):
            req = urllib.request.Request(
                self.url,
                data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
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
                detail = e.read().decode("utf-8", "replace")[:300]
                last = RuntimeError(f"rerank 接口 HTTP {e.code}：{detail}")
                if e.code == 429 or 500 <= e.code < 600:
                    time.sleep(2 ** attempt)
                    continue
                raise last from e
            except (urllib.error.URLError, TimeoutError) as e:
                last = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"rerank 重试 {self.max_retries} 次仍失败：{last}")

    def score(self, query: str, documents: list[str]) -> list[float]:
        """返回与 documents 等长的相关性分数（顺序与输入一致）。"""
        if not documents:
            return []
        body = {
            "model": self.model,
            "input": {"query": query, "documents": documents},
            "parameters": {"top_n": len(documents), "return_documents": False},
        }
        t0 = time.time()
        resp = self._post(body)
        elapsed = time.time() - t0
        print(
            f"[rerank] {len(documents)} 条 / {sum(len(d) for d in documents)} 字符 -> {elapsed:.2f}s",
            file=sys.stderr,
            flush=True,
        )
        results = (resp.get("output") or {}).get("results") or []

        scores = [0.0] * len(documents)
        seen = 0
        for r in results:
            idx = r.get("index")
            if not isinstance(idx, int) or not (0 <= idx < len(documents)):
                continue
            value = r.get("relevance_score", r.get("score", 0.0))
            scores[idx] = float(value)
            seen += 1
        if seen == 0:
            raise RuntimeError(f"rerank 返回里没有可解析的分数：{json.dumps(resp, ensure_ascii=False)[:300]}")
        return scores


def build_reranker(url: str, api_key: str, model: str) -> DashScopeReranker:
    return DashScopeReranker(url=url, api_key=api_key, model=model)
