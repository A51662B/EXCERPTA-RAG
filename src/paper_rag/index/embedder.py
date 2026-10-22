# -*- coding: utf-8 -*-
"""阶段 6：向量化。

接口约定（实测 qwen3.7-text-embedding-flash）
--------------------------------------------
    POST {base_url}/embeddings
    body: {"model": "...", "input": ["文本1", "文本2", ...]}
    返回: {"data": [{"index": 0, "embedding": [...]}, ...], "usage": {"total_tokens": N}}
    维度: 1024
    批量: 一次最多 25 条（25 条 OK，50 条报 batch size invalid）

规则
----
1. 父块不编码（它不参与检索，编码纯属浪费）。
2. 按批发送，每批 ≤ 20 条；返回按 index 排序还原顺序（不依赖服务端顺序）。
3. 只对 content 编码；空的 content 用空格占位（接口不接受空串）。
4. 429 / 5xx 重试 3 次（指数退避），其它错误直接抛。
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol

from paper_rag.models import Chunk


class EmbeddingClient(Protocol):
    """最小接口：一批文本 -> 一批向量。"""

    dims: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAICompatEmbedder:
    """OpenAI 兼容的 /embeddings 客户端（零依赖实现）。"""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        batch: int = 20,
        timeout: int = 60,
        max_retries: int = 3,
    ):
        if not base_url:
            raise ValueError("缺少 base_url：请在 .env 里配置 EMBEDDING（如 https://.../compatible-mode/v1）")
        if not api_key:
            raise ValueError("缺少 api_key：请在 .env 里配置 EMBEDDING_API_KEY")
        self.url = base_url.rstrip("/") + "/embeddings"
        self.api_key = api_key
        self.model = model
        self.batch = max(1, min(batch, 25))
        self.timeout = timeout
        self.max_retries = max_retries
        self.dims = 0
        self.total_tokens = 0

    # -- 单批 --------------------------------------------------------------

    def _post(self, texts: list[str]) -> dict:
        body = {"model": self.model, "input": texts}
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            req = urllib.request.Request(
                self.url,
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
                detail = e.read().decode("utf-8", "replace")[:300]
                last_error = RuntimeError(f"嵌入接口 HTTP {e.code}：{detail}")
                if e.code == 429 or 500 <= e.code < 600:
                    time.sleep(2 ** attempt)
                    continue
                raise last_error from e
            except (urllib.error.URLError, TimeoutError) as e:
                last_error = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"嵌入接口重试 {self.max_retries} 次仍失败：{last_error}")

    def _embed_one_batch(self, texts: list[str]) -> list[list[float]]:
        resp = self._post(texts)
        data = resp.get("data") or []
        if len(data) != len(texts):
            raise RuntimeError(f"返回条数不符：请求 {len(texts)} 条，返回 {len(data)} 条")
        # 按 index 还原顺序
        ordered = sorted(data, key=lambda d: d.get("index", 0))
        vectors = [d["embedding"] for d in ordered]
        if vectors and not self.dims:
            self.dims = len(vectors[0])
        usage = resp.get("usage") or {}
        self.total_tokens += int(usage.get("total_tokens") or usage.get("prompt_tokens") or 0)
        return vectors

    # -- 对外 --------------------------------------------------------------

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            out.extend(self._embed_one_batch(texts[i: i + self.batch]))
        return out


@dataclass
class EmbedStats:
    embedded: int = 0
    skipped_parent: int = 0
    skipped_existing: int = 0
    dims: int = 0
    tokens: int = 0
    seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"已编码 {self.embedded} 条，跳过父块 {self.skipped_parent} 条"
            f"，跳过已有向量 {self.skipped_existing} 条；"
            f"维度 {self.dims}，token {self.tokens}，耗时 {self.seconds:.1f}s"
        )


def embed_chunks(
    chunks: list[Chunk],
    client: EmbeddingClient,
    force: bool = False,
    limit: int = 0,
) -> EmbedStats:
    """给需要检索的 chunk 填上 vector；父块跳过。"""
    stats = EmbedStats()
    started = time.time()

    targets = [c for c in chunks if c.kind != "parent"]
    stats.skipped_parent = len(chunks) - len(targets)

    todo: list[Chunk] = []
    for c in targets:
        if c.vector and not force:
            stats.skipped_existing += 1
            continue
        todo.append(c)
    if limit:
        todo = todo[:limit]

    if todo:
        texts = [c.content if c.content.strip() else " " for c in todo]
        vectors = client.embed(texts)
        if len(vectors) != len(todo):
            raise RuntimeError(f"向量条数不符：需要 {len(todo)}，实际 {len(vectors)}")

        dims = len(vectors[0]) if vectors else 0
        for c, v in zip(todo, vectors):
            if len(v) != dims:
                raise RuntimeError(f"{c.chunk_id} 的向量维度 {len(v)} 与第一批 {dims} 不一致")
            c.vector = v
        stats.embedded = len(todo)
        stats.dims = dims

    stats.tokens = getattr(client, "total_tokens", 0)
    stats.seconds = time.time() - started
    return stats
