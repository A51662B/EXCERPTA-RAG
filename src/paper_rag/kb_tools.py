# -*- coding: utf-8 -*-
"""给 agent 用的工具函数：把流水线包装成「结构化输入 → 结构化输出」。

设计原则
--------
1. **不打印**：MCP 服务端的 stdout 是协议通道，任何 print 都会破坏消息流。
2. **参数收敛**：只暴露模型能判断的参数（query / top_n / doc_id / md_path）。
   工程旋钮（--force、--no-rerank、candidates）走 .env 默认值。
3. **错误变数据**：找不到文档、参数不对，返回 {"ok": False, "error": ...}，
   不抛异常给协议层。
4. **返回裁剪**：单段文本最多 MAX_TEXT_CHARS 字符，避免一次塞爆模型上下文。
"""
from __future__ import annotations

import sys
import time

from dataclasses import dataclass
from typing import Any

from paper_rag.config import Settings, load_settings
from paper_rag.index.embedder import OpenAICompatEmbedder
from paper_rag.index.es_store import ESStore
from paper_rag.index.reranker import build_reranker
from paper_rag.retrieve.pipeline import retrieve
from paper_rag.tokenizer import tokenize_text

# 单段返回给模型的文本上限
MAX_TEXT_CHARS = 6000


# --------------------------------------------------------------------------
# 懒加载的运行时对象：列工具不需要连 ES / 调 API，只有真正执行才建连接
# --------------------------------------------------------------------------


@dataclass
class Runtime:
    settings: Settings
    embedder: OpenAICompatEmbedder
    store: ESStore
    reranker: Any = None


_runtime: Runtime | None = None


def get_runtime() -> Runtime:
    global _runtime
    if _runtime is None:
        s = load_settings()
        _runtime = Runtime(
            settings=s,
            embedder=OpenAICompatEmbedder(
                base_url=s.embed_base_url,
                api_key=s.embed_api_key,
                model=s.embed_model,
                batch=s.embed_batch,
            ),
            store=ESStore(s.es_url, s.es_index),
            reranker=(
                build_reranker(s.rerank_url, s.rerank_api_key, s.rerank_model)
                if s.rerank_enabled
                else None
            ),
        )
    return _runtime


def reset_runtime() -> None:
    """测试用：让下次调用重新读配置。"""
    global _runtime
    _runtime = None


# --------------------------------------------------------------------------
# 工具：列表
# --------------------------------------------------------------------------


def kb_list() -> dict:
    """列出库里已入库的论文。"""
    docs = get_runtime().store.list_docs()
    return {
        "ok": True,
        "count": len(docs),
        "documents": [
            {
                "doc_id": d["doc_id"],
                "doc_name": d["doc_name"],
                "chunks": d["chunks"],
                "searchable": d["searchable"],
            }
            for d in docs
        ],
    }


# --------------------------------------------------------------------------
# 工具：检索
# --------------------------------------------------------------------------


def kb_search(query: str, top_n: int = 2, doc_id: str = "") -> dict:
    """检索与问题相关的段落，返回带章节引用的结果。"""
    rt = get_runtime()
    s = rt.settings

    t0 = time.time()
    query_vector = rt.embedder.embed([query])[0]
    t1 = time.time()

    result = retrieve(
        store=rt.store,
        query=query,
        query_vector=query_vector,
        query_tokens=tokenize_text(query),
        reranker=rt.reranker,
        candidates=s.retrieve_candidates,
        top_n=max(1, min(int(top_n), 6)),
        threshold=s.retrieve_threshold,
        char_budget=s.prompt_char_budget,
    )
    t2 = time.time()

    # 计时日志走 stderr：stdout 是 MCP 协议通道，不能写
    print(
        f"[kb_search] 嵌入 {t1 - t0:.2f}s | 检索+重排 {t2 - t1:.2f}s | 合计 {t2 - t0:.2f}s",
        file=sys.stderr,
        flush=True,
    )

    passages = []
    for p in result.passages:
        text = p.content
        truncated = len(text) > MAX_TEXT_CHARS
        if truncated:
            text = text[:MAX_TEXT_CHARS]
        passages.append(
            {
                "sec_no": p.sec_no,
                "sec_path": p.sec_path,
                "doc_name": p.doc_name,
                "score": round(p.score, 4),
                "chars": p.chars,
                "truncated": truncated,
                "text": text,
            }
        )

    return {
        "ok": True,
        "query": query,
        "reranked": result.reranked,
        "logs": result.logs,
        "total_chars": sum(p["chars"] for p in passages),
        "passages": passages,
    }


# --------------------------------------------------------------------------
# 工具：删除
# --------------------------------------------------------------------------


def kb_delete(doc_id: str) -> dict:
    """删除一篇论文的全部记录；doc_id 必须先通过 kb_list 拿到。"""
    store = get_runtime().store
    docs = {d["doc_id"]: d for d in store.list_docs()}
    if doc_id not in docs:
        return {
            "ok": False,
            "error": "not_found",
            "message": f"库里没有 doc_id={doc_id}，先用 kb_list 查",
            "available": sorted(docs),
        }
    deleted = store.delete_by_doc(doc_id)
    store.refresh()
    return {
        "ok": True,
        "doc_id": doc_id,
        "doc_name": docs[doc_id]["doc_name"],
        "deleted": deleted,
        "remaining": store.count(),
    }


# --------------------------------------------------------------------------
# 工具：入库
# --------------------------------------------------------------------------


def kb_ingest(md_path: str, title: str = "", force: bool = False) -> dict:
    """入库一篇 full.md。首次约 10~30 秒；同一文件重复调用命中缓存。"""
    from paper_rag.ingest.runner import ingest_paper

    try:
        report = ingest_paper(
            get_runtime().settings,
            md_path=md_path,
            title=title,
            force=force,
            no_index=False,
        )
    except FileNotFoundError as e:
        return {"ok": False, "error": "file_not_found", "message": str(e)}
    except ValueError as e:
        return {"ok": False, "error": "bad_input", "message": str(e)}
    return report.to_dict()
