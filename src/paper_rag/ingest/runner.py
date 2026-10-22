# -*- coding: utf-8 -*-
"""入库流水线（可复用）：CLI 的 ingest 和 MCP 的 kb_ingest 都调它。

为什么不直接写在 cli.py 里：CLI 要 print，而 MCP 服务端的 stdout 是协议通道，
一个 print 就会破坏消息流。所以执行逻辑放这里（只返回数据），打印由调用方决定。
"""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from paper_rag.cache import Cache, source_hash
from paper_rag.config import Settings
from paper_rag.index.embedder import OpenAICompatEmbedder, embed_chunks
from paper_rag.index.es_store import ESStore
from paper_rag.ingest.block_merge import merge_heading_only
from paper_rag.ingest.chunk_split import doc_id_from_title, split_blocks
from paper_rag.ingest.heading_levels import apply_levels, classify_levels
from paper_rag.ingest.markdown_blocks import extract_blocks
from paper_rag.ingest.section_index import number_sections
from paper_rag.llm import build_client
from paper_rag.tokenizer import add_token_fields


@dataclass
class IngestReport:
    md_path: str
    doc_id: str
    doc_name: str
    source_hash: str
    cache_dir: str
    cache_levels: bool = False
    cache_chunks: bool = False
    chunks: int = 0
    kinds: dict = field(default_factory=dict)
    levels_from: str = ""
    llm_calls: int = 0
    embedded: int = 0
    indexed: int = 0
    index_errors: int = 0
    deleted_before: int = 0
    index_total: int = 0
    seconds: float = 0.0

    def to_dict(self) -> dict:
        """给 MCP / HTTP 用的结构化结果。"""
        return {
            "ok": self.index_errors == 0,
            "doc_id": self.doc_id,
            "doc_name": self.doc_name,
            "source_hash": self.source_hash,
            "chunks": self.chunks,
            "kinds": self.kinds,
            "levels_from": self.levels_from,
            "cost": {"llm_calls": self.llm_calls, "embedded": self.embedded},
            "indexed": self.indexed,
            "index_errors": self.index_errors,
            "index_total": self.index_total,
            "seconds": round(self.seconds, 1),
        }


def ingest_paper(
    settings: Settings,
    md_path: str | Path,
    title: str = "",
    doc_id: str = "",
    force: bool = False,
    no_index: bool = False,
    no_prefix: bool = False,
) -> IngestReport:
    """把一篇 full.md 入库（含缓存判断）。返回结构化结果，不打印。"""
    started = time.time()
    md = Path(md_path)
    if not md.exists():
        raise FileNotFoundError(f"找不到 {md}")

    cache = Cache(settings.cache_dir)
    src_hash = source_hash(md)

    ex = extract_blocks(str(md))
    headings = [b.title for b in ex.blocks]
    if not headings:
        raise ValueError("没有切出任何块")

    doc_name = title or headings[0]
    final_doc_id = doc_id or doc_id_from_title(doc_name, fallback_path=str(md))
    entry = cache.entry(final_doc_id, src_hash)

    report = IngestReport(
        md_path=str(md),
        doc_id=final_doc_id,
        doc_name=doc_name,
        source_hash=src_hash,
        cache_dir=str(entry.dir),
        cache_levels=entry.has_levels(),
        cache_chunks=entry.has_chunks(),
    )

    if entry.has_chunks() and not force:
        chunks = entry.load_chunks()
        report.levels_from = "缓存"
    else:
        if entry.has_levels() and not force:
            blocks = entry.load_levels()
            report.levels_from = "缓存"
        else:
            client = build_client(
                settings.llm_base_url, settings.llm_api_key, settings.llm_model
            )
            result = classify_levels(headings, paper_title=doc_name, client=client)
            report.llm_calls = result.attempts or 1
            report.levels_from = f"LLM（{result.source}）"
            apply_levels(ex.blocks, result.levels)
            number_sections(ex.blocks)
            blocks = merge_heading_only(ex.blocks)
            entry.save_levels(blocks, source=str(md), doc_name=doc_name)

        chunks = split_blocks(
            blocks,
            settings,
            doc_id=final_doc_id,
            with_title_prefix=not no_prefix,
            doc_name=doc_name,
        )
        embedder = OpenAICompatEmbedder(
            base_url=settings.embed_base_url,
            api_key=settings.embed_api_key,
            model=settings.embed_model,
            batch=settings.embed_batch,
        )
        stats = embed_chunks(chunks, embedder)
        report.embedded = stats.embedded
        add_token_fields(chunks)
        entry.save_chunks(chunks, doc_name=doc_name)

    kinds = Counter(c.kind for c in chunks)
    report.chunks = len(chunks)
    report.kinds = dict(kinds)

    if no_index:
        report.seconds = time.time() - started
        return report

    dims = next((len(c.vector) for c in chunks if c.vector), 0)
    if not dims:
        raise ValueError("这批 chunk 没有向量（缓存可能是旧格式），加 force=True 重跑")

    store = ESStore(settings.es_url, settings.es_index)
    store.create_index(dims, force=False)
    report.deleted_before = store.delete_by_doc(final_doc_id)
    stats_idx = store.bulk_index(chunks, refresh=True)
    store.refresh()
    report.indexed = stats_idx["indexed"]
    report.index_errors = stats_idx["errors"]
    report.index_total = store.count()
    report.seconds = time.time() - started
    return report
