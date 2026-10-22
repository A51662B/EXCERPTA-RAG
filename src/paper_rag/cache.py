# -*- coding: utf-8 -*-
"""中间结果缓存：同一篇论文重跑时不再花 LLM 和 embedding 的钱。

缓存键 = 标题哈希 + 源文件哈希
------------------------------
只按标题不行：同一篇论文重新解析过（MinerU 换了参数、重跑了一次），
`full.md` 变了，但标题没变。用标题做键会把**旧内容的切块结果**当成新的。
源文件哈希一变，键就变，缓存自然失效——这是"正确性优先"的选择。

目录结构
--------
    data/cache/<doc_id>/<source_hash[:12]>/
        meta.json     版本、来源、时间、阶段标记
        levels.json   带层级和编号的 blocks（阶段 2~3 的产物）
        chunks.json   带分词和向量的最终 chunk（阶段 5~6 的产物）

CACHE_VERSION 一变，所有旧缓存自动视为失效。
"""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from paper_rag.ingest.markdown_blocks import load_blocks, write_blocks_json
from paper_rag.io_json import load_chunks, write_chunks
from paper_rag.models import Block, Chunk

# 切块逻辑 / 分词逻辑 / 字段结构一变，就把这个数字 +1
CACHE_VERSION = 1


def source_hash(path: str | Path) -> str:
    """源文件哈希（行尾归一化，避免 CRLF/LF 造成假失效）。"""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


@dataclass
class CacheEntry:
    root: Path
    doc_id: str
    hash_prefix: str

    @property
    def dir(self) -> Path:
        return self.root / self.doc_id / self.hash_prefix

    @property
    def meta_path(self) -> Path:
        return self.dir / "meta.json"

    @property
    def levels_path(self) -> Path:
        return self.dir / "levels.json"

    @property
    def chunks_path(self) -> Path:
        return self.dir / "chunks.json"

    # -- meta --------------------------------------------------------------

    def meta(self) -> dict:
        if not self.meta_path.exists():
            return {}
        try:
            return json.loads(self.meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _write_meta(self, **updates) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        meta = self.meta()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        meta.setdefault("created_at", now)      # 首次写入时记下创建时间
        meta.update({"version": CACHE_VERSION, "doc_id": self.doc_id})
        meta.update(updates)
        meta["updated_at"] = now
        self.meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _valid(self) -> bool:
        return self.meta().get("version") == CACHE_VERSION

    # -- levels ------------------------------------------------------------

    def has_levels(self) -> bool:
        return self._valid() and self.levels_path.exists()

    def load_levels(self) -> list[Block]:
        return load_blocks(self.levels_path)

    def save_levels(self, blocks: list[Block], source: str = "", doc_name: str = "") -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        write_blocks_json(blocks, self.levels_path)
        stages = dict(self.meta().get("stages", {}))
        stages["levels"] = True
        self._write_meta(stages=stages, source=source, doc_name=doc_name)

    # -- chunks ------------------------------------------------------------

    def has_chunks(self) -> bool:
        return self._valid() and self.chunks_path.exists()

    def load_chunks(self) -> list[Chunk]:
        return load_chunks(self.chunks_path)

    def save_chunks(self, chunks: list[Chunk], doc_name: str = "") -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        write_chunks(chunks, self.chunks_path)
        meta = self.meta().get("stages", {})
        meta["chunks"] = True
        self._write_meta(
            stages=meta,
            doc_name=doc_name or self.meta().get("doc_name", ""),
            chunk_count=len(chunks),
        )

    def summary(self) -> dict:
        m = self.meta()
        return {
            "doc_id": self.doc_id,
            "doc_name": m.get("doc_name", ""),
            "hash": self.hash_prefix,
            "levels": self.has_levels(),
            "chunks": self.has_chunks(),
            "chunk_count": m.get("chunk_count", 0),
            "created_at": m.get("created_at", ""),
            "updated_at": m.get("updated_at", ""),
        }


class Cache:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def entry(self, doc_id: str, src_hash: str) -> CacheEntry:
        return CacheEntry(root=self.root, doc_id=doc_id, hash_prefix=src_hash[:12])

    def entries(self) -> list[dict]:
        if not self.root.exists():
            return []
        out = []
        for doc_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            for hash_dir in sorted(p for p in doc_dir.iterdir() if p.is_dir()):
                out.append(CacheEntry(self.root, doc_dir.name, hash_dir.name).summary())
        return out

    def clear(self, doc_id: str = "") -> int:
        """清缓存；doc_id 为空时清全部。返回删掉的目录数。"""
        if not self.root.exists():
            return 0
        targets = [self.root / doc_id] if doc_id else [p for p in self.root.iterdir()]
        removed = 0
        for t in targets:
            if t.is_dir():
                shutil.rmtree(t, ignore_errors=True)
                removed += 1
        return removed
