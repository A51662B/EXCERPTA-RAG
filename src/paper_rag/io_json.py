# -*- coding: utf-8 -*-
"""chunks 的 JSON 读写。

为什么要单独一个模块：切块、向量化、入库三个阶段都要读/写同一份 chunks 文件，
格式定义只应该有一处。
"""
from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

from paper_rag.models import Chunk

# Chunk 的合法字段名，读文件时用来过滤多余键
_FIELDS = {f.name for f in fields(Chunk)}


def write_chunks(chunks: list[Chunk], path: str | Path, include_vectors: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for c in chunks:
        d = c.to_dict()
        if not include_vectors:
            d.pop("vector", None)
        payload.append(d)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_chunks(path: str | Path) -> list[Chunk]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: list[Chunk] = []
    for d in data:
        known = {k: v for k, v in d.items() if k in _FIELDS}
        known.setdefault("vector", [])
        out.append(Chunk(**known))
    return out
