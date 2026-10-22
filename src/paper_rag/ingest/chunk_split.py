# -*- coding: utf-8 -*-
"""阶段 5：把 Block 切成 Chunk（含父子块）。

切分规则（按你的要求）
----------------------
1. 阈值只有一个：`embed_max_chars`（编码模型能吃下的字符数）
       一段 ≤ 上限 → 一条 "solo"
       一段 > 上限 → 一条 "parent"（整节，不编码不检索）+ N 条 "child"
2. **只按句子切**，不做字符级硬切：语义完整优先于大小整齐。
3. **表格整段不拆**：`<table>` 整块作为一个单元，哪怕它本身超过上限。
4. 因此"每条 ≤ 上限"是**目标**而不是硬保证：长句、表格可能超限。
   超限的块由 `oversize()` 挑出来，命令行会打印警告，不会静默通过。

为什么子块只存 parent_id
------------------------
RAGFlow 的做法是每个子块都塞一份父块全文（mom_with_weight），入库时从子块身上
"重建"父块。代价是父块正文被存 N+1 次（一节 13653 字符切 11 段 ≈ 164 KB）。
我们这里父块本来就是 Block，直接生成一条独立的父块记录，子块只存 parent_id，
检索时按 id 批量取（ES 的 mget）。多一次请求，省数倍存储。
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

from paper_rag.config import Settings
from paper_rag.models import Block, Chunk

# 表格（整段识别，不内部拆）
TABLE_RE = re.compile(r"<table\b[^>]*>.*?</table>", re.IGNORECASE | re.DOTALL)

# 段落分隔：空行
PARA_SPLIT_RE = re.compile(r"\n\s*\n")

# 句子边界（中英文）
SENTENCE_RE = re.compile(r"(?<=[。！？；.!?;])\s*")


def make_id(content: str, doc_id: str) -> str:
    """内容哈希当主键：内容相同 → id 相同 → 写入是覆盖。"""
    return hashlib.sha1((content + "\x00" + doc_id).encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------
# 切分
# --------------------------------------------------------------------------


def _split_paragraph(paragraph: str, cap: int) -> list[str]:
    """切一个段落：表格整段保留；其余按句子切，单句超长也不硬切。"""
    if len(paragraph) <= cap:
        return [paragraph]

    if TABLE_RE.search(paragraph):
        return [paragraph]          # 表格不拆

    pieces: list[str] = []
    cur = ""
    for sentence in SENTENCE_RE.split(paragraph):
        if not sentence:
            continue
        if cur and len(cur) + len(sentence) > cap:
            pieces.append(cur)
            cur = sentence
        else:
            cur += sentence
    if cur:
        pieces.append(cur)
    return pieces


def split_body(body: str, cap: int) -> list[str]:
    """把一节正文切成若干块：按段落/句子边界，尽量接近 cap。"""
    body = body.strip()
    if not body:
        return []
    if len(body) <= cap:
        return [body]

    # 1. 按空行分段，超长段落内部按句子切
    units: list[str] = []
    for para in PARA_SPLIT_RE.split(body):
        para = para.strip()
        if not para:
            continue
        units.extend(_split_paragraph(para, cap))

    # 2. 贪心累加，不超过 cap
    pieces: list[str] = []
    cur = ""
    for unit in units:
        if not cur:
            cur = unit
        elif len(cur) + 2 + len(unit) <= cap:
            cur = cur + "\n\n" + unit
        else:
            pieces.append(cur)
            cur = unit
    if cur:
        pieces.append(cur)

    # 3. 尾巴太短，且合起来不超上限，才并进前一块
    tail_min = max(1, cap // 3)
    if len(pieces) > 1 and len(pieces[-1]) < tail_min:
        if len(pieces[-2]) + 2 + len(pieces[-1]) <= cap:
            tail = pieces.pop()
            pieces[-1] = pieces[-1] + "\n\n" + tail

    return pieces


# --------------------------------------------------------------------------
# Block -> Chunk
# --------------------------------------------------------------------------


def _child_content(title: str, piece: str, with_title_prefix: bool, cap: int) -> str:
    """给子块加标题前缀；加完会明显超限时就不加（标题已在 sec_path 字段里）。"""
    if not with_title_prefix or not title.strip():
        return piece
    candidate = f"{title.strip()}\n\n{piece}"
    return candidate if len(candidate) <= cap else piece


def split_block(
    block: Block,
    doc_id: str,
    settings: Settings,
    with_title_prefix: bool = True,
    doc_name: str = "",
) -> list[Chunk]:
    """把一个 Block 变成 1 条或 1+N 条 Chunk。"""
    cap = settings.embed_max_chars
    common = dict(
        doc_id=doc_id,
        doc_name=doc_name,
        sec_no=block.sec_no,
        sec_path=block.sec_path,
        title=block.title.strip(),
    )
    whole = block.body.strip()

    if block.chars <= cap:
        return [
            Chunk(chunk_id=make_id(whole, doc_id), kind="solo", parent_id="", content=whole, **common)
        ]

    parent_id = make_id(whole, doc_id) + "-p"
    chunks: list[Chunk] = [
        Chunk(chunk_id=parent_id, kind="parent", parent_id="", content=whole, **common)
    ]
    for piece in split_body(whole, settings.child_target_chars):
        content = _child_content(block.title, piece, with_title_prefix, cap)
        chunks.append(
            Chunk(
                chunk_id=make_id(content, doc_id),
                kind="child",
                parent_id=parent_id,
                content=content,
                **common,
            )
        )
    return chunks


def split_blocks(
    blocks: list[Block],
    settings: Settings,
    doc_id: str = "",
    with_title_prefix: bool = True,
    doc_name: str = "",
) -> list[Chunk]:
    out: list[Chunk] = []
    for block in blocks:
        if block.chars == 0:
            continue
        out.extend(split_block(block, doc_id, settings, with_title_prefix, doc_name))
    return out


def oversize(chunks: list[Chunk], cap: int) -> list[Chunk]:
    """会被编码、但长度超过上限的 chunk（表格/长句导致，正常会有少量）。"""
    return [c for c in chunks if c.kind != "parent" and c.chars > cap]


# --------------------------------------------------------------------------
# 文档身份
# --------------------------------------------------------------------------
#
# doc_id 用「论文标题」的哈希，不用目录名、也不用 UUID：
#   目录名    改文件夹名 → doc_id 变 → 整篇重复入库
#   UUID      需要持久化映射（RAGFlow 用 MySQL 存），本方案不想引入状态文件
#   标题哈希  改目录名、改正文都不影响；只有改标题才会变

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_title(title: str) -> str:
    t = unicodedata.normalize("NFKC", title or "").strip().lower()
    t = _PUNCT_RE.sub("", t)
    return _WS_RE.sub("", t)


def doc_id_from_title(title: str, fallback_path: str = "") -> str:
    key = normalize_title(title)
    if not key and fallback_path:
        key = normalize_title(Path(fallback_path).parent.name or Path(fallback_path).stem)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
