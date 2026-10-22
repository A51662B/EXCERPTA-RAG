# -*- coding: utf-8 -*-
"""阶段 4：把「只有标题、没有正文」的块合并掉。

输入：完成章节编号的 blocks
输出：新的 blocks 列表（空块不再单独存在）

合并规则
--------
1. 空块优先并入**下一块**（保持阅读顺序）。
2. 没有下一块（末尾空块）→ 并入上一块。
3. 连续多个空块 → 一路往下找第一个有正文的块，全部并进去。
4. 合并后，目标块的 sec_no / sec_path 保持它原本的（正文属于它，不属于被并掉的标题）。

标题文字是否要补进正文？
------------------------
如果被并掉的空块是目标块的**上级**，它的标题已经在目标块的 sec_path 里，正文不重复写；
否则（同级、或无法归入路径），把它的标题补一行到正文开头，保证不丢字。
"""
from __future__ import annotations

from paper_rag.ingest.section_index import is_ancestor_of
from paper_rag.models import Block


def _absorb(target: Block, pending: list[Block]) -> None:
    """把 pending 里的空块并入 target。"""
    prefix: list[str] = []
    for empty in pending:
        if not is_ancestor_of(empty, target):
            title = empty.title.strip()
            if title:
                prefix.append(title)
        target.merged_from.append(empty.idx)

    if prefix:
        parts = prefix + ([target.body] if target.body else [])
        target.body = "\n\n".join(parts)


def merge_heading_only(blocks: list[Block]) -> list[Block]:
    """合并空正文的块，返回新的列表。"""
    out: list[Block] = []
    pending: list[Block] = []

    for block in blocks:
        if block.chars == 0:
            pending.append(block)
            continue

        if pending:
            _absorb(block, pending)
            pending = []
        out.append(block)

    # 末尾还剩空块：并入最后一个有正文的块
    if pending:
        if out:
            tail = out[-1]
            extra = [p.title.strip() for p in pending if p.title.strip()]
            tail.merged_from.extend(p.idx for p in pending)
            if extra:
                tail.body = "\n\n".join([tail.body] + extra) if tail.body else "\n\n".join(extra)
        else:
            # 极端情况：整篇都是空块，原样返回，避免全部丢失
            return blocks

    return out
