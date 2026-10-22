# -*- coding: utf-8 -*-
"""把切块结果和源文件做内容守恒比对。

独立成模块的原因：它是"验收标准"而不是"生产步骤"，以后每个阶段都可以
加自己的核对函数（层级序列校验、合并前后块数校验等）放这里。
"""
from __future__ import annotations

import re
from pathlib import Path

from paper_rag.ingest.markdown_blocks import IMAGE_MD_RE
from paper_rag.models import Block

_HEADING_PREFIX_RE = re.compile(r"^#{1,6}[ \t]+")


def nonws_chars(text: str) -> int:
    """非空白字符数。用它而不是 len()，可以忽略换行/空行被 strip 带来的差异。"""
    return sum(1 for ch in text if not ch.isspace())


def source_nonws_chars(md_path: str | Path, cut_line: int | None) -> int:
    """统计源文件在「截断点之前」的非空白字符数。

    统计时去掉井号前缀和图片标记，与 blocks 的口径保持一致。
    cut_line 为参考文献标题所在行号（1 起），该行及其之后不计入。
    """
    lines = Path(md_path).read_text(encoding="utf-8").splitlines()
    limit = len(lines) if not cut_line else cut_line - 1

    total = 0
    for line in lines[:limit]:
        line = IMAGE_MD_RE.sub("", line)
        line = _HEADING_PREFIX_RE.sub("", line)
        total += nonws_chars(line)
    return total


def blocks_nonws_chars(blocks: list[Block]) -> int:
    return sum(nonws_chars(b.title) + nonws_chars(b.body) for b in blocks)
