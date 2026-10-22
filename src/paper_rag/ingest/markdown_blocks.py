# -*- coding: utf-8 -*-
"""把 MinerU 产出的 full.md 切成 blocks。

本模块只负责「切」：不做合并、不判语义层级、不生成位置标签。

切分规则：
    1. 行首 1~6 个 `#` 都是块边界（`#` 与 `##` 一视同仁，避免丢信息）
    2. 图片不进正文：整行只有图片 → 整行丢弃；行内夹着图片 → 删掉图片标记
    3. 遇到 REFERENCES / BIBLIOGRAPHY / 参考文献 标题 → 整篇截断，该标题也不保留
    4. 第一个标题之前的正文单独成一块（title 为空字符串）
    5. 块内保留段落之间的空行，仅去掉首尾空行
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from paper_rag.models import Block

# 行首 1~6 个 #，后面必须跟空白，再跟标题文字
HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.*)$")

# markdown 图片语法：![alt](path)，可能出现在行首、行中、行尾
IMAGE_MD_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")

# 参考文献标题（中英文）
REFERENCES_RE = re.compile(r"^(references|bibliography|参考文献|参考文獻)[ \t]*$", re.IGNORECASE)


@dataclass
class BlockStats:
    """切块过程的统计，用于自检（内容是否守恒）。"""

    total_lines: int = 0
    image_lines_dropped: int = 0      # 整行只有图片，整行丢弃
    inline_images_removed: int = 0    # 行内夹着图片，只删图片标记
    refs_cut_line: int | None = None  # 在第几行遇到参考文献标题

    def to_dict(self) -> dict:
        return {
            "total_lines": self.total_lines,
            "image_lines_dropped": self.image_lines_dropped,
            "inline_images_removed": self.inline_images_removed,
            "refs_cut_line": self.refs_cut_line,
        }


@dataclass
class BlockExtraction:
    """一次切块的完整结果。"""

    source: Path
    blocks: list[Block] = field(default_factory=list)
    stats: BlockStats = field(default_factory=BlockStats)

    @property
    def total_chars(self) -> int:
        return sum(b.chars for b in self.blocks)

    def empty_blocks(self) -> list[Block]:
        return [b for b in self.blocks if b.chars == 0]


def extract_blocks(md_path: str | Path) -> BlockExtraction:
    """把 md 文件切成 blocks。"""
    md_path = Path(md_path)
    lines = md_path.read_text(encoding="utf-8").splitlines()

    result = BlockExtraction(source=md_path, stats=BlockStats(total_lines=len(lines)))
    body_lines: list[str] = []
    blocks = result.blocks
    cur: dict | None = None  # 当前块的临时容器

    def flush() -> None:
        """把当前块收尾，写入 blocks。"""
        if cur is None:
            return
        body = "\n".join(body_lines).strip("\n")
        blocks.append(
            Block(
                idx=cur["idx"],
                line=cur["line"],
                md_level=cur["md_level"],
                title=cur["title"],
                body=body,
                level=None,
            )
        )

    for lineno, line in enumerate(lines, 1):
        m = HEADING_RE.match(line)
        if m:
            title = m.group(2).strip()

            # 参考文献：停在标题处，标题本身也不保留
            # 注意：break 之前必须先 flush，否则参考文献前的最后一块会被丢掉
            if REFERENCES_RE.match(title):
                result.stats.refs_cut_line = lineno
                flush()
                cur = None
                break

            flush()  # 结束上一块
            body_lines = []
            cur = {
                "idx": len(blocks),
                "line": lineno,
                "md_level": len(m.group(1)),
                "title": title,
            }
            continue

        # 图片处理：先删图片标记；整行只剩空白的，整行丢弃
        if IMAGE_MD_RE.search(line):
            cleaned = IMAGE_MD_RE.sub("", line)
            if not cleaned.strip():
                result.stats.image_lines_dropped += 1
                continue
            result.stats.inline_images_removed += 1
            line = cleaned

        # 第一个标题之前的正文：单独成一块
        if cur is None:
            if not line.strip():
                continue
            cur = {"idx": len(blocks), "line": lineno, "md_level": 0, "title": ""}
            body_lines = []

        body_lines.append(line)

    flush()
    return result


def write_blocks_json(blocks: list[Block], json_path: str | Path) -> Path:
    """把 blocks 写成 JSON（对象列表），便于人工审计。"""
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [b.to_dict() for b in blocks]
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return json_path


def dump_blocks(extraction: BlockExtraction, json_path: str | Path) -> Path:
    """把一次切块的结果写成 JSON。"""
    return write_blocks_json(extraction.blocks, json_path)


def load_blocks(json_path: str | Path) -> list[Block]:
    """从 blocks*.json 读回 blocks（跳过解析与层级判定，用于续跑/调试）。"""
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    blocks: list[Block] = []
    for d in data:
        blocks.append(
            Block(
                idx=d["idx"],
                line=d["line"],
                md_level=d["md_level"],
                title=d["title"],
                body=d.get("body", ""),
                level=d.get("level"),
                sec_no=d.get("sec_no", ""),
                sec_path=d.get("sec_path", ""),
                merged_from=list(d.get("merged_from") or []),
            )
        )
    return blocks
