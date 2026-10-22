# -*- coding: utf-8 -*-
"""数据模型：流水线各阶段在内存里传递的结构。

Block   = 一节（标题 + 正文），是"切片前"的形态，一份文档 20 个左右
Chunk   = 一条可入库记录，是"切片后"的形态，可能比 Block 多（父子块是 1:N）

原则：内存里始终是结构化对象，只有写盘或交给下游库时才降级成 dict。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Block:
    """一个「标题 + 正文」块，是切块阶段的产物。

    idx         顺序号，从 0 开始
    line        标题在原 md 中的行号（1 起）
    md_level    标题的井号个数（# → 1，## → 2）；首个标题之前的正文记 0
    title       标题文字
    body        正文，保留段落之间的空行，仅去掉首尾空行
    level       语义层级（1=章，2=节…）；切块阶段为 None，由层级识别阶段填
    sec_no      章节编号，如 "3.2"；题目块为 "0"
    sec_path    章节路径，如 "III. METHOD > A. Overall Architecture"
    merged_from 被并进这一块的空块 idx 列表（阶段 4 填）
    """

    idx: int
    line: int
    md_level: int
    title: str
    body: str = ""
    level: int | None = None
    sec_no: str = ""
    sec_path: str = ""
    merged_from: list[int] = field(default_factory=list)

    @property
    def chars(self) -> int:
        return len(self.body)

    @property
    def is_document_title(self) -> bool:
        """文档题目块：整篇第一个块，且是真正的 H1（MinerU 用 # 标题目）。"""
        return self.idx == 0 and self.md_level == 1 and bool(self.title)

    def to_dict(self) -> dict[str, Any]:
        return {
            "idx": self.idx,
            "line": self.line,
            "md_level": self.md_level,
            "title": self.title,
            "level": self.level,
            "sec_no": self.sec_no,
            "sec_path": self.sec_path,
            "body": self.body,
            "chars": self.chars,
            "merged_from": list(self.merged_from),
        }


@dataclass
class Chunk:
    """一条可入库记录（父子块机制下的父、子、独立块统一用这个结构）。

    chunk_id      内容哈希，同时作为 ES 的 _id
    doc_id        文档标识（一篇论文一个），机器用：过滤、删除、找父块
    doc_name      文档名（论文标题），人用：检索结果里显示来源
    kind          "parent" | "child" | "solo"
    parent_id     子块指向父块；父块与独立块为空
    sec_no        章节编号，从 Block 继承
    sec_path      章节路径，从 Block 继承
    title         标题文字，从 Block 继承
    content       入库正文（父块=整节；子块=标题前缀 + 该段；独立块=整节）
    content_ltks  正文粗分词（空格分隔），给检索/排序用
    content_sm_ltks 正文细分词
    title_tks     标题粗分词（排序时权重 ×2）
    title_sm_tks  标题细分词
    create_time   入库时间
    vector        向量；父块不编码，保持空列表
    """

    chunk_id: str
    doc_id: str
    doc_name: str
    kind: str
    parent_id: str
    sec_no: str
    sec_path: str
    title: str
    content: str
    content_ltks: str = ""
    content_sm_ltks: str = ""
    title_tks: str = ""
    title_sm_tks: str = ""
    create_time: str = ""
    vector: list[float] = field(default_factory=list)

    @property
    def chars(self) -> int:
        return len(self.content)

    @property
    def available_int(self) -> int:
        """1 = 参与检索；0 = 只作为父块被回取。"""
        return 0 if self.kind == "parent" else 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "doc_name": self.doc_name,
            "kind": self.kind,
            "parent_id": self.parent_id,
            "sec_no": self.sec_no,
            "sec_path": self.sec_path,
            "title": self.title,
            "content": self.content,
            "content_ltks": self.content_ltks,
            "content_sm_ltks": self.content_sm_ltks,
            "title_tks": self.title_tks,
            "title_sm_tks": self.title_sm_tks,
            "create_time": self.create_time,

            "vector": self.vector,
            "chars": self.chars,
            "available_int": self.available_int,
        }
