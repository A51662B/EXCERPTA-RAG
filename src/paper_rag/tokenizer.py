# -*- coding: utf-8 -*-
"""分词：把正文/标题切成「空格分隔」的字符串，给 ES 的 whitespace 字段用。

为什么自己做而不用 ES 的分析器
------------------------------
1. 中文用 ES 默认分析器会被切成单字，中文检索基本废掉。
2. RAGFlow 的排序是在应用内存里重算的（标题词 ×2、关键词 ×5），
   所以分词结果必须能取回来，不能只在 ES 内部存在。

两种粒度
--------
tokenize_text   粗粒度：按词切（英文按字母数字边界，中文按 2-gram）
fine_tokenize   细粒度：在粗粒度基础上再拆——字母和数字分开
                （"rtx4000" → "rtx" + "4000"），中文词再拆成单字

为什么要 2-gram 而不是单字
--------------------------
没有中文词典时，单字检索噪音太大（"方法"拆成"方"和"法"会和"方向"、"办法"混淆）。
2-gram 保留了一部分搭配信息，是不装 jieba 时的合理折中。
"""
from __future__ import annotations

import re
import unicodedata
from datetime import datetime

from paper_rag.models import Chunk

# 连续的中日韩字符
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+")
# 拉丁字母 / 数字串
WORD_RE = re.compile(r"[a-z0-9]+")
# 字母与数字的交界（细粒度切分用）
ALPHA_DIGIT_RE = re.compile(r"([a-z]+)([0-9]+)|([0-9]+)([a-z]+)")

# 拉丁词的最小长度：1 个字母的 token（s、a）噪音大，默认丢掉
MIN_LATIN_LEN = 2


def _cjk_grams(run: str) -> list[str]:
    """中文串切成 2-gram（1 个字时保留单字）。"""
    if len(run) <= 1:
        return [run]
    return [run[i: i + 2] for i in range(len(run) - 1)]


def tokenize_text(text: str) -> str:
    """粗粒度分词，返回空格分隔的字符串。"""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text).lower()

    out: list[str] = []
    pos = 0
    # 按“中文串 / 其它”分段处理
    for m in CJK_RE.finditer(text):
        out.extend(WORD_RE.findall(text[pos: m.start()]))
        out.extend(_cjk_grams(m.group(0)))
        pos = m.end()
    out.extend(WORD_RE.findall(text[pos:]))

    # 单字母（s、a）噪音大丢掉；纯数字即使是 1 位也保留（页码、表号、指标值都要靠它）
    kept = [t for t in out if len(t) >= MIN_LATIN_LEN or t.isdigit() or CJK_RE.match(t)]
    # 去重会丢词频信息，这里只去掉完全重复的相邻项（压缩存储）
    result: list[str] = []
    for t in kept:
        if not result or result[-1] != t:
            result.append(t)
    return " ".join(result)


def _split_alpha_digit(token: str) -> list[str]:
    """把 'rtx4000' 拆成 ['rtx', '4000']；'abc' 原样返回。"""
    if not ALPHA_DIGIT_RE.fullmatch(token):
        return [token]
    return [p for p in re.split(r"(?<=[a-z])(?=[0-9])|(?<=[0-9])(?=[a-z])", token) if p]


def fine_tokenize(tokens: str) -> str:
    """细粒度分词：在粗分词结果上再拆字母/数字，中文词拆成单字。"""
    if not tokens:
        return ""
    out: list[str] = []
    for token in tokens.split():
        if CJK_RE.fullmatch(token):
            out.extend(list(token))
            continue
        for piece in _split_alpha_digit(token):
            out.append(piece)
    return " ".join(out)


def add_token_fields(chunks: list[Chunk], now: str | None = None) -> None:
    """给每条 chunk 补上分词字段和 create_time（原地修改）。"""
    stamp = now or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for c in chunks:
        c.content_ltks = tokenize_text(c.content)
        c.content_sm_ltks = fine_tokenize(c.content_ltks)
        c.title_tks = tokenize_text(c.title)
        c.title_sm_tks = fine_tokenize(c.title_tks)
        c.create_time = stamp
