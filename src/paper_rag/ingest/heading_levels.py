# -*- coding: utf-8 -*-
"""阶段 2：给每个标题判语义层级。

输入：标题列表（按出现顺序）
输出：与标题等长的层级序列，1=章、2=节、3=目…

设计要点
--------
1. 只用一次 LLM 调用拿整条序列，不逐条问（20 个标题问 20 次是 20 倍成本）。
2. 用 tool call 而不是自由文本：
   - 参数天生是 JSON，不用从散文里扒；
   - finish_reason=tool_calls 能明确判断"它照做了"。
   实测注意：thinking 开着时不允许指定 tool_choice（会 400），所以这里强制 thinking=False。
3. LLM 的输出必须过三层校验：长度、首项、不跳级。
   格式合法 ≠ 内容正确——实测中 thinking 开着时 JSON 完全合法，但层级判错。
4. 校验不过 → 带着错误信息重试一次 → 仍然不过 → 退回规则法，保证流程不断。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from paper_rag.llm import ChatClient

TOOL_NAME = "submit_heading_levels"

TOOL_SPEC = [
    {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": "提交与标题顺序一一对应的层级序列（1=章，2=节，3=目…）",
            "parameters": {
                "type": "object",
                "properties": {
                    "levels": {
                        "type": "array",
                        "items": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                        "description": "层级序列，数量必须与给定标题数量完全一致",
                    }
                },
                "required": ["levels"],
            },
        },
    }
]

SYSTEM_PROMPT = """你是论文章节结构分析器。

输入：一篇论文的标题，以及按文中出现顺序排列的章节标题。
任务：给每个标题标注层级。

层级定义：
    1 = 章：如 "I. INTRODUCTION"、"1. Introduction"、"第1章 绪论"、"一、引言"
    2 = 节：如 "A. Overall Architecture"、"1.1 Motivation"、"1.1 研究背景"
    3 = 目：如 "1.1.1 xxx"、"(a) xxx"

硬性约束：
- 第一个标题必须是 1
- 层级只能逐级加深，不能跳级（不能从 1 直接到 3）
- 所有值必须是 1~5 的整数
- 输出数量必须与标题数量完全一致

判断依据：标题文字里的编号形式（罗马数字、字母、阿拉伯数字、中文数字）优先；
没有编号时，按论文的常规章节顺序推断（摘要/引言 为章，方法/实验小节为节）。"""


def build_messages(paper_title: str, headings: list[str]) -> list[dict]:
    lines = [f"{i}. {t}" for i, t in enumerate(headings)]
    user = (
        f"论文标题：{paper_title or '(未提供)'}\n\n"
        f"章节标题（共 {len(headings)} 个，按顺序）：\n" + "\n".join(lines)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# --------------------------------------------------------------------------
# 解析 LLM 响应
# --------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_BRACE_RE = re.compile(r"\{.*\}", re.DOTALL)


def _coerce_levels(payload) -> list[int] | None:
    """把 {"levels": [...]} 或裸列表统一成 list[int]。"""
    if isinstance(payload, dict):
        payload = payload.get("levels")
    if not isinstance(payload, list) or not payload:
        return None
    out: list[int] = []
    for v in payload:
        if isinstance(v, bool):
            return None
        if isinstance(v, str) and v.strip().isdigit():
            v = int(v.strip())
        if not isinstance(v, int):
            return None
        out.append(v)
    return out


def parse_levels(resp: dict) -> tuple[list[int] | None, str]:
    """从原始响应里取层级序列，返回 (levels, raw_text)。"""
    try:
        message = resp["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return None, ""

    calls = message.get("tool_calls") or []
    if calls:
        raw = calls[0]["function"].get("arguments") or ""
        try:
            return _coerce_levels(json.loads(raw)), raw
        except json.JSONDecodeError:
            return None, raw

    content = message.get("content") or ""
    if not content.strip():
        return None, ""

    # 去掉 ```json 包裹
    m = _FENCE_RE.search(content)
    candidate = m.group(1) if m else content
    m = _BRACE_RE.search(candidate)
    if m:
        candidate = m.group(0)
    try:
        return _coerce_levels(json.loads(candidate)), content
    except json.JSONDecodeError:
        return None, content


# --------------------------------------------------------------------------
# 校验与兜底
# --------------------------------------------------------------------------


def validate_levels(levels: list[int], expected: int) -> list[str]:
    """返回错误列表，空列表表示通过。"""
    errors: list[str] = []
    if len(levels) != expected:
        errors.append(f"数量不符：期望 {expected} 个，实际 {len(levels)} 个")
        return errors  # 长度不对，后面逐项检查没有意义

    if any(not (1 <= v <= 5) for v in levels):
        errors.append("存在超出 1~5 的取值")

    if levels[0] != 1:
        errors.append(f"首项必须是 1，实际是 {levels[0]}")

    for i in range(1, len(levels)):
        if levels[i] > levels[i - 1] + 1:
            errors.append(f"第 {i} 项跳级：{levels[i - 1]} → {levels[i]}")
    return errors


_ROMAN_RE = re.compile(r"^[IVXLC]+\.\s")
_LETTER_RE = re.compile(r"^[A-Z]\.\s")
_NUM3_RE = re.compile(r"^\d+\.\d+\.\d+")
_NUM2_RE = re.compile(r"^\d+\.\d+")
_NUM1_RE = re.compile(r"^\d+\.\s")
_CN_CHAPTER_RE = re.compile(r"^第[零一二三四五六七八九十百0-9]+[章节]")
_CN_LEVEL1_RE = re.compile(r"^[一二三四五六七八九十]+、")


def rule_level(title: str) -> int | None:
    """按编号形式猜层级；猜不出返回 None。"""
    text = title.strip()
    if _NUM3_RE.match(text):
        return 3
    if _NUM2_RE.match(text):
        return 2
    if _NUM1_RE.match(text):
        return 1
    if _ROMAN_RE.match(text):
        return 1
    if _LETTER_RE.match(text):
        return 2
    if _CN_CHAPTER_RE.match(text) or _CN_LEVEL1_RE.match(text):
        return 1
    return None


def rule_based_levels(headings: list[str]) -> list[int]:
    """没有 LLM 时的兜底：能按编号判的就判，判不出沿用上一项。"""
    levels: list[int] = []
    prev = 1
    for i, title in enumerate(headings):
        lv = rule_level(title)
        if lv is None:
            lv = 1 if i == 0 else prev
        levels.append(lv)
        prev = lv
    if levels and levels[0] != 1:
        levels[0] = 1
    return levels


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------


@dataclass
class HeadingLevelResult:
    levels: list[int]
    source: str                      # "llm" | "rule"
    errors: list[str] = field(default_factory=list)
    attempts: int = 0
    raw: str = ""

    @property
    def ok(self) -> bool:
        return self.source == "llm" and not self.errors


def classify_levels(
    headings: list[str],
    paper_title: str = "",
    client: ChatClient | None = None,
    max_attempts: int = 2,
) -> HeadingLevelResult:
    """判层级。client 为 None 时直接用规则法。"""
    if not headings:
        return HeadingLevelResult(levels=[], source="rule")

    if client is None:
        return HeadingLevelResult(levels=rule_based_levels(headings), source="rule")

    messages = build_messages(paper_title, headings)
    last_errors: list[str] = []
    last_raw = ""

    for attempt in range(1, max_attempts + 1):
        resp = client.complete(
            messages,
            tools=TOOL_SPEC,
            tool_choice={"type": "function", "function": {"name": TOOL_NAME}},
            thinking=False,  # 与 named tool_choice 冲突，必须关
        )
        levels, raw = parse_levels(resp)
        last_raw = raw

        if levels is None:
            last_errors = ["无法从响应中解析出层级序列"]
        else:
            last_errors = validate_levels(levels, len(headings))
            if not last_errors:
                return HeadingLevelResult(
                    levels=levels, source="llm", attempts=attempt, raw=raw
                )

        # 带着错误信息重试：把上次的输出当作 assistant 回合，再补一条纠正指令
        messages = build_messages(paper_title, headings) + [
            {"role": "assistant", "content": last_raw or "(空)"},
            {
                "role": "user",
                "content": "上一次提交不合格：" + "；".join(last_errors) + "。请修正后重新调用工具提交。",
            },
        ]

    return HeadingLevelResult(
        levels=rule_based_levels(headings),
        source="rule",
        errors=last_errors,
        attempts=max_attempts,
        raw=last_raw,
    )


def apply_levels(blocks: list, levels: list[int]) -> None:
    """把层级写回 Block。"""
    if len(blocks) != len(levels):
        raise ValueError(f"块数 {len(blocks)} 与层级数 {len(levels)} 不一致")
    for block, lv in zip(blocks, levels):
        block.level = lv
