# -*- coding: utf-8 -*-
"""heading_levels 的单元测试：解析、校验、规则兜底、重试降级。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from paper_rag.ingest.heading_levels import (  # noqa: E402
    classify_levels,
    parse_levels,
    rule_based_levels,
    validate_levels,
)

HEADINGS = [
    "WDCANet: A Wavelet-Driven ...",
    "I. INTRODUCTION",
    "II. RELATED WORK",
    "A. Frequency-Aware Modeling ...",
    "B. Attention Mechanisms ...",
    "III. METHOD",
    "A. Overall Architecture",
    "B. Frequency-Aware Directional Attention Module",
    "IV. EXPERIMENTS",
    "A. Datasets",
    "V. DISCUSSION",
    "VI. CONCLUSION",
]
GOOD = [1, 1, 1, 2, 2, 1, 2, 2, 1, 2, 1, 1]


def tool_response(arguments: str) -> dict:
    return {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "submit_heading_levels", "arguments": arguments},
                        }
                    ],
                },
            }
        ]
    }


def text_response(content: str) -> dict:
    return {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": content}}]}


class TestParseLevels(unittest.TestCase):
    def test_parse_tool_call(self):
        levels, raw = parse_levels(tool_response('{"levels": [1,2,2]}'))
        self.assertEqual(levels, [1, 2, 2])
        self.assertIn("1,2,2", raw)

    def test_parse_plain_json_content(self):
        levels, _ = parse_levels(text_response('{"levels": [1,1,2]}'))
        self.assertEqual(levels, [1, 1, 2])

    def test_parse_json_in_code_fence(self):
        levels, _ = parse_levels(text_response('```json\n{"levels": [1,2]}\n```'))
        self.assertEqual(levels, [1, 2])

    def test_parse_bare_array_content(self):
        levels, _ = parse_levels(text_response("[1, 2, 3]"))
        self.assertEqual(levels, [1, 2, 3])

    def test_parse_rejects_prose(self):
        levels, _ = parse_levels(text_response("我觉得应该是 1,2,2"))
        self.assertIsNone(levels)

    def test_parse_rejects_bool(self):
        levels, _ = parse_levels(tool_response('{"levels": [true, 1]}'))
        self.assertIsNone(levels)

    def test_parse_handles_missing_message(self):
        self.assertEqual(parse_levels({"choices": []}), (None, ""))


class TestValidateLevels(unittest.TestCase):
    def test_good(self):
        self.assertEqual(validate_levels(GOOD, len(GOOD)), [])

    def test_length_mismatch(self):
        errs = validate_levels([1, 1], 5)
        self.assertTrue(any("数量不符" in e for e in errs))

    def test_first_must_be_one(self):
        errs = validate_levels([2, 2, 2], 3)
        self.assertTrue(any("首项" in e for e in errs))

    def test_no_level_jump(self):
        errs = validate_levels([1, 3, 3], 3)
        self.assertTrue(any("跳级" in e for e in errs))

    def test_range(self):
        errs = validate_levels([1, 9], 2)
        self.assertTrue(any("1~5" in e for e in errs))


class TestRuleFallback(unittest.TestCase):
    def test_roman_is_chapter_letter_is_section(self):
        levels = rule_based_levels(["I. INTRODUCTION", "A. Datasets", "II. METHOD"])
        self.assertEqual(levels, [1, 2, 1])

    def test_numeric_depth(self):
        levels = rule_based_levels(["1. Introduction", "1.1 Background", "1.1.1 Detail", "2. Method"])
        self.assertEqual(levels, [1, 2, 3, 1])

    def test_unknown_inherits_previous(self):
        levels = rule_based_levels(["I. INTRODUCTION", "Abstract", "A. Datasets"])
        self.assertEqual(levels, [1, 1, 2])


class FakeClient:
    """按预设序列依次返回响应，用于测试重试逻辑。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def complete(self, messages, **kwargs):
        self.calls += 1
        return self.responses.pop(0)


class TestClassifyLevels(unittest.TestCase):
    def test_llm_ok_first_try(self):
        client = FakeClient([tool_response(f'{{"levels": {GOOD}}}')])
        r = classify_levels(HEADINGS, "T", client=client)
        self.assertEqual(r.source, "llm")
        self.assertEqual(r.levels, GOOD)
        self.assertEqual(r.attempts, 1)
        self.assertEqual(client.calls, 1)

    def test_retry_then_success(self):
        bad = tool_response('{"levels": [1,1]}')  # 长度不对
        client = FakeClient([bad, tool_response(f'{{"levels": {GOOD}}}')])
        r = classify_levels(HEADINGS, "T", client=client)
        self.assertEqual(r.source, "llm")
        self.assertEqual(r.attempts, 2)
        self.assertEqual(client.calls, 2)

    def test_gives_up_and_falls_back(self):
        bad = tool_response('{"levels": [1,1]}')
        client = FakeClient([bad, bad])
        r = classify_levels(HEADINGS, "T", client=client)
        self.assertEqual(r.source, "rule")
        self.assertEqual(r.levels, rule_based_levels(HEADINGS))
        self.assertTrue(r.errors)

    def test_no_client_uses_rules(self):
        r = classify_levels(HEADINGS, "T", client=None)
        self.assertEqual(r.source, "rule")
        self.assertEqual(len(r.levels), len(HEADINGS))

    def test_empty_headings(self):
        r = classify_levels([], "T", client=None)
        self.assertEqual(r.levels, [])


if __name__ == "__main__":
    unittest.main()
