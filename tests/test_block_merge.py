# -*- coding: utf-8 -*-
"""block_merge 的单元测试：空块合并的三种情况 + 不丢字。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from paper_rag.ingest.block_merge import merge_heading_only  # noqa: E402
from paper_rag.ingest.section_index import number_sections  # noqa: E402
from paper_rag.models import Block  # noqa: E402


def b(idx: int, title: str, level: int, body: str) -> Block:
    return Block(idx=idx, line=idx + 1, md_level=2, title=title, body=body, level=level)


class TestMergeHeadingOnly(unittest.TestCase):
    def test_parent_heading_absorbed_by_first_child(self):
        blocks = number_sections(
            [
                b(0, "I. INTRODUCTION", 1, "intro text"),
                b(1, "II. RELATED WORK", 1, ""),
                b(2, "A. Prior Work", 2, "prior text"),
            ]
        )
        out = merge_heading_only(blocks)
        self.assertEqual([x.title for x in out], ["I. INTRODUCTION", "A. Prior Work"])

        child = out[1]
        self.assertEqual(child.sec_no, "2.1")                       # 位置取有正文的那块
        self.assertEqual(child.body, "prior text")                 # 标题已在 sec_path，不重复写
        self.assertIn("II. RELATED WORK", child.sec_path)
        self.assertEqual(child.merged_from, [1])

    def test_same_level_empty_heading_title_is_prepended(self):
        blocks = number_sections(
            [
                b(0, "I. INTRODUCTION", 1, "intro text"),
                b(1, "II. FUTURE WORK", 1, ""),   # 同级空块，路径里不会包含
                b(2, "III. CONCLUSION", 1, "conclusion text"),
            ]
        )
        out = merge_heading_only(blocks)
        last = out[-1]
        self.assertEqual(last.title, "III. CONCLUSION")
        self.assertTrue(last.body.startswith("II. FUTURE WORK\n\n"))
        self.assertIn("conclusion text", last.body)
        self.assertEqual(last.merged_from, [1])

    def test_consecutive_empty_blocks_all_merged(self):
        blocks = number_sections(
            [
                b(0, "I. INTRO", 1, "text"),
                b(1, "II. A", 1, ""),
                b(2, "III. B", 1, ""),
                b(3, "IV. C", 1, "target text"),
            ]
        )
        out = merge_heading_only(blocks)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[-1].merged_from, [1, 2])
        self.assertIn("II. A", out[-1].body)
        self.assertIn("III. B", out[-1].body)

    def test_trailing_empty_merged_into_previous(self):
        blocks = number_sections(
            [
                b(0, "I. INTRO", 1, "text"),
                b(1, "II. APPENDIX", 1, ""),
            ]
        )
        out = merge_heading_only(blocks)
        self.assertEqual(len(out), 1)
        self.assertTrue(out[0].body.endswith("II. APPENDIX"))
        self.assertEqual(out[0].merged_from, [1])

    def test_nothing_lost(self):
        blocks = number_sections(
            [
                b(0, "I. INTRO", 1, "aaaa"),
                b(1, "II. PARENT", 1, ""),
                b(2, "A. Child", 2, "bbbb"),
                b(3, "III. SIBLING", 1, ""),
                b(4, "IV. NEXT", 1, "cccc"),
                b(5, "V. TAIL", 1, ""),
            ]
        )
        empty_titles = [x.title for x in blocks if x.chars == 0]
        out = merge_heading_only(blocks)
        haystack = "\n".join(x.title + "\n" + x.body + "\n" + x.sec_path for x in out)
        for title in empty_titles:
            self.assertIn(title, haystack, f"空块标题丢失：{title}")

    def test_all_empty_returns_original(self):
        blocks = number_sections([b(0, "I. A", 1, ""), b(1, "II. B", 1, "")])
        out = merge_heading_only(blocks)
        self.assertEqual(len(out), 2)


if __name__ == "__main__":
    unittest.main()
