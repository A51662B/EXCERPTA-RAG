# -*- coding: utf-8 -*-
"""section_index 的单元测试：章节编号与路径。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from paper_rag.ingest.section_index import is_ancestor_of, number_sections  # noqa: E402
from paper_rag.models import Block  # noqa: E402


def make_block(idx: int, md_level: int, title: str, level: int, body: str = "x") -> Block:
    return Block(idx=idx, line=idx + 1, md_level=md_level, title=title, body=body, level=level)


def sample() -> list[Block]:
    return [
        make_block(0, 1, "A Paper Title", 1),
        make_block(1, 2, "I. INTRODUCTION", 1),
        make_block(2, 2, "II. RELATED WORK", 1, body=""),
        make_block(3, 2, "A. Prior Work", 2),
        make_block(4, 2, "B. More Work", 2),
        make_block(5, 2, "III. METHOD", 1, body=""),
        make_block(6, 2, "A. Overall Architecture", 2),
    ]


class TestNumberSections(unittest.TestCase):
    def setUp(self) -> None:
        self.blocks = number_sections(sample())

    def by_title(self, title: str) -> Block:
        return next(b for b in self.blocks if b.title == title)

    def test_document_title_is_zero_and_does_not_consume_chapter_one(self):
        title = self.by_title("A Paper Title")
        self.assertEqual(title.sec_no, "0")
        self.assertEqual(title.sec_path, "A Paper Title")

        intro = self.by_title("I. INTRODUCTION")
        self.assertEqual(intro.sec_no, "1")

    def test_chapters_increment(self):
        self.assertEqual(self.by_title("II. RELATED WORK").sec_no, "2")
        self.assertEqual(self.by_title("III. METHOD").sec_no, "3")

    def test_subsections_use_chapter_prefix(self):
        self.assertEqual(self.by_title("A. Prior Work").sec_no, "2.1")
        self.assertEqual(self.by_title("B. More Work").sec_no, "2.2")
        self.assertEqual(self.by_title("A. Overall Architecture").sec_no, "3.1")

    def test_section_path_is_ancestor_chain(self):
        self.assertEqual(
            self.by_title("A. Prior Work").sec_path,
            "II. RELATED WORK > A. Prior Work",
        )
        self.assertEqual(
            self.by_title("A. Overall Architecture").sec_path,
            "III. METHOD > A. Overall Architecture",
        )

    def test_level_three(self):
        blocks = number_sections(
            [
                make_block(0, 2, "1. Introduction", 1),
                make_block(1, 2, "1.1 Background", 2),
                make_block(2, 2, "1.1.1 Detail", 3),
                make_block(3, 2, "1.2 Motivation", 2),
            ]
        )
        self.assertEqual([b.sec_no for b in blocks], ["1", "1.1", "1.1.1", "1.2"])
        self.assertEqual(blocks[2].sec_path, "1. Introduction > 1.1 Background > 1.1.1 Detail")
        self.assertEqual(blocks[3].sec_path, "1. Introduction > 1.2 Motivation")

    def test_untitled_block_gets_no_number(self):
        blocks = number_sections([make_block(0, 0, "", 1), make_block(1, 2, "I. INTRO", 1)])
        self.assertEqual(blocks[0].sec_no, "")
        self.assertEqual(blocks[1].sec_no, "1")

    def test_is_ancestor_of(self):
        parent = self.by_title("II. RELATED WORK")
        child = self.by_title("A. Prior Work")
        sibling = self.by_title("III. METHOD")
        self.assertTrue(is_ancestor_of(parent, child))
        self.assertFalse(is_ancestor_of(parent, sibling))


if __name__ == "__main__":
    unittest.main()
