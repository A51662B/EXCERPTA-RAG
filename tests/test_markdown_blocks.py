# -*- coding: utf-8 -*-
"""markdown_blocks 的单元测试。

用 unittest（不引入 pytest 依赖），跑法：
    python -m unittest discover -s tests
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from paper_rag.ingest.markdown_blocks import extract_blocks  # noqa: E402
from paper_rag.quality import blocks_nonws_chars, source_nonws_chars  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "sample.md"


class TestExtractBlocks(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ex = extract_blocks(FIXTURE)
        cls.blocks = cls.ex.blocks

    # --- 块边界 ---

    def test_first_block_is_the_h1_title(self):
        self.assertEqual(self.blocks[0].title, "A Paper Title")
        self.assertEqual(self.blocks[0].md_level, 1)

    def test_hash_and_double_hash_are_both_boundaries(self):
        found = {(b.title, b.md_level) for b in self.blocks}
        self.assertIn(("I. INTRODUCTION", 2), found)
        self.assertIn(("1.1 Detail level three", 3), found)

    # --- 图片 ---

    def test_image_only_lines_are_dropped(self):
        # 封面图 + 插图，两行都整行丢弃
        self.assertEqual(self.ex.stats.image_lines_dropped, 2)
        for b in self.blocks:
            self.assertNotIn("images/", b.body)

    def test_caption_text_is_kept(self):
        b = next(b for b in self.blocks if b.title == "A. Prior Work")
        self.assertIn("Fig. 1. A caption stays.", b.body)

    # --- 空块与截断 ---

    def test_empty_body_block_is_kept(self):
        b = next(b for b in self.blocks if b.title == "II. RELATED WORK")
        self.assertEqual(b.body, "")
        self.assertEqual(b.chars, 0)

    def test_references_heading_is_excluded(self):
        titles = [b.title for b in self.blocks]
        self.assertNotIn("REFERENCES", titles)
        self.assertIsNotNone(self.ex.stats.refs_cut_line)

    def test_nothing_after_references(self):
        self.assertFalse(any("Someone, Something" in b.body for b in self.blocks))

    # --- 结构完整性 ---

    def test_idx_is_contiguous(self):
        self.assertEqual([b.idx for b in self.blocks], list(range(len(self.blocks))))

    def test_content_is_conserved(self):
        src = source_nonws_chars(FIXTURE, self.ex.stats.refs_cut_line)
        self.assertEqual(src, blocks_nonws_chars(self.blocks))


if __name__ == "__main__":
    unittest.main()
