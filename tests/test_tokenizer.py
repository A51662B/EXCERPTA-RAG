# -*- coding: utf-8 -*-
"""tokenizer 的单元测试。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from paper_rag.models import Chunk  # noqa: E402
from paper_rag.tokenizer import add_token_fields, fine_tokenize, tokenize_text  # noqa: E402


class TestTokenizeText(unittest.TestCase):
    def test_lowercase_and_punctuation(self):
        self.assertEqual(
            tokenize_text("The Overall Architecture, is illustrated (Fig. 2)."),
            "the overall architecture is illustrated fig 2",
        )

    def test_hyphen_and_slash_split(self):
        self.assertEqual(tokenize_text("encoder–decoder / ResNet18"), "encoder decoder resnet18")

    def test_single_letters_dropped(self):
        # "sens-S ing" 这种断词痕迹，单字母被丢，减少噪音
        self.assertEqual(tokenize_text("sens-S ing"), "sens ing")

    def test_numbers_kept(self):
        self.assertEqual(tokenize_text("84.36% mIoU"), "84 36 miou")

    def test_chinese_bigrams(self):
        self.assertEqual(tokenize_text("语义分割"), "语义 义分 分割")

    def test_chinese_single_char(self):
        self.assertEqual(tokenize_text("图"), "图")

    def test_mixed(self):
        out = tokenize_text("使用 DWT 分解 features")
        self.assertEqual(out, "使用 dwt 分解 features")

    def test_empty(self):
        self.assertEqual(tokenize_text(""), "")


class TestFineTokenize(unittest.TestCase):
    def test_alpha_digit_split(self):
        self.assertEqual(fine_tokenize("rtx4000 a4000"), "rtx 4000 a 4000")

    def test_plain_word_kept(self):
        self.assertEqual(fine_tokenize("segmentation"), "segmentation")

    def test_chinese_split_to_chars(self):
        self.assertEqual(fine_tokenize("语义 义分"), "语 义 义 分")

    def test_empty(self):
        self.assertEqual(fine_tokenize(""), "")


class TestAddTokenFields(unittest.TestCase):
    def test_fills_all_five_fields(self):
        c = Chunk(
            chunk_id="c1", doc_id="d", doc_name="P", kind="solo", parent_id="",
            sec_no="1", sec_path="I. INTRO", title="A. FADA Module",
            content="The FADA module uses DWT."
        )
        add_token_fields([c], now="2026-05-07 10:00:00")

        # 标题里的 "A." 是编号，单字母被丢掉；正文里 "Fig. 2" 的 2 要保留
        self.assertEqual(c.title_tks, "fada module")
        self.assertEqual(c.title_sm_tks, "fada module")
        self.assertEqual(c.content_ltks, "the fada module uses dwt")
        self.assertEqual(c.content_sm_ltks, "the fada module uses dwt")
        self.assertEqual(c.create_time, "2026-05-07 10:00:00")

    def test_default_timestamp_format(self):
        c = Chunk(chunk_id="c1", doc_id="d", doc_name="P", kind="solo", parent_id="",
                  sec_no="1", sec_path="p", title="t", content="x")
        add_token_fields([c])
        self.assertRegex(c.create_time, r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")

    def test_same_timestamp_for_batch(self):
        chunks = [
            Chunk(chunk_id=f"c{i}", doc_id="d", doc_name="P", kind="solo", parent_id="",
                  sec_no="1", sec_path="p", title="t", content=f"text {i}")
            for i in range(3)
        ]
        add_token_fields(chunks, now="2026-01-01 00:00:00")
        self.assertEqual({c.create_time for c in chunks}, {"2026-01-01 00:00:00"})


if __name__ == "__main__":
    unittest.main()
