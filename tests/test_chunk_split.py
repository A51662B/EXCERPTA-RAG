# -*- coding: utf-8 -*-
"""chunk_split 的单元测试。

重点验证两条规则：
    1. 表格整段不拆
    2. 只按句子切，不做字符硬切（宁可个别块超限，也不切断语义）
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from paper_rag.config import Settings  # noqa: E402
from paper_rag.ingest.chunk_split import (  # noqa: E402
    doc_id_from_title,
    make_id,
    oversize,
    split_blocks,
    split_body,
)
from paper_rag.models import Block  # noqa: E402

# 上限 200、子块目标 60，方便构造小样本
SMALL = Settings(embed_max_chars=200, child_target_chars=60)


def blk(sec_no: str, title: str, body: str) -> Block:
    return Block(idx=0, line=1, md_level=2, title=title, body=body, level=2, sec_no=sec_no,
                 sec_path=f"II. RELATED WORK > {title}")


def paras(n: int, size: int) -> str:
    return "\n\n".join(f"P{i} " + "x" * (size - len(f"P{i} ")) for i in range(n))


class TestSplitBlocks(unittest.TestCase):
    def test_short_block_becomes_solo(self):
        chunks = split_blocks([blk("2.1", "A. Prior", "short text")], SMALL, doc_id="d1")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].kind, "solo")
        self.assertEqual(chunks[0].available_int, 1)
        self.assertEqual(chunks[0].parent_id, "")
        self.assertEqual(chunks[0].sec_no, "2.1")

    def test_over_cap_block_becomes_parent_plus_children(self):
        body = paras(6, 50)          # 6 段 × 50 ≈ 305 字符，超过 200
        chunks = split_blocks([blk("2.1", "A. Prior", body)], SMALL, doc_id="d1")

        parent = chunks[0]
        self.assertEqual(parent.kind, "parent")
        self.assertEqual(parent.available_int, 0)
        self.assertEqual(parent.content, body)

        children = chunks[1:]
        self.assertGreater(len(children), 1)
        for c in children:
            self.assertEqual(c.kind, "child")
            self.assertEqual(c.available_int, 1)
            self.assertEqual(c.parent_id, parent.chunk_id)
            self.assertEqual(c.sec_no, "2.1")

    def test_children_have_title_prefix(self):
        body = paras(6, 50)
        chunks = split_blocks([blk("2.1", "A. Prior", body)], SMALL, doc_id="d1")
        for c in chunks[1:]:
            self.assertTrue(c.content.startswith("A. Prior\n\n"))

    def test_no_prefix_option(self):
        body = paras(6, 50)
        chunks = split_blocks([blk("2.1", "A. Prior", body)], SMALL, doc_id="d1",
                              with_title_prefix=False)
        for c in chunks[1:]:
            self.assertFalse(c.content.startswith("A. Prior"))

    def test_content_is_conserved(self):
        body = paras(8, 40)
        chunks = split_blocks([blk("2.1", "A. Prior", body)], SMALL, doc_id="d1",
                              with_title_prefix=False)
        joined = "\n\n".join(c.content for c in chunks[1:])

        def nonws(s: str) -> int:
            return sum(1 for ch in s if not ch.isspace())

        self.assertEqual(nonws(joined), nonws(body))

    def test_chunks_carry_doc_name_and_id(self):
        chunks = split_blocks([blk("2.1", "A. Prior", "short")], SMALL,
                              doc_id="d1", doc_name="Paper X")
        self.assertEqual(chunks[0].doc_name, "Paper X")
        self.assertEqual(chunks[0].doc_id, "d1")

    def test_ids_unique_and_deterministic(self):
        body = paras(6, 50)
        first = split_blocks([blk("2.1", "A. Prior", body)], SMALL, doc_id="d1")
        second = split_blocks([blk("2.1", "A. Prior", body)], SMALL, doc_id="d1")
        self.assertEqual([c.chunk_id for c in first], [c.chunk_id for c in second])
        self.assertEqual(len({c.chunk_id for c in first}), len(first))

    def test_id_depends_on_doc(self):
        a = split_blocks([blk("1", "T", "same text")], SMALL, doc_id="d1")[0].chunk_id
        b = split_blocks([blk("1", "T", "same text")], SMALL, doc_id="d2")[0].chunk_id
        self.assertNotEqual(a, b)

    def test_parent_id_has_suffix(self):
        chunks = split_blocks([blk("2.1", "A. Prior", paras(6, 50))], SMALL, doc_id="d1")
        self.assertTrue(chunks[0].chunk_id.endswith("-p"))

    def test_normal_split_keeps_every_child_under_cap(self):
        body = paras(10, 40)         # 每段 40，子块目标 60
        chunks = split_blocks([blk("2.1", "A. Prior", body)], SMALL, doc_id="d1")
        self.assertEqual(oversize(chunks, SMALL.embed_max_chars), [])


class TestSplitBody(unittest.TestCase):
    def test_short_body_not_split(self):
        self.assertEqual(split_body("hello", 100), ["hello"])

    def test_split_prefers_paragraph_boundary(self):
        body = "A" * 50 + "\n\n" + "B" * 50 + "\n\n" + "C" * 50
        pieces = split_body(body, 60)
        self.assertEqual([p[0] for p in pieces], ["A", "B", "C"])

    def test_split_by_sentence(self):
        body = "这是一句。" * 30          # 无空行，一句 5 字
        pieces = split_body(body, 40)
        self.assertGreater(len(pieces), 1)
        for p in pieces:
            self.assertLessEqual(len(p), 40)

    def test_table_is_never_split(self):
        table = "<table>" + "<tr><td>x</td></tr>" * 40 + "</table>"   # 远超 cap
        pieces = split_body(table, 40)
        self.assertEqual(len(pieces), 1)
        self.assertEqual(pieces[0], table)

    def test_table_mixed_paragraph_not_split(self):
        body = "Intro sentence. <table>" + "<tr><td>y</td></tr>" * 20 + "</table>"
        pieces = split_body(body, 40)
        self.assertEqual(len(pieces), 1)

    def test_long_sentence_not_hard_split(self):
        # 没有任何句号的一长串 → 保持完整，交给 oversize() 报告
        body = "字" * 300
        pieces = split_body(body, 60)
        self.assertEqual(len(pieces), 1)
        self.assertEqual(pieces[0], body)

    def test_tail_not_merged_when_it_would_exceed_cap(self):
        body = "A" * 600 + "\n\n" + "B" * 600 + "\n\n" + "C" * 100
        pieces = split_body(body, 600)
        # 100 字尾巴虽短，但并进前一块会到 702 > 600，所以不并
        self.assertEqual(len(pieces), 3)
        self.assertTrue(pieces[-1].endswith("C" * 100))

    def test_tail_merged_when_it_fits(self):
        body = "A" * 400 + "\n\n" + "B" * 450 + "\n\n" + "C" * 50
        pieces = split_body(body, 600)   # 450+2+50 = 502 ≤ 600 → 可以并
        self.assertEqual(len(pieces), 2)
        self.assertTrue(pieces[-1].endswith("C" * 50))


class TestDocId(unittest.TestCase):
    def test_title_hash_ignores_case_space_and_punctuation(self):
        a = doc_id_from_title("WDCANet: A Wavelet-Driven Network")
        b = doc_id_from_title("  wdcanet  a wavelet driven network  ")
        self.assertEqual(a, b)

    def test_different_titles_differ(self):
        self.assertNotEqual(doc_id_from_title("Paper A"), doc_id_from_title("Paper B"))

    def test_renaming_folder_does_not_change_id(self):
        title = "A Paper Title"
        self.assertEqual(
            doc_id_from_title(title, r"C:\papers\old name\full.md"),
            doc_id_from_title(title, r"C:\papers\new name\full.md"),
        )

    def test_falls_back_to_path_when_title_empty(self):
        self.assertEqual(len(doc_id_from_title("", r"C:\papers\Some Dir\full.md")), 12)

    def test_make_id_stable(self):
        self.assertEqual(make_id("abc", "d"), make_id("abc", "d"))
        self.assertNotEqual(make_id("abc", "d"), make_id("abc", "e"))


if __name__ == "__main__":
    unittest.main()
