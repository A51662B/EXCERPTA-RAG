# -*- coding: utf-8 -*-
"""缓存模块的单元测试：键、版本失效、往返、清理。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import paper_rag.cache as cache_mod  # noqa: E402
from paper_rag.cache import Cache, source_hash  # noqa: E402
from paper_rag.models import Block, Chunk  # noqa: E402

TMP = Path(__file__).resolve().parent / "_tmp_cache"


def blk(title: str, body: str, level: int = 2) -> Block:
    return Block(idx=0, line=1, md_level=2, title=title, body=body, level=level,
                 sec_no="1", sec_path=title)


def ck(cid: str, kind: str = "solo") -> Chunk:
    return Chunk(chunk_id=cid, doc_id="d1", doc_name="Paper", kind=kind,
                 parent_id="", sec_no="1", sec_path="p", title="t", content="text",
                 content_ltks="text", vector=[0.1, 0.2])


class TestSourceHash(unittest.TestCase):
    def setUp(self):
        TMP.mkdir(exist_ok=True)

    def tearDown(self):
        for p in TMP.glob("*"):
            p.unlink()

    def test_crlf_and_lf_hash_the_same(self):
        a = TMP / "a.md"
        b = TMP / "b.md"
        a.write_bytes(b"# Title\r\nbody\r\n")
        b.write_bytes(b"# Title\nbody\n")
        self.assertEqual(source_hash(a), source_hash(b))

    def test_content_change_changes_hash(self):
        a = TMP / "a.md"
        a.write_text("# A\nbody", encoding="utf-8")
        h1 = source_hash(a)
        a.write_text("# A\nbody changed", encoding="utf-8")
        self.assertNotEqual(h1, source_hash(a))


class TestCacheEntry(unittest.TestCase):
    def setUp(self):
        TMP.mkdir(exist_ok=True)
        self.cache = Cache(TMP)

    def tearDown(self):
        self.cache.clear()

    def test_miss_on_empty(self):
        e = self.cache.entry("doc1", "abcdef123456")
        self.assertFalse(e.has_levels())
        self.assertFalse(e.has_chunks())

    def test_levels_roundtrip(self):
        e = self.cache.entry("doc1", "abcdef123456")
        blocks = [blk("A", "body a"), blk("B", "body b")]
        e.save_levels(blocks, source="full.md", doc_name="Paper")
        self.assertTrue(e.has_levels())
        back = e.load_levels()
        self.assertEqual([b.title for b in back], ["A", "B"])
        self.assertEqual(e.meta()["doc_name"], "Paper")

    def test_chunks_roundtrip_keeps_vector(self):
        e = self.cache.entry("doc1", "abcdef123456")
        chunks = [ck("c1"), ck("c2")]
        e.save_chunks(chunks, doc_name="Paper")
        self.assertTrue(e.has_chunks())
        back = e.load_chunks()
        self.assertEqual(len(back), 2)
        self.assertEqual(back[0].vector, [0.1, 0.2])
        self.assertEqual(e.summary()["chunk_count"], 2)

    def test_version_bump_invalidates(self):
        e = self.cache.entry("doc1", "abcdef123456")
        e.save_chunks([ck("c1")], doc_name="Paper")
        self.assertTrue(e.has_chunks())

        old = cache_mod.CACHE_VERSION
        cache_mod.CACHE_VERSION = old + 1
        try:
            self.assertFalse(e.has_chunks())   # 版本不匹配 → 视为未命中
            self.assertFalse(e.has_levels())
        finally:
            cache_mod.CACHE_VERSION = old

    def test_different_source_hash_is_different_entry(self):
        e1 = self.cache.entry("doc1", "aaaa1111")
        e1.save_chunks([ck("c1")], doc_name="Paper")
        e2 = self.cache.entry("doc1", "bbbb2222")
        self.assertFalse(e2.has_chunks())
        self.assertEqual(len(self.cache.entries()), 1)


class TestCacheList(unittest.TestCase):
    def setUp(self):
        TMP.mkdir(exist_ok=True)
        self.cache = Cache(TMP)

    def tearDown(self):
        self.cache.clear()

    def test_entries_and_clear_one(self):
        self.cache.entry("doc1", "aaaa").save_chunks([ck("c1")], doc_name="P1")
        self.cache.entry("doc2", "bbbb").save_chunks([ck("c2")], doc_name="P2")
        self.assertEqual(len(self.cache.entries()), 2)

        self.assertEqual(self.cache.clear("doc1"), 1)
        remaining = self.cache.entries()
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["doc_id"], "doc2")

    def test_clear_all(self):
        self.cache.entry("doc1", "aaaa").save_chunks([ck("c1")], doc_name="P1")
        self.assertGreaterEqual(self.cache.clear(), 1)
        self.assertEqual(self.cache.entries(), [])


if __name__ == "__main__":
    unittest.main()
