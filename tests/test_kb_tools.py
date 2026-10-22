# -*- coding: utf-8 -*-
"""kb_tools 的单元测试：用假运行时替掉 ES / 嵌入 / 重排，验证工具契约。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from paper_rag import kb_tools  # noqa: E402
from paper_rag.config import Settings  # noqa: E402
from paper_rag.retrieve.pipeline import Passage, RetrieveResult  # noqa: E402


class FakeStore:
    def __init__(self, docs=None):
        self.docs = docs or [
            {"doc_id": "d1", "doc_name": "Paper One", "chunks": 80, "searchable": 68}
        ]
        self.deleted = []

    def list_docs(self):
        return list(self.docs)

    def delete_by_doc(self, doc_id):
        self.deleted.append(doc_id)
        return 80

    def refresh(self):
        pass

    def count(self):
        return 0 if self.deleted else sum(d["chunks"] for d in self.docs)


class FakeEmbedder:
    dims = 4

    def embed(self, texts):
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class FakeReranker:
    def score(self, query, documents):
        return [0.9 for _ in documents]


def make_runtime(store):
    return kb_tools.Runtime(
        settings=Settings(embed_base_url="http://x", embed_api_key="k", es_url="http://x"),
        embedder=FakeEmbedder(),
        store=store,
        reranker=None,
    )


class TestKbList(unittest.TestCase):
    def setUp(self):
        kb_tools._runtime = make_runtime(FakeStore())

    def tearDown(self):
        kb_tools.reset_runtime()

    def test_list(self):
        out = kb_tools.kb_list()
        self.assertTrue(out["ok"])
        self.assertEqual(out["count"], 1)
        self.assertEqual(out["documents"][0]["doc_id"], "d1")


class TestKbDelete(unittest.TestCase):
    def test_not_found_returns_available(self):
        store = FakeStore()
        kb_tools._runtime = make_runtime(store)
        out = kb_tools.kb_delete("nope")
        self.assertFalse(out["ok"])
        self.assertEqual(out["error"], "not_found")
        self.assertEqual(out["available"], ["d1"])
        self.assertEqual(store.deleted, [])

    def test_delete_ok(self):
        store = FakeStore()
        kb_tools._runtime = make_runtime(store)
        out = kb_tools.kb_delete("d1")
        self.assertTrue(out["ok"])
        self.assertEqual(out["deleted"], 80)
        self.assertEqual(store.deleted, ["d1"])


class TestKbSearch(unittest.TestCase):
    def setUp(self):
        self.runtime = make_runtime(FakeStore())
        long_text = "x" * (kb_tools.MAX_TEXT_CHARS + 500)
        original_retrieve = kb_tools.retrieve
        self._orig = original_retrieve

        def fake_retrieve(**kwargs):
            return RetrieveResult(
                passages=[
                    Passage(
                        chunk_id="p1",
                        doc_id="d1",
                        doc_name="Paper One",
                        sec_no="4.4",
                        sec_path="IV. EXPERIMENTS > D. Performance Comparison",
                        title="D. Performance Comparison",
                        content=long_text,
                        score=0.9997,
                        kind="parent",
                    ),
                    Passage(
                        chunk_id="p2",
                        doc_id="d1",
                        doc_name="Paper One",
                        sec_no="4.6",
                        sec_path="IV. EXPERIMENTS > F. Ablation Study",
                        title="F. Ablation Study",
                        content="short",
                        score=0.95,
                        kind="parent",
                    ),
                ],
                candidates=20,
                reranked=True,
                logs=["ES 召回 20 条"],
            )

        kb_tools.retrieve = fake_retrieve
        kb_tools._runtime = self.runtime

    def tearDown(self):
        kb_tools.retrieve = self._orig
        kb_tools.reset_runtime()

    def test_long_passage_is_truncated(self):
        out = kb_tools.kb_search("q", top_n=2)
        self.assertTrue(out["ok"])
        first = out["passages"][0]
        self.assertTrue(first["truncated"])
        self.assertEqual(len(first["text"]), kb_tools.MAX_TEXT_CHARS)
        self.assertEqual(first["chars"], kb_tools.MAX_TEXT_CHARS + 500)

    def test_short_passage_not_truncated(self):
        out = kb_tools.kb_search("q", top_n=2)
        second = out["passages"][1]
        self.assertFalse(second["truncated"])
        self.assertEqual(second["text"], "short")

    def test_top_n_is_clamped(self):
        captured = {}
        original = kb_tools.retrieve

        def capture(**kwargs):
            captured.update(kwargs)
            return original(**kwargs)

        kb_tools.retrieve = capture
        try:
            kb_tools.kb_search("q", top_n=99)
        finally:
            kb_tools.retrieve = original
        self.assertEqual(captured["top_n"], 6)
        self.assertGreaterEqual(captured["top_n"], 1)

    def test_citations_present(self):
        out = kb_tools.kb_search("q")
        p = out["passages"][0]
        self.assertEqual(p["sec_no"], "4.4")
        self.assertIn("Performance Comparison", p["sec_path"])


class TestKbIngest(unittest.TestCase):
    def setUp(self):
        kb_tools._runtime = make_runtime(FakeStore())

    def tearDown(self):
        kb_tools.reset_runtime()

    def test_missing_file_returns_error(self):
        out = kb_tools.kb_ingest(r"C:\definitely\not\here\full.md")
        self.assertFalse(out["ok"])
        self.assertEqual(out["error"], "file_not_found")


if __name__ == "__main__":
    unittest.main()
