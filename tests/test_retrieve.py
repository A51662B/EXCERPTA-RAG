# -*- coding: utf-8 -*-
"""retrieve.pipeline 的单元测试：重排 → 阈值 → 前 N → 聚合 → 预算。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from paper_rag.retrieve.pipeline import (  # noqa: E402
    aggregate_by_parent,
    apply_budget,
    retrieve,
    rerank_hits,
)


def hit(chunk_id: str, content: str, score: float = 1.0,
        parent_id: str = "", kind: str = "child", sec_no: str = "1") -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": "d1",
        "doc_name": "Paper",
        "sec_no": sec_no,
        "sec_path": f"I. INTRO > {sec_no}",
        "title": "I. INTRO",
        "content": content,
        "kind": kind,
        "parent_id": parent_id,
        "_score": score,
    }


class FakeStore:
    def __init__(self, hits: list[dict], parents: dict[str, dict] | None = None):
        self.hits = hits
        self.parents = parents or {}
        self.search_args: dict = {}

    def search(self, **kwargs):
        self.search_args = kwargs
        return [dict(h) for h in self.hits]

    def get_by_ids(self, ids):
        return {i: self.parents[i] for i in ids if i in self.parents}


class FakeReranker:
    def __init__(self, scores: dict[str, float]):
        self.scores = scores
        self.calls = 0

    def score(self, query, documents):
        self.calls += 1
        out = []
        for doc in documents:
            out.append(next((v for k, v in self.scores.items() if k in doc), 0.0))
        return out


class TestRerank(unittest.TestCase):
    def test_sorted_by_rerank_score(self):
        hits = [hit("a", "alpha text"), hit("b", "beta text"), hit("c", "gamma text")]
        reranker = FakeReranker({"alpha": 0.2, "beta": 0.9, "gamma": 0.5})
        out = rerank_hits(hits, reranker, "q")
        self.assertEqual([h["chunk_id"] for h in out], ["b", "c", "a"])

    def test_single_call_regardless_of_count(self):
        hits = [hit(f"c{i}", f"text {i}") for i in range(50)]
        reranker = FakeReranker({})
        rerank_hits(hits, reranker, "q")
        self.assertEqual(reranker.calls, 1)


class TestAggregate(unittest.TestCase):
    def test_children_of_same_parent_collapse_to_one(self):
        hits = [hit("c1", "p1", score=0.9, parent_id="P1"),
                hit("c2", "p2", score=0.7, parent_id="P1"),
                hit("c3", "p3", score=0.8, parent_id="P2")]
        store = FakeStore(hits, {
            "P1": {"doc_id": "d1", "doc_name": "Paper", "sec_no": "1", "sec_path": "I. INTRO",
                   "title": "I. INTRO", "content": "FULL SECTION 1"},
            "P2": {"doc_id": "d1", "doc_name": "Paper", "sec_no": "2", "sec_path": "II. WORK",
                   "title": "II. WORK", "content": "FULL SECTION 2"},
        })
        out = aggregate_by_parent(hits, store, reranked=True)
        self.assertEqual(len(out), 2)                      # 3 条 → 2 个父块
        first = out[0]
        self.assertEqual(first.kind, "parent")
        self.assertEqual(first.content, "FULL SECTION 1")  # 用父块正文
        self.assertEqual(first.matched_child_id, "c1")     # 分数最高的子块
        self.assertAlmostEqual(first.score, 0.9)           # 分数取 max

    def test_solo_passes_through(self):
        hits = [hit("s1", "solo text", score=0.95, kind="solo")]
        out = aggregate_by_parent(hits, FakeStore(hits), reranked=False)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].kind, "solo")
        self.assertEqual(out[0].content, "solo text")

    def test_missing_parent_falls_back_to_child(self):
        hits = [hit("c1", "child text", score=0.5, parent_id="P9")]
        out = aggregate_by_parent(hits, FakeStore(hits, {}), reranked=True)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].kind, "child")
        self.assertEqual(out[0].content, "child text")


class TestBudget(unittest.TestCase):
    def test_keeps_until_budget(self):
        from paper_rag.retrieve.pipeline import Passage
        ps = [Passage(chunk_id=str(i), doc_id="d", doc_name="P", sec_no="1", sec_path="p",
                      title="t", content="x" * 100, score=1.0 - i * 0.1, kind="solo")
              for i in range(3)]
        kept, dropped = apply_budget(ps, 250)
        self.assertEqual(len(kept), 2)
        self.assertEqual(dropped, 1)

    def test_falls_back_to_child_when_parent_too_big(self):
        from paper_rag.retrieve.pipeline import Passage
        p = Passage(chunk_id="P1", doc_id="d", doc_name="P", sec_no="1", sec_path="p",
                    title="t", content="X" * 5000, score=1.0, kind="parent",
                    fallback_content="small child")
        kept, dropped = apply_budget([p], 1000)
        self.assertEqual(dropped, 0)
        self.assertEqual(kept[0].kind, "child")
        self.assertEqual(kept[0].content, "small child")


class TestRetrievePipeline(unittest.TestCase):
    def make_store(self):
        hits = [
            hit("c1", "alpha", score=1.0, parent_id="P1"),
            hit("c2", "beta", score=0.9, parent_id="P1"),
            hit("c3", "gamma", score=0.8, parent_id="P2"),
            hit("s1", "delta", score=0.7, kind="solo"),
        ]
        parents = {
            "P1": {"doc_id": "d1", "doc_name": "Paper", "sec_no": "1", "sec_path": "I. INTRO",
                   "title": "I. INTRO", "content": "SEC1"},
            "P2": {"doc_id": "d1", "doc_name": "Paper", "sec_no": "2", "sec_path": "II. WORK",
                   "title": "II. WORK", "content": "SEC2"},
        }
        return FakeStore(hits, parents)

    def test_full_flow_with_rerank(self):
        store = self.make_store()
        reranker = FakeReranker({"alpha": 0.9, "beta": 0.4, "gamma": 0.8, "delta": 0.5})
        r = retrieve(store=store, query="q", query_vector=[0.1], query_tokens="q",
                     reranker=reranker, candidates=10, top_n=4, threshold=0.0,
                     char_budget=1000)
        # 4 条候选 → 3 个父块/独立块
        self.assertEqual(len(r.passages), 3)
        self.assertTrue(r.reranked)
        self.assertEqual(store.search_args["top"], 10)

    def test_threshold_drops_low_scores(self):
        store = self.make_store()
        reranker = FakeReranker({"alpha": 0.9, "beta": 0.2, "gamma": 0.8, "delta": 0.1})
        r = retrieve(store=store, query="q", query_vector=[0.1], query_tokens="q",
                     reranker=reranker, candidates=10, top_n=4, threshold=0.5,
                     char_budget=1000)
        self.assertEqual(r.dropped_by_threshold, 2)
        self.assertEqual(len(r.passages), 2)

    def test_top_n_truncates_before_aggregation(self):
        store = self.make_store()
        reranker = FakeReranker({"alpha": 0.9, "beta": 0.85, "gamma": 0.8, "delta": 0.1})
        r = retrieve(store=store, query="q", query_vector=[0.1], query_tokens="q",
                     reranker=reranker, candidates=10, top_n=2, threshold=0.0,
                     char_budget=1000)
        # 前 2 条都是 P1 的子块 → 聚合后只剩 1 条（RAGFlow 的行为）
        self.assertEqual(len(r.passages), 1)
        self.assertEqual(r.passages[0].kind, "parent")

    def test_without_rerank_uses_es_score(self):
        store = self.make_store()
        r = retrieve(store=store, query="q", query_vector=[0.1], query_tokens="q",
                     reranker=None, candidates=10, top_n=4, threshold=0.0,
                     char_budget=1000)
        self.assertFalse(r.reranked)
        self.assertEqual(r.passages[0].score, 1.0)


if __name__ == "__main__":
    unittest.main()
