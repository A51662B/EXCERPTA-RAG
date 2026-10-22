# -*- coding: utf-8 -*-
"""es_store 的单元测试：mapping、bulk 载荷、先删后写、混合查询的 boost。

不起真实 ES：覆盖 _request，只验证发出去的请求长什么样。
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from paper_rag.index.es_store import ESStore, build_mapping  # noqa: E402
from paper_rag.models import Chunk  # noqa: E402


def ck(chunk_id: str, kind: str = "solo", vector: list[float] | None = None) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc1",
        doc_name="Paper",
        kind=kind,
        parent_id="" if kind != "child" else "p1-p",
        sec_no="1",
        sec_path="I. INTRO",
        title="I. INTRO",
        content="hello world",
        content_ltks="hello world",
        vector=vector or [0.1, 0.2, 0.3],
    )


class TestMapping(unittest.TestCase):
    def test_vector_field(self):
        m = build_mapping(1024)
        vec = m["mappings"]["properties"]["vector"]
        self.assertEqual(vec["type"], "dense_vector")
        self.assertEqual(vec["dims"], 1024)
        self.assertTrue(vec["index"])

    def test_content_is_not_indexed(self):
        # 对齐 RAGFlow：正文只存不搜，检索靠 content_ltks
        m = build_mapping(8)
        self.assertFalse(m["mappings"]["properties"]["content"]["index"])

    def test_keyword_fields(self):
        props = build_mapping(8)["mappings"]["properties"]
        for f in ("doc_id", "sec_no", "kind", "parent_id", "available_int", "create_time"):
            self.assertIn(f, props)
        self.assertEqual(props["doc_id"]["type"], "keyword")
        self.assertEqual(props["available_int"]["type"], "integer")

    def test_vector_excluded_from_source(self):
        m = build_mapping(8)
        self.assertIn("vector", m["mappings"]["_source"]["excludes"])

    def test_sec_path_has_keyword_subfield(self):
        props = build_mapping(8)["mappings"]["properties"]
        self.assertEqual(props["sec_path"]["type"], "text")
        self.assertIn("keyword", props["sec_path"]["fields"])


class TestRequests(unittest.TestCase):
    def make(self) -> tuple[ESStore, list[tuple]]:
        store = ESStore("http://localhost:9200", "papers")
        store.timeout = 1
        calls: list[tuple] = []

        def fake_request(method, path, body=None, ndjson=None):
            calls.append((method, path, body, ndjson))
            if method == "HEAD":
                return {}
            if path.endswith("/_count"):
                return {"count": 2}
            if path.endswith("/_delete_by_query?refresh=false&conflicts=proceed"):
                return {"deleted": 3}
            if path.startswith("/_bulk"):
                return {"errors": False, "items": [{"index": {}} for _ in range(2)]}
            if path.endswith("/_mget"):
                return {"docs": [{"_id": "p1-p", "found": True, "_source": {"content": "whole"}}]}
            return {}

        store._request = fake_request          # type: ignore[assignment]
        store.index_exists = lambda: False     # type: ignore[assignment]
        return store, calls

    def test_create_index_puts_mapping(self):
        store, calls = self.make()
        msg = store.create_index(1024)
        self.assertIn("已创建索引", msg)
        method, path, body, _ = calls[0]
        self.assertEqual((method, path), ("PUT", "/papers"))
        self.assertEqual(body["mappings"]["properties"]["vector"]["dims"], 1024)

    def test_bulk_payload_shape(self):
        store, calls = self.make()
        stats = store.bulk_index([ck("a"), ck("b")])
        self.assertEqual(stats["indexed"], 2)
        _, path, _, ndjson = calls[0]
        self.assertTrue(path.startswith("/_bulk"))
        lines = [l for l in ndjson.strip().split("\n")]
        self.assertEqual(len(lines), 4)               # 2 条 action + 2 条 doc
        action = json.loads(lines[0])
        self.assertEqual(action["index"]["_id"], "a")
        doc = json.loads(lines[1])
        self.assertIn("vector", doc)
        self.assertNotIn("chars", doc)                # 派生字段不写

    def test_parent_vector_omitted(self):
        store, calls = self.make()
        parent = ck("p1-p", kind="parent")
        parent.vector = []
        store.bulk_index([parent])
        _, _, _, ndjson = calls[0]
        doc = json.loads(ndjson.strip().split("\n")[1])
        self.assertNotIn("vector", doc)               # 空向量会让 dense_vector 报错

    def test_delete_by_doc_uses_term_on_doc_id(self):
        store, calls = self.make()
        deleted = store.delete_by_doc("doc1")
        self.assertEqual(deleted, 3)
        _, path, body, _ = calls[0]
        self.assertIn("_delete_by_query", path)
        self.assertEqual(body["query"], {"term": {"doc_id": "doc1"}})

    def test_search_payload_boosts_and_title_boost(self):
        store, calls = self.make()
        store.search([0.1, 0.2], "wavelet", top=3, k=40)
        _, path, body, _ = calls[0]
        self.assertTrue(path.endswith("/_search"))
        self.assertEqual(body["size"], 3)
        self.assertEqual(body["knn"]["boost"], 0.95)
        self.assertEqual(body["knn"]["k"], 40)
        self.assertEqual(body["query"]["bool"]["boost"], 0.05)
        fields = body["query"]["bool"]["must"][0]["multi_match"]["fields"]
        self.assertIn("title_tks^2", fields)
        self.assertEqual(body["query"]["bool"]["filter"], [{"term": {"available_int": 1}}])

    def test_search_doc_filter(self):
        store, calls = self.make()
        store.search([0.1], "x", doc_id="doc1")
        _, _, body, _ = calls[0]
        self.assertIn({"term": {"doc_id": "doc1"}}, body["query"]["bool"]["filter"])

    def test_get_by_ids_returns_found_only(self):
        store, _ = self.make()
        got = store.get_by_ids(["p1-p", "missing"])
        self.assertIn("p1-p", got)
        self.assertEqual(got["p1-p"]["content"], "whole")


if __name__ == "__main__":
    unittest.main()
