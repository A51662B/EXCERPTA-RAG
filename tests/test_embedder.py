# -*- coding: utf-8 -*-
"""embedder 的单元测试：父块跳过、批内分批、顺序还原、已有向量复用、JSON 往返。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from paper_rag.index.embedder import OpenAICompatEmbedder, embed_chunks  # noqa: E402
from paper_rag.io_json import load_chunks, write_chunks  # noqa: E402
from paper_rag.models import Chunk  # noqa: E402

# 不用 tempfile：沙箱里临时目录可能不可写。测试专用目录放在 tests/ 下。
TMP_DIR = Path(__file__).resolve().parent / "_tmp"


def ck(chunk_id: str, kind: str, content: str = "text") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="d1",
        doc_name="Paper",
        kind=kind,
        parent_id="p1" if kind == "child" else "",
        sec_no="1",
        sec_path="I. INTRO",
        title="I. INTRO",
        content=content,
    )


class FakeEmbedder:
    """假客户端：向量第 0 位 = 文本长度，便于验证顺序。"""

    def __init__(self, dims: int = 4):
        self.dims = dims
        self.total_tokens = 0
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        self.total_tokens += sum(len(t) for t in texts)
        return [[float(len(t))] + [0.0] * (self.dims - 1) for t in texts]


class TestEmbedChunks(unittest.TestCase):
    def test_parents_are_skipped(self):
        chunks = [ck("p", "parent", "whole section"), ck("c1", "child"), ck("s1", "solo")]
        stats = embed_chunks(chunks, FakeEmbedder())

        self.assertEqual(stats.skipped_parent, 1)
        self.assertEqual(stats.embedded, 2)
        self.assertEqual(chunks[0].vector, [])
        self.assertEqual(len(chunks[1].vector), 4)

    def test_vector_order_matches_input(self):
        chunks = [ck("a", "solo", "a"), ck("b", "solo", "bb"), ck("c", "solo", "ccc")]
        embed_chunks(chunks, FakeEmbedder())
        self.assertEqual([c.vector[0] for c in chunks], [1.0, 2.0, 3.0])

    def test_existing_vectors_are_reused(self):
        chunks = [ck("a", "solo")]
        chunks[0].vector = [9.0, 9.0, 9.0, 9.0]
        client = FakeEmbedder()
        stats = embed_chunks(chunks, client)
        self.assertEqual(stats.skipped_existing, 1)
        self.assertEqual(stats.embedded, 0)
        self.assertEqual(client.calls, 0)

    def test_force_reembeds(self):
        chunks = [ck("a", "solo")]
        chunks[0].vector = [9.0, 9.0, 9.0, 9.0]
        stats = embed_chunks(chunks, FakeEmbedder(), force=True)
        self.assertEqual(stats.embedded, 1)
        self.assertNotEqual(chunks[0].vector[0], 9.0)

    def test_limit(self):
        chunks = [ck(f"c{i}", "solo") for i in range(10)]
        stats = embed_chunks(chunks, FakeEmbedder(), limit=3)
        self.assertEqual(stats.embedded, 3)
        self.assertEqual(sum(1 for c in chunks if c.vector), 3)

    def test_dims_recorded(self):
        stats = embed_chunks([ck("a", "solo")], FakeEmbedder(dims=8))
        self.assertEqual(stats.dims, 8)

    def test_mismatched_return_count_raises(self):
        class Broken(FakeEmbedder):
            def embed(self, texts):
                return [[0.0] * 4]

        with self.assertRaises(RuntimeError):
            embed_chunks([ck("a", "solo"), ck("b", "solo")], Broken())


class TestOpenAICompatEmbedder(unittest.TestCase):
    """不起网络：覆盖 _post，只验证分批与顺序还原。"""

    def make(self, batch: int = 20, reverse: bool = False) -> OpenAICompatEmbedder:
        client = OpenAICompatEmbedder(
            base_url="http://example.invalid", api_key="k", model="m", batch=batch
        )
        client.batches = []          # type: ignore[attr-defined]

        def fake_post(texts):
            client.batches.append(list(texts))  # type: ignore[attr-defined]
            data = [
                {"index": i, "embedding": [float(len(t)), float(i)]} for i, t in enumerate(texts)
            ]
            if reverse:
                data.reverse()
            return {"data": data, "usage": {"total_tokens": sum(len(t) for t in texts)}}

        client._post = fake_post     # type: ignore[assignment]
        return client

    def test_batching(self):
        client = self.make(batch=20)
        client.embed(["x"] * 45)
        self.assertEqual([len(b) for b in client.batches], [20, 20, 5])

    def test_batch_capped_at_25(self):
        client = self.make(batch=100)
        self.assertEqual(client.batch, 25)

    def test_order_restored_when_response_unordered(self):
        client = self.make(batch=10, reverse=True)
        vectors = client.embed(["a", "bb", "ccc"])
        self.assertEqual([v[0] for v in vectors], [1.0, 2.0, 3.0])

    def test_dims_detected_and_tokens_summed(self):
        client = self.make()
        client.embed(["a", "bb"])
        self.assertEqual(client.dims, 2)
        self.assertEqual(client.total_tokens, 3)


class TestChunkJson(unittest.TestCase):
    def setUp(self) -> None:
        TMP_DIR.mkdir(exist_ok=True)

    def tearDown(self) -> None:
        for p in TMP_DIR.glob("*.json"):
            try:
                p.unlink()
            except OSError:
                pass

    def test_roundtrip_keeps_vector(self):
        chunks = [ck("a", "solo", "hello"), ck("p", "parent", "whole")]
        chunks[0].vector = [0.1, 0.2]

        path = write_chunks(chunks, TMP_DIR / "c.json")
        back = load_chunks(path)

        self.assertEqual([c.chunk_id for c in back], ["a", "p"])
        self.assertEqual(back[0].vector, [0.1, 0.2])
        self.assertEqual(back[1].vector, [])

    def test_write_without_vectors(self):
        chunks = [ck("a", "solo")]
        chunks[0].vector = [0.5]
        path = write_chunks(chunks, TMP_DIR / "c2.json", include_vectors=False)
        self.assertNotIn("vector", path.read_text(encoding="utf-8"))

    def test_load_ignores_unknown_keys(self):
        path = TMP_DIR / "c3.json"
        path.write_text(
            '[{"chunk_id":"a","doc_id":"d","doc_name":"n","kind":"solo",'
            '"parent_id":"","sec_no":"1","sec_path":"p","title":"t",'
            '"content":"c","unknown_key":123,"chars":1,"available_int":1}]',
            encoding="utf-8",
        )
        back = load_chunks(path)
        self.assertEqual(len(back), 1)
        self.assertEqual(back[0].chunk_id, "a")


if __name__ == "__main__":
    unittest.main()
