# -*- coding: utf-8 -*-
"""阶段 7：写入 Elasticsearch。

mapping 的设计对齐 RAGFlow（见 conf 里的 mapping.json）：
    - 正文 content 只存不索引（index:false），检索靠 content_ltks
    - content_ltks / title_tks 等用 whitespace 分析器（词是我们自己切好的）
    - doc_id / sec_no / available_int 用 keyword/integer，只做过滤与聚合
    - vector 用 dense_vector(cosine)，dims 必须与模型一致

两处有意的差异：
    1. _source 排除 vector：1024 维向量在 _source 里约 8KB/条，检索时不需要取回
    2. 融合权重放在查询侧（knn.boost / bool.boost），沿用 RAGFlow 的 0.95 / 0.05

写入策略：**按 doc_id 先删后写**。
chunk_id 是内容哈希，阈值一变 id 就变，旧的记录会变成孤儿；只有按 doc_id 整体
删除再写，才能保证库里没有残留。
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Iterable

from paper_rag.models import Chunk

# 索引字段类型：对齐 RAGFlow 的思路，但用显式 mapping（不做"后缀推导类型"）
MAPPING_PROPERTIES: dict[str, Any] = {
    "chunk_id": {"type": "keyword"},
    "doc_id": {"type": "keyword"},
    "doc_name": {"type": "keyword"},
    "kind": {"type": "keyword"},
    "parent_id": {"type": "keyword"},
    "sec_no": {"type": "keyword"},
    "sec_path": {
        "type": "text",
        "analyzer": "whitespace",
        "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
    },
    "title": {"type": "keyword"},
    "create_time": {"type": "date", "format": "yyyy-MM-dd HH:mm:ss"},
    # 正文：只存不索引（RAGFlow 里 content_with_weight 也是 index:false）
    "content": {"type": "text", "index": False},
    # 分词字段：我们自己切好的词，用 whitespace 按空格收下
    "content_ltks": {"type": "text", "analyzer": "whitespace"},
    "content_sm_ltks": {"type": "text", "analyzer": "whitespace"},
    "title_tks": {"type": "text", "analyzer": "whitespace"},
    "title_sm_tks": {"type": "text", "analyzer": "whitespace"},
    "available_int": {"type": "integer"},
}

# 检索时希望返回的字段（不含 vector）
SOURCE_FIELDS = [
    "chunk_id", "doc_id", "doc_name", "kind", "parent_id",
    "sec_no", "sec_path", "title", "create_time", "content",
    "content_ltks", "available_int",
]


def build_mapping(dims: int) -> dict[str, Any]:
    props = dict(MAPPING_PROPERTIES)
    props["vector"] = {
        "type": "dense_vector",
        "dims": dims,
        "similarity": "cosine",
        "index": True,
    }
    return {
        "settings": {
            "index": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "refresh_interval": "1s",
            }
        },
        "mappings": {
            "_source": {"excludes": ["vector"]},
            "properties": props,
        },
    }


class ESStore:
    """极简 ES 客户端（只用标准库），够用于建索引、写入、检索。"""

    def __init__(self, url: str, index: str, timeout: int = 60):
        self.url = url.rstrip("/")
        self.index = index
        self.timeout = timeout

    # -- 底层 --------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        body: Any = None,
        ndjson: str | None = None,
    ) -> dict:
        data = None
        headers = {}
        if ndjson is not None:
            data = ndjson.encode("utf-8")
            headers["Content-Type"] = "application/x-ndjson"
        elif body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(
            self.url + path, data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw.strip() else {}
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            raise RuntimeError(f"ES {method} {path} 失败 HTTP {e.code}：{detail[:400]}") from e

    # -- 索引管理 ----------------------------------------------------------

    def ping(self) -> dict:
        return self._request("GET", "/")

    def index_exists(self) -> bool:
        try:
            self._request("HEAD", f"/{self.index}")
            return True
        except RuntimeError as e:
            if "404" in str(e):
                return False
            raise

    def create_index(self, dims: int, force: bool = False) -> str:
        """建索引。force=True 时先删后建（会丢数据）。"""
        if self.index_exists():
            if not force:
                return f"索引 {self.index} 已存在（未改动；要重建请加 --force）"
            self.delete_index()
        self._request("PUT", f"/{self.index}", build_mapping(dims))
        return f"已创建索引 {self.index}（dims={dims}）"

    def delete_index(self) -> None:
        self._request("DELETE", f"/{self.index}")

    def refresh(self) -> None:
        self._request("POST", f"/{self.index}/_refresh")

    def count(self) -> int:
        r = self._request("GET", f"/{self.index}/_count")
        return int(r.get("count", 0))

    # -- 写入 --------------------------------------------------------------

    def delete_by_doc(self, doc_id: str) -> int:
        r = self._request(
            "POST",
            f"/{self.index}/_delete_by_query?refresh=false&conflicts=proceed",
            {"query": {"term": {"doc_id": doc_id}}},
        )
        return int(r.get("deleted", 0))

    def list_docs(self) -> list[dict]:
        """列出库里有哪些论文（按 doc_id 聚合），用于删除前先看清家底。"""
        body = {
            "size": 0,
            "aggs": {
                "docs": {
                    "terms": {"field": "doc_id", "size": 1000},
                    "aggs": {
                        "name": {"top_hits": {"size": 1, "_source": ["doc_name", "doc_id"]}},
                        "searchable": {"filter": {"term": {"available_int": 1}}},
                    },
                }
            },
        }
        r = self._request("POST", f"/{self.index}/_search", body)
        out = []
        for b in r.get("aggregations", {}).get("docs", {}).get("buckets", []):
            name = ""
            hits = b.get("name", {}).get("hits", {}).get("hits", [])
            if hits:
                name = hits[0].get("_source", {}).get("doc_name", "")
            out.append(
                {
                    "doc_id": b["key"],
                    "doc_name": name,
                    "chunks": int(b["doc_count"]),
                    "searchable": int(b.get("searchable", {}).get("doc_count", 0)),
                }
            )
        return sorted(out, key=lambda d: d["doc_name"])

    def bulk_index(self, chunks: Iterable[Chunk], refresh: bool = False) -> dict:
        """批量写入；_id = chunk_id（幂等覆盖）。父块没有向量，字段直接省略。"""
        lines: list[str] = []
        n = 0
        for c in chunks:
            lines.append(json.dumps({"index": {"_index": self.index, "_id": c.chunk_id}}))
            doc = c.to_dict()

            doc.pop("chars", None)         # 派生字段，不存
            if not doc.get("vector"):
                doc.pop("vector", None)    # 父块：空数组会让 dense_vector 报维度错误
            lines.append(json.dumps(doc, ensure_ascii=False))
            n += 1

        ndjson = "\n".join(lines) + "\n"
        r = self._request(
            "POST", f"/_bulk?refresh={'true' if refresh else 'false'}", ndjson=ndjson
        )
        errors = [i for i in r.get("items", []) if "error" in i.get("index", {})]
        return {"indexed": n - len(errors), "errors": len(errors), "error_sample": errors[:3]}

    def get_by_ids(self, ids: list[str]) -> dict[str, dict]:
        """按 _id 批量取（用于回取父块）。

        注意：mget 的 `ids` 简写形式不支持按文档指定 _source，必须写成
        {"docs": [{"_id": ..., "_source": [...]}, ...]}。
        """
        if not ids:
            return {}
        body = {"docs": [{"_id": i, "_source": SOURCE_FIELDS} for i in ids]}
        r = self._request("POST", f"/{self.index}/_mget", body)
        out = {}
        for doc in r.get("docs", []):
            if doc.get("found"):
                out[doc["_id"]] = doc.get("_source", {})
        return out

    # -- 检索 --------------------------------------------------------------

    def search(
        self,
        query_vector: list[float] | None = None,
        query_text: str = "",
        top: int = 5,
        k: int = 50,
        num_candidates: int = 100,
        doc_id: str = "",
        text_boost: float = 0.05,
        vector_boost: float = 0.95,
    ) -> list[dict]:
        """向量 + 关键词混合检索（融合方式对齐 RAGFlow：boost 0.95 / 0.05）。

        注意 query_text 必须是**已经切好词、小写、空格分隔**的字符串。
        因为字段用的是 whitespace 分析器——它只按空格切，不做小写归一。
        直接把原始问句传进来会一条都匹配不上（"WDCANet" ≠ "wdcanet"）。
        """
        filters = [{"term": {"available_int": 1}}]
        if doc_id:
            filters.append({"term": {"doc_id": doc_id}})

        body: dict[str, Any] = {
            "size": top,
            "_source": SOURCE_FIELDS,
        }
        bool_query: dict[str, Any] = {"filter": filters}
        if query_text.strip():
            bool_query["must"] = [
                {
                    "multi_match": {
                        "query": query_text,
                        "fields": ["content_ltks", "title_tks^2"],
                        "type": "best_fields",
                    }
                }
            ]
            bool_query["boost"] = text_boost
        body["query"] = {"bool": bool_query}

        if query_vector:
            body["knn"] = {
                "field": "vector",
                "query_vector": query_vector,
                "k": k,
                "num_candidates": num_candidates,
                "filter": {"bool": {"filter": filters}},
                "boost": vector_boost,
            }

        r = self._request("POST", f"/{self.index}/_search", body)
        hits = []
        for h in r.get("hits", {}).get("hits", []):
            src = dict(h.get("_source", {}))
            src["_score"] = h.get("_score")
            hits.append(src)
        return hits
