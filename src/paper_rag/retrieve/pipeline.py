# -*- coding: utf-8 -*-
"""检索流水线：ES 召回 → 子块重排 → 阈值 → 取前 N → 聚合父块 → 预算裁剪。

顺序为什么是这个
----------------
1. **先重排**：rerank 模型的窗口通常只有 512 token，子块（约 300 token）塞得下，
   父块（可能 3400 token）塞不下。所以只能在子块上重排。
2. **再过滤**：重排分数是唯一可比的分数（ES 的 _score 和 rerank 分数不同量纲）。
3. **再取前 N**：对齐 RAGFlow（top_n=6）。
4. **最后聚合**：命中同一父块的多个子块合成一条，分数取子块的 max。
   代价是聚合后可能不足 N 条——如果出现这种情况，说明同一个父块占了多个名额，
   即"这一节强相关"，结果少但质量不差（对应 RAGFlow 的行为）。
5. **预算裁剪**：父块可能很大（13653 字符），按字符预算装，装不下就退回"命中的子块"。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from paper_rag.models import Chunk  # noqa: F401  (仅为类型提示可读性)


@dataclass
class Passage:
    """最终喂给 LLM 的一条材料。"""

    chunk_id: str
    doc_id: str
    doc_name: str
    sec_no: str
    sec_path: str
    title: str
    content: str
    score: float
    kind: str                     # parent（聚合而来） / solo
    parent_id: str = ""
    matched_child_id: str = ""    # 命中的子块 id，引用要指到它
    fallback_content: str = ""    # 父块太大时用子块正文顶替

    @property
    def chars(self) -> int:
        return len(self.content)


@dataclass
class RetrieveResult:
    passages: list[Passage] = field(default_factory=list)
    candidates: int = 0
    reranked: bool = False
    dropped_by_threshold: int = 0
    dropped_by_budget: int = 0
    logs: list[str] = field(default_factory=list)

    @property
    def total_chars(self) -> int:
        return sum(p.chars for p in self.passages)


def _score_of(hit: dict, reranked: bool) -> float:
    """取这条候选的排序分：重排过就用重排分，否则用 ES 分。"""
    if reranked and "rerank_score" in hit:
        return float(hit["rerank_score"])
    return float(hit.get("_score") or 0.0)


def rerank_hits(hits: list[dict], reranker, query: str) -> list[dict]:
    """对子块打重排分，并按分数降序返回。"""
    if not hits:
        return []
    scores = reranker.score(query, [h.get("content") or "" for h in hits])
    for h, s in zip(hits, scores):
        h["rerank_score"] = s
    return sorted(hits, key=lambda h: h["rerank_score"], reverse=True)


def aggregate_by_parent(hits: list[dict], store, reranked: bool) -> list[Passage]:
    """把命中同一父块的子块合成一条；分数取 max，并记住最相关的那个子块。"""
    groups: dict[str, list[dict]] = {}
    passages: list[Passage] = []

    for h in hits:
        pid = h.get("parent_id") or ""
        if pid:
            groups.setdefault(pid, []).append(h)
        else:
            passages.append(
                Passage(
                    chunk_id=h.get("chunk_id", ""),
                    doc_id=h.get("doc_id", ""),
                    doc_name=h.get("doc_name", ""),
                    sec_no=h.get("sec_no", ""),
                    sec_path=h.get("sec_path", ""),
                    title=h.get("title", ""),
                    content=h.get("content") or "",
                    score=_score_of(h, reranked),
                    kind=h.get("kind", "solo"),
                    matched_child_id=h.get("chunk_id", ""),
                )
            )

    if not groups:
        return passages

    parents = store.get_by_ids(list(groups.keys()))
    for pid, children in groups.items():
        best = max(children, key=lambda h: _score_of(h, reranked))
        parent = parents.get(pid)
        if parent:  # 父块还在：用父块正文
            passages.append(
                Passage(
                    chunk_id=pid,
                    doc_id=parent.get("doc_id", ""),
                    doc_name=parent.get("doc_name", ""),
                    sec_no=parent.get("sec_no", ""),
                    sec_path=parent.get("sec_path", ""),
                    title=parent.get("title", ""),
                    content=parent.get("content") or "",
                    score=_score_of(best, reranked),
                    kind="parent",
                    parent_id=pid,
                    matched_child_id=best.get("chunk_id", ""),
                    fallback_content=best.get("content") or "",
                )
            )
        else:  # 父块缺失（被删或索引不全）：退回最佳子块
            passages.append(
                Passage(
                    chunk_id=best.get("chunk_id", ""),
                    doc_id=best.get("doc_id", ""),
                    doc_name=best.get("doc_name", ""),
                    sec_no=best.get("sec_no", ""),
                    sec_path=best.get("sec_path", ""),
                    title=best.get("title", ""),
                    content=best.get("content") or "",
                    score=_score_of(best, reranked),
                    kind=best.get("kind", "child"),
                    matched_child_id=best.get("chunk_id", ""),
                )
            )

    return sorted(passages, key=lambda p: p.score, reverse=True)


def apply_budget(passages: list[Passage], char_budget: int) -> tuple[list[Passage], int]:
    """按字符预算装填；父块装不下时退回子块正文。返回 (保留, 丢弃数)。"""
    kept: list[Passage] = []
    used = 0
    dropped = 0

    for p in passages:
        if used + p.chars <= char_budget:
            kept.append(p)
            used += p.chars
            continue
        # 父块太大，试试用命中的子块顶替（短得多）
        if p.fallback_content and used + len(p.fallback_content) <= char_budget:
            p.content = p.fallback_content
            p.kind = "child"
            kept.append(p)
            used += p.chars
            continue
        dropped += 1

    return kept, dropped


def retrieve(
    *,
    store,
    query: str,
    query_vector: list[float],
    query_tokens: str,
    reranker=None,
    candidates: int = 64,
    top_n: int = 6,
    threshold: float = 0.0,
    char_budget: int = 24000,
) -> RetrieveResult:
    result = RetrieveResult()

    hits = store.search(
        query_vector=query_vector,
        query_text=query_tokens,
        top=candidates,
        k=candidates,
    )
    result.candidates = len(hits)
    result.logs.append(f"ES 召回 {len(hits)} 条")
    if not hits:
        return result

    if reranker is not None:
        rerank_chars = sum(len(h.get("content") or "") for h in hits)
        hits = rerank_hits(hits, reranker, query)
        result.reranked = True
        result.logs.append(
            f"重排 {len(hits)} 条（送去 {rerank_chars} 字符），最高分 {hits[0]['rerank_score']:.4f}"
        )
    else:
        hits.sort(key=lambda h: h.get("_score") or 0.0, reverse=True)

    if threshold > 0:
        before = len(hits)
        hits = [h for h in hits if _score_of(h, result.reranked) >= threshold]
        result.dropped_by_threshold = before - len(hits)
        result.logs.append(f"阈值 {threshold} 过滤掉 {result.dropped_by_threshold} 条")

    hits = hits[:top_n]
    result.logs.append(f"截断到前 {len(hits)} 条")

    passages = aggregate_by_parent(hits, store, result.reranked)
    result.logs.append(f"聚合父块后 {len(passages)} 条")

    kept, dropped = apply_budget(passages, char_budget)
    result.dropped_by_budget = dropped
    if dropped:
        result.logs.append(f"预算 {char_budget} 字符，丢弃 {dropped} 条")

    result.passages = kept
    return result
