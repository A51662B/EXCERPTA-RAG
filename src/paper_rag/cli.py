# -*- coding: utf-8 -*-
"""命令行入口。

    python run.py blocks    <full.md> [--verify] [--dump data/blocks.json]
    python run.py levels    <full.md> [--no-llm] [--dump data/blocks_leveled.json]
    python run.py structure [<full.md>] [--from data/blocks_leveled.json] [--dump data/blocks_structured.json]
    python run.py chunks    [<full.md>] [--from data/blocks_structured.json] [--dump data/chunks.json]

`--from` 可以接任意一阶段的 blocks JSON，会自动判断还缺哪一步：
    blocks.json            → 缺 level   → 报错，让你先跑 levels
    blocks_leveled.json    → 缺 sec_no  → 自动编号 + 合并空块
    blocks_structured.json → 什么都不缺 → 直接用
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

from paper_rag.cache import Cache
from paper_rag.config import Settings, load_settings
from paper_rag.ingest.block_merge import merge_heading_only
from paper_rag.ingest.chunk_split import doc_id_from_title, oversize, split_blocks
from paper_rag.ingest.heading_levels import apply_levels, classify_levels
from paper_rag.ingest.markdown_blocks import (
    BlockExtraction,
    dump_blocks,
    extract_blocks,
    load_blocks,
    write_blocks_json,
)
from paper_rag.index.embedder import OpenAICompatEmbedder, embed_chunks
from paper_rag.index.es_store import ESStore
from paper_rag.index.reranker import build_reranker
from paper_rag.retrieve.pipeline import retrieve
from paper_rag.ingest.runner import ingest_paper
from paper_rag.ingest.section_index import number_sections
from paper_rag.llm import build_client
from paper_rag.io_json import load_chunks, write_chunks
from paper_rag.tokenizer import add_token_fields, tokenize_text
from paper_rag.models import Block, Chunk
from paper_rag.quality import blocks_nonws_chars, source_nonws_chars


# --------------------------------------------------------------------------
# 公共：拿到「已完成阶段 3/4」的 blocks
# --------------------------------------------------------------------------


def _prepare_blocks_from_json(path: str) -> list[Block]:
    blocks = load_blocks(path)
    missing = [b for b in blocks if b.level is None]
    if missing:
        raise SystemExit(
            f"{path} 里的块还没有 level（{len(missing)} 个为空）。先跑 levels，再用它的产物。"
        )

    need_number = any(b.title.strip() and not b.sec_no for b in blocks)
    need_merge = any(b.chars == 0 for b in blocks)
    if need_number:
        number_sections(blocks)
        print("  自动补做：章节编号")
    if need_merge:
        before = len(blocks)
        blocks = merge_heading_only(blocks)
        print(f"  自动补做：空块合并（{before} → {len(blocks)} 块）")
    return blocks


def _prepare_blocks_from_md(path: str, args, settings: Settings) -> list[Block]:
    ex = extract_blocks(path)
    headings = [b.title for b in ex.blocks]
    if not headings:
        raise SystemExit("没有切出任何块")
    client = None
    if not args.no_llm:
        client = build_client(settings.llm_base_url, settings.llm_api_key, settings.llm_model)
    result = classify_levels(headings, paper_title=args.title or headings[0], client=client)
    apply_levels(ex.blocks, result.levels)
    print(f"  层级来源：{result.source}")
    number_sections(ex.blocks)
    return merge_heading_only(ex.blocks)


def _get_blocks(args, settings: Settings) -> list[Block]:
    if args.from_json:
        print(f"载入：{args.from_json}")
        return _prepare_blocks_from_json(args.from_json)
    if not args.source:
        raise SystemExit("需要提供 full.md 路径，或用 --from 指定上一阶段的 JSON")
    return _prepare_blocks_from_md(args.source, args, settings)


# --------------------------------------------------------------------------
# blocks
# --------------------------------------------------------------------------


def _print_blocks_report(ex: BlockExtraction, show: int, verify: bool) -> None:
    st = ex.stats
    print("=" * 78)
    print(f"源文件              : {ex.source}")
    print(f"文件总行数          : {st.total_lines}")
    print(f"整行只有图片、已丢弃: {st.image_lines_dropped} 行")
    print(f"行内图片标记、已清除: {st.inline_images_removed} 处")
    cut = f"{st.refs_cut_line} 行" if st.refs_cut_line else "未遇到"
    print(f"参考文献截断于      : {cut}")
    print(f"切出的块数          : {len(ex.blocks)}")
    print(f"正文总字符          : {ex.total_chars}")

    if verify:
        src = source_nonws_chars(ex.source, st.refs_cut_line)
        got = blocks_nonws_chars(ex.blocks)
        print(f"内容守恒核对        : 源 {src} / 产物 {got} → {'OK' if src == got else '!! 不一致'}")

    empty = ex.empty_blocks()
    print(f"空正文的块          : {len(empty)}")
    for b in empty:
        print(f"    idx={b.idx:<3} line={b.line:<4} md_level={b.md_level} title={b.title!r}")
    print("=" * 78)

    print(f"\n前 {show} 块：")
    for b in ex.blocks[:show]:
        first = b.body.splitlines()[0][:60] if b.body else ""
        print(
            f"  idx={b.idx:<3} line={b.line:<4} md_level={b.md_level} "
            f"chars={b.chars:<6} title={b.title[:50]!r}"
        )
        if first:
            print(f"        正文首行: {first!r}")


def cmd_blocks(args: argparse.Namespace) -> int:
    ex = extract_blocks(args.md)
    _print_blocks_report(ex, args.show, args.verify)

    assert [b.idx for b in ex.blocks] == list(range(len(ex.blocks))), "idx 不连续"

    if args.verify:
        src = source_nonws_chars(ex.source, ex.stats.refs_cut_line)
        if src != blocks_nonws_chars(ex.blocks):
            print("\n内容守恒核对失败：源与产物字符数不一致", file=sys.stderr)
            return 2

    if args.dump:
        print(f"\n已写出：{dump_blocks(ex, args.dump)}（{len(ex.blocks)} 块）")
    return 0


# --------------------------------------------------------------------------
# levels
# --------------------------------------------------------------------------


def _print_level_tree(ex: BlockExtraction, result, show: int) -> None:
    print("=" * 78)
    print(f"块数            : {len(ex.blocks)}")
    print(f"层级来源        : {result.source}" + ("（LLM）" if result.source == "llm" else "（规则兜底）"))
    print(f"尝试次数        : {result.attempts}")
    if result.errors:
        print(f"未消除的错误    : {'；'.join(result.errors)}")
    dist = Counter(b.level for b in ex.blocks)
    print("层级分布        : " + ", ".join(f"{k}级×{v}" for k, v in sorted(dist.items())))
    print("=" * 78)

    print(f"\n章节树（前 {show} 块）：")
    for b in ex.blocks[:show]:
        indent = "  " * ((b.level or 1) - 1)
        mark = "（空正文）" if b.chars == 0 else f"{b.chars} 字符"
        print(f"{indent}[{b.level}] {b.title}   {mark}")


def cmd_levels(args: argparse.Namespace) -> int:
    settings = load_settings()
    ex = extract_blocks(args.md)
    headings = [b.title for b in ex.blocks]
    if not headings:
        print("没有切出任何块", file=sys.stderr)
        return 2

    client = None
    if not args.no_llm:
        client = build_client(settings.llm_base_url, settings.llm_api_key, settings.llm_model)
        print(f"LLM: {settings.llm_model} @ {settings.llm_base_url}")

    result = classify_levels(headings, paper_title=args.title or headings[0], client=client)
    apply_levels(ex.blocks, result.levels)
    _print_level_tree(ex, result, args.show)

    if args.dump:
        print(f"\n已写出：{dump_blocks(ex, args.dump)}（{len(ex.blocks)} 块，含 level）")
    return 0 if result.source == "llm" else 1


# --------------------------------------------------------------------------
# structure：章节编号 + 空块合并
# --------------------------------------------------------------------------


def _print_structure(blocks: list[Block], before: int, show: int) -> None:
    merged = sum(len(b.merged_from) for b in blocks)
    print("=" * 78)
    print(f"合并前块数      : {before}")
    print(f"合并后块数      : {len(blocks)}")
    print(f"被并入的空块    : {merged} 个")
    if merged:
        pairs = [f"{b.sec_no}←{b.merged_from}" for b in blocks if b.merged_from]
        print("合并明细        : " + "，".join(pairs))
    print("=" * 78)

    print(f"\n章节结构（前 {show} 块）：")
    print(f"  {'编号':<8}{'层级':<6}{'字符':>7}  标题")
    for b in blocks[:show]:
        indent = "  " * ((b.level or 1) - 1)
        print(f"  {b.sec_no or '-':<8}{b.level:<6}{b.chars:>7}  {indent}{b.title[:56]}")

    print("\n章节路径抽样（前 6 块）：")
    for b in blocks[:6]:
        print(f"  [{b.sec_no or '-'}] {b.sec_path or '(无)'}")


def cmd_structure(args: argparse.Namespace) -> int:
    settings = load_settings()
    blocks = _get_blocks(args, settings)
    _print_structure(blocks, len(blocks), args.show)

    if args.dump:
        print(f"\n已写出：{write_blocks_json(blocks, args.dump)}（{len(blocks)} 块）")
    return 0


# --------------------------------------------------------------------------
# chunks：切父子块
# --------------------------------------------------------------------------


def _print_chunks(chunks: list[Chunk], blocks: list[Block], show: int, settings: Settings) -> None:
    kinds = Counter(c.kind for c in chunks)
    print("=" * 78)
    print(f"编码上限        : {settings.embed_max_chars} 字符（超过就拆，保证每条 ≤ 上限）")
    print(f"子块目标        : {settings.child_target_chars} 字符")
    print(f"输入块数        : {len(blocks)}")
    print(f"输出 chunk 数   : {len(chunks)}"
          f"（solo {kinds.get('solo', 0)}，parent {kinds.get('parent', 0)}，child {kinds.get('child', 0)}）")
    body_chars = sum(b.chars for b in blocks)
    chunk_chars = sum(c.chars for c in chunks if c.kind != "parent")
    print(f"正文字符        : 输入 {body_chars} → 可检索 {chunk_chars}"
          f"（多出 {chunk_chars - body_chars}，来自子块标题前缀）")

    parents = [c for c in chunks if c.kind == "parent"]
    if parents:
        print("\n被拆开的节：")
        for p in parents:
            kids = [c for c in chunks if c.parent_id == p.chunk_id]
            avg = sum(c.chars for c in kids) // max(1, len(kids))
            print(f"  [{p.sec_no}] {p.title[:46]:<46} {p.chars:>6} 字符 → {len(kids):>2} 个子块（平均 {avg} 字符）")

    print(f"\n\n前 {show} 个 chunk：")
    for c in chunks[:show]:
        tag = {"solo": "  ", "parent": "P ", "child": "  └"}[c.kind]
        print(f"  {tag} [{c.sec_no:>4}] {c.chars:>6} 字符  {c.title[:40]}")
        print(f"        chunk_id={c.chunk_id}  available={c.available_int}"
              + (f"  parent={c.parent_id}" if c.parent_id else ""))
    print("=" * 78)


def cmd_chunks(args: argparse.Namespace) -> int:
    settings = load_settings()
    if args.embed_max_chars or args.child_chars:
        settings = replace(
            settings,
            embed_max_chars=args.embed_max_chars or settings.embed_max_chars,
            child_target_chars=args.child_chars or settings.child_target_chars,
        )

    blocks = _get_blocks(args, settings)
    if not blocks:
        print("没有可切分的块", file=sys.stderr)
        return 2

    # 文档身份：名字给人看，id 给机器用（由标题算，见 chunk_split 里的说明）
    doc_name = args.doc_name or blocks[0].title.strip()
    doc_id = args.doc_id or doc_id_from_title(doc_name, fallback_path=args.from_json or args.source)

    chunks = split_blocks(
        blocks,
        settings,
        doc_id=doc_id,
        with_title_prefix=not args.no_prefix,
        doc_name=doc_name,
    )

    print(f"doc_name        : {doc_name[:70]}")
    print(f"doc_id          : {doc_id}")
    _print_chunks(chunks, blocks, args.show, settings)

    bad = oversize(chunks, settings.embed_max_chars)
    if bad:
        print(
            f"\n警告：{len(bad)} 条可检索块超过编码上限 {settings.embed_max_chars} 字符"
            f"（表格或超长句子，按规则不再硬切）："
        )
        for c in bad[:6]:
            print(f"  [{c.sec_no or '-':>4}] {c.chars:>6} 字符  {c.title[:36]}")

    if args.dump:
        print(f"\n已写出：{write_chunks(chunks, args.dump)}（{len(chunks)} 条）")
    return 0


# --------------------------------------------------------------------------
# embed：向量化
# --------------------------------------------------------------------------


def cmd_embed(args: argparse.Namespace) -> int:
    settings = load_settings()
    chunks = load_chunks(args.from_json)
    if not chunks:
        print("没有读到 chunk", file=sys.stderr)
        return 2

    client = OpenAICompatEmbedder(
        base_url=settings.embed_base_url,
        api_key=settings.embed_api_key,
        model=settings.embed_model,
        batch=args.batch or settings.embed_batch,
    )

    print(f"嵌入模型        : {settings.embed_model}")
    print(f"接口            : {client.url}")
    print(f"批次大小        : {client.batch}")
    print(f"输入 chunk 数   : {len(chunks)}（其中父块 {sum(1 for c in chunks if c.kind == 'parent')} 条不编码）")

    stats = embed_chunks(chunks, client, force=args.force, limit=args.limit)
    print("\n" + stats.summary())

    if args.dump:
        out = write_chunks(chunks, args.dump, include_vectors=not args.no_vectors)
        size_mb = out.stat().st_size / 1024 / 1024
        print(f"已写出：{out}（{len(chunks)} 条，{size_mb:.1f} MB）")
    return 0


# --------------------------------------------------------------------------
# tokenize：补分词字段（不花钱，可反复跑）
# --------------------------------------------------------------------------


def cmd_tokenize(args: argparse.Namespace) -> int:
    chunks = load_chunks(args.from_json)
    if not chunks:
        print("没有读到 chunk", file=sys.stderr)
        return 2

    add_token_fields(chunks)
    print(f"已给 {len(chunks)} 条 chunk 补上分词字段和 create_time")

    for c in chunks[: args.show]:
        print(f"\n  [{c.sec_no or '-':>4}] {c.kind:<6} {c.chars:>5} 字符  {c.title[:34]}")
        print(f"    title_tks      : {c.title_tks[:70]}")
        print(f"    content_ltks   : {c.content_ltks[:90]}")
        print(f"    content_sm_ltks: {c.content_sm_ltks[:90]}")

    if args.dump:
        out = write_chunks(chunks, args.dump, include_vectors=True)
        print(f"\n已写出：{out}（{len(chunks)} 条）")
    return 0


# --------------------------------------------------------------------------
# index / search：写 ES、查 ES
# --------------------------------------------------------------------------


def cmd_index(args: argparse.Namespace) -> int:
    settings = load_settings()
    chunks = load_chunks(args.from_json)
    if not chunks:
        print("没有读到 chunk", file=sys.stderr)
        return 2

    dims = next((len(c.vector) for c in chunks if c.vector), 0)
    if not dims:
        print("这批 chunk 没有向量，先跑 embed", file=sys.stderr)
        return 2

    store = ESStore(settings.es_url, settings.es_index)
    info = store.ping()
    print(f"ES              : {settings.es_url}（{info.get('version', {}).get('number')}）")
    print(f"索引            : {settings.es_index}")
    print(f"向量维度        : {dims}")
    print(store.create_index(dims, force=args.force))

    for d in sorted({c.doc_id for c in chunks}):
        removed = store.delete_by_doc(d)
        print(f"  清理 doc_id={d} 的旧记录：删除 {removed} 条")

    stats = store.bulk_index(chunks, refresh=True)
    store.refresh()
    print(f"\n写入完成：成功 {stats['indexed']} 条，失败 {stats['errors']} 条")
    if stats["errors"]:
        print(f"  错误样例：{stats['error_sample']}")
    print(f"索引内文档总数  : {store.count()}")
    return 0 if not stats["errors"] else 1


def cmd_search(args: argparse.Namespace) -> int:
    settings = load_settings()
    client = OpenAICompatEmbedder(
        base_url=settings.embed_base_url,
        api_key=settings.embed_api_key,
        model=settings.embed_model,
    )
    query_vector = client.embed([args.q])[0]
    query_tokens = tokenize_text(args.q)   # 字段用 whitespace 分析器，查询必须先切好词并小写

    store = ESStore(settings.es_url, settings.es_index)
    hits = store.search(
        query_vector=query_vector,
        query_text=query_tokens,
        top=args.top,
        k=args.k,
        doc_id=args.doc_id,
        text_boost=args.text_boost,
        vector_boost=args.vector_boost,
    )

    print(f"查询            : {args.q}")
    print(f"查询分词        : {query_tokens}")
    print(f"融合权重        : 文本 {args.text_boost} / 向量 {args.vector_boost}（knn k={args.k}）")
    if not hits:
        print("\n没有命中。")
        return 0

    print(f"\n{'#':<3}{'分数':>8}  {'sec':<6}{'角色':<8}{'字符':>6}  {'标题':<34}片段")
    for i, h in enumerate(hits, 1):
        preview = (h.get("content") or "").replace("\n", " ")[:40]
        title = (h.get("title") or "")[:32]
        print(
            f"{i:<3}{h['_score']:>8.4f}  {h.get('sec_no', ''):<6}{h.get('kind', ''):<8}"
            f"{len(h.get('content') or ''):>6}  {title:<34}{preview}"
        )
        if h.get("parent_id"):
            print(f"   └ parent_id={h['parent_id']}")
    return 0


# --------------------------------------------------------------------------
# retrieve：检索 → 重排 → 过滤 → 前 N → 聚合父块 → 预算
# --------------------------------------------------------------------------


def cmd_retrieve(args: argparse.Namespace) -> int:
    settings = load_settings()
    client = OpenAICompatEmbedder(
        base_url=settings.embed_base_url,
        api_key=settings.embed_api_key,
        model=settings.embed_model,
    )
    query_vector = client.embed([args.q])[0]
    query_tokens = tokenize_text(args.q)

    store = ESStore(settings.es_url, settings.es_index)
    reranker = None
    if not args.no_rerank:
        reranker = build_reranker(
            settings.rerank_url, settings.rerank_api_key, settings.rerank_model
        )

    result = retrieve(
        store=store,
        query=args.q,
        query_vector=query_vector,
        query_tokens=query_tokens,
        reranker=reranker,
        candidates=args.candidates or settings.retrieve_candidates,
        top_n=args.top or settings.retrieve_top_n,
        threshold=settings.retrieve_threshold if args.threshold is None else args.threshold,
        char_budget=args.budget or settings.prompt_char_budget,
    )

    print(f"查询            : {args.q}")
    print(f"查询分词        : {query_tokens}")
    print(f"重排模型        : {settings.rerank_model if reranker else '（未启用）'}")
    print("流程            : " + " → ".join(result.logs))

    if not result.passages:
        print("\n没有结果。")
        return 0

    print(f"\n最终 {len(result.passages)} 条，共 {result.total_chars} 字符：")
    for i, p in enumerate(result.passages, 1):
        tag = "父块" if p.kind == "parent" else "独立"
        preview = p.content.replace("\n", " ")[:60]
        print(f"\n  {i}. [{tag}] 分数 {p.score:.4f}  sec {p.sec_no}  {p.chars} 字符")
        print(f"     路径: {p.sec_path}")
        print(f"     开头: {preview}")
        if p.matched_child_id:
            print(f"     命中子块: {p.matched_child_id}")
    return 0


# --------------------------------------------------------------------------
# delete：删掉一篇论文的全部 chunk
# --------------------------------------------------------------------------


def cmd_delete(args: argparse.Namespace) -> int:
    settings = load_settings()
    store = ESStore(settings.es_url, settings.es_index)
    docs = store.list_docs()

    if args.list:
        if not docs:
            print("库里还没有论文。")
            return 0
        print(f"{'doc_id':<14}{'条数':>5}{'可检索':>7}  论文")
        for d in docs:
            print(f"{d['doc_id']:<14}{d['chunks']:>5}{d['searchable']:>7}  {d['doc_name'][:60]}")
        print(f"\n共 {len(docs)} 篇，{sum(d['chunks'] for d in docs)} 条记录。")
        return 0

    if args.all:
        targets = docs
    elif args.doc_id:
        targets = [d for d in docs if d["doc_id"] == args.doc_id]
        if not targets:
            print(f"库里没有 doc_id={args.doc_id}（用 --list 看看有哪些）", file=sys.stderr)
            return 2
    elif args.doc_name:
        needle = args.doc_name.strip().lower()
        targets = [d for d in docs if needle in d["doc_name"].strip().lower()]
        if not targets:
            print(f"没有标题匹配 {args.doc_name!r} 的论文（用 --list 看看有哪些）", file=sys.stderr)
            return 2
    else:
        print("要指定删哪个：--doc-id / --doc-name / --all（先 --list 看家底）", file=sys.stderr)
        return 2

    print(f"将删除 {len(targets)} 篇：")
    for d in targets:
        print(f"  {d['doc_id']}  {d['chunks']:>4} 条  {d['doc_name'][:56]}")
    total = sum(d["chunks"] for d in targets)

    if args.dry_run:
        print(f"\n（dry-run）共会删除 {total} 条记录，未执行。")
        return 0

    if not args.yes:
        print(f"\n确认后会删除 {total} 条记录。加 --yes 执行，或先 --dry-run 看看。")
        return 0

    deleted = 0
    for d in targets:
        deleted += store.delete_by_doc(d["doc_id"])
    store.refresh()
    print(f"\n已删除 {deleted} 条记录；剩余 {store.count()} 条。")
    return 0


# --------------------------------------------------------------------------
# ingest：一条命令跑完全流程（带缓存）
# --------------------------------------------------------------------------


def cmd_ingest(args: argparse.Namespace) -> int:
    settings = load_settings()
    try:
        report = ingest_paper(
            settings,
            args.md,
            title=args.title,
            doc_id=args.doc_id,
            force=args.force,
            no_index=args.no_index,
            no_prefix=args.no_prefix,
        )
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 2
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2

    kinds = report.kinds
    print("=" * 78)
    print(f"源文件          : {report.md_path}")
    print(f"doc_name        : {report.doc_name[:70]}")
    print(f"doc_id          : {report.doc_id}")
    print(f"源文件哈希      : {report.source_hash[:12]}")
    print(f"缓存位置        : {report.cache_dir}")
    print(f"缓存状态        : levels={report.cache_levels}  chunks={report.cache_chunks}")
    if report.cache_chunks and not args.force:
        print("命中缓存        : 直接载入 chunk，本次 0 次 LLM、0 次 embedding")
    print(f"层级来源        : {report.levels_from}")
    print(
        f"chunk           : 共 {report.chunks} 条"
        f"（solo {kinds.get('solo', 0)}，parent {kinds.get('parent', 0)}，child {kinds.get('child', 0)}）"
    )

    if args.no_index:
        print(f"\n（--no-index）未写入 ES。耗时 {report.seconds:.1f}s")
        return 0

    print(f"清理旧记录      : {report.deleted_before} 条")
    print(f"写入            : 成功 {report.indexed} 条，失败 {report.index_errors} 条")
    print(f"索引总数        : {report.index_total} 条")
    print(
        f"本次花费        : LLM {report.llm_calls} 次，embedding {report.embedded} 条，"
        f"耗时 {report.seconds:.1f}s"
    )
    print("=" * 78)
    return 0 if report.index_errors == 0 else 1


# --------------------------------------------------------------------------
# cache：看缓存、清缓存
# --------------------------------------------------------------------------


def cmd_cache(args: argparse.Namespace) -> int:
    settings = load_settings()
    cache = Cache(settings.cache_dir)

    if args.clear_all or args.clear:
        removed = cache.clear(args.clear)
        print(f"已清除 {removed} 个缓存目录（{settings.cache_dir}）")
        return 0

    entries = cache.entries()
    if not entries:
        print(f"缓存为空：{settings.cache_dir}")
        return 0
    print(f"{'doc_id':<14}{'hash':<14}{'块数':>5}  levels chunks  论文")
    for e in entries:
        print(
            f"{e['doc_id']:<14}{e['hash']:<14}{e['chunk_count']:>5}"
            f"  {'是' if e['levels'] else '否':<6} {'是' if e['chunks'] else '否':<6}  "
            f"{e['doc_name'][:46]}"
        )
    print(f"\n共 {len(entries)} 条缓存，目录 {settings.cache_dir}")
    return 0


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="paper_rag", description="本地论文知识库流水线")
    sub = ap.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("blocks", aliases=["parse"], help="把 full.md 切成 blocks")
    p1.add_argument("md", help="full.md 路径")
    p1.add_argument("--show", type=int, default=5, help="打印前 N 块")
    p1.add_argument("--dump", default="", help="导出 JSON 的路径")
    p1.add_argument("--verify", action="store_true", help="做内容守恒核对")
    p1.set_defaults(func=cmd_blocks)

    p2 = sub.add_parser("levels", help="判每个标题的语义层级")
    p2.add_argument("md", help="full.md 路径")
    p2.add_argument("--show", type=int, default=25, help="章节树打印前 N 块")
    p2.add_argument("--dump", default="", help="导出 JSON 的路径")
    p2.add_argument("--title", default="", help="论文标题")
    p2.add_argument("--no-llm", action="store_true", help="不调 LLM，只用规则兜底")
    p2.set_defaults(func=cmd_levels)

    for name, func, help_text in (
        ("structure", cmd_structure, "章节编号 + 空块合并"),
        ("chunks", cmd_chunks, "切父子块"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("source", nargs="?", default="", help="full.md 路径（用 --from 时可省略）")
        p.add_argument("--from", dest="from_json", default="", help="从上一阶段的 blocks JSON 载入")
        p.add_argument("--show", type=int, default=20, help="打印前 N 项")
        p.add_argument("--dump", default="", help="导出 JSON 的路径")
        p.add_argument("--title", default="", help="论文标题")
        p.add_argument("--no-llm", action="store_true", help="不调 LLM，只用规则兜底")
        p.set_defaults(func=func)

    # chunks 专属参数
    chunks_parser = sub.choices["chunks"]
    chunks_parser.add_argument("--embed-max-chars", type=int, default=0, help="编码模型能吃下的最大字符数（切分阈值）")
    chunks_parser.add_argument("--child-chars", type=int, default=0, help="子块目标字符数")
    chunks_parser.add_argument("--doc-id", default="", help="文档标识（默认由标题推导）")
    chunks_parser.add_argument("--doc-name", default="", help="文档名（默认取第一个标题）")
    chunks_parser.add_argument("--no-prefix", action="store_true", help="子块正文不加标题前缀")

    p4 = sub.add_parser("embed", help="向量化（父块跳过）")
    p4.add_argument("--from", dest="from_json", required=True, help="chunks JSON 路径")
    p4.add_argument("--dump", default="", help="导出带向量的 JSON 路径")
    p4.add_argument("--batch", type=int, default=0, help="每批条数（接口上限 25）")
    p4.add_argument("--limit", type=int, default=0, help="只编码前 N 条（调试用）")
    p4.add_argument("--force", action="store_true", help="已有向量也重新编码")
    p4.add_argument("--no-vectors", action="store_true", help="导出时不写向量数组")
    p4.set_defaults(func=cmd_embed)

    p5 = sub.add_parser("tokenize", help="补分词字段（content_ltks / title_tks 等）")
    p5.add_argument("--from", dest="from_json", required=True, help="chunks JSON 路径")
    p5.add_argument("--dump", default="", help="导出 JSON 路径")
    p5.add_argument("--show", type=int, default=2, help="打印前 N 条样例")
    p5.set_defaults(func=cmd_tokenize)

    p6 = sub.add_parser("index", help="写入 Elasticsearch（先按 doc_id 删、再批量写）")
    p6.add_argument("--from", dest="from_json", required=True, help="带向量的 chunks JSON")
    p6.add_argument("--force", action="store_true", help="删掉索引重建")
    p6.set_defaults(func=cmd_index)

    p7 = sub.add_parser("search", help="向量 + 关键词混合检索")
    p7.add_argument("q", help="查询文本")
    p7.add_argument("--top", type=int, default=5, help="返回条数")
    p7.add_argument("--k", type=int, default=50, help="knn 候选数")
    p7.add_argument("--doc-id", default="", help="只搜某篇论文")
    p7.add_argument("--text-boost", type=float, default=0.05, help="关键词路权重")
    p7.add_argument("--vector-boost", type=float, default=0.95, help="向量路权重")
    p7.set_defaults(func=cmd_search)

    p8 = sub.add_parser("retrieve", help="检索 → 重排 → 过滤 → 前 N → 聚合父块")
    p8.add_argument("q", help="查询文本")
    p8.add_argument("--top", type=int, default=0, help="最终条数（默认取配置 retrieve_top_n=2）")
    p8.add_argument("--candidates", type=int, default=0, help="ES 召回候选数（默认 10，决定重排费用与耗时）")
    p8.add_argument("--threshold", type=float, default=None, help="分数阈值")
    p8.add_argument("--budget", type=int, default=0, help="文本字符预算")
    p8.add_argument("--no-rerank", action="store_true", help="不启用重排模型")
    p8.set_defaults(func=cmd_retrieve)

    p9 = sub.add_parser("delete", help="删掉一篇论文的全部记录")
    p9.add_argument("--list", action="store_true", help="列出库里有哪些论文")
    p9.add_argument("--doc-id", default="", help="按 doc_id 删")
    p9.add_argument("--doc-name", default="", help="按标题片段删（不区分大小写）")
    p9.add_argument("--all", action="store_true", help="删全部")
    p9.add_argument("--dry-run", action="store_true", help="只看会删什么，不执行")
    p9.add_argument("--yes", action="store_true", help="确认执行")
    p9.set_defaults(func=cmd_delete)

    p10 = sub.add_parser("ingest", help="一条命令跑完：解析 → 层级 → 切块 → 向量 → 入库（带缓存）")
    p10.add_argument("md", help="full.md 路径")
    p10.add_argument("--title", default="", help="手工指定论文标题（默认取第一个标题）")
    p10.add_argument("--doc-id", default="", help="手工指定 doc_id")
    p10.add_argument("--force", action="store_true", help="忽略缓存，全部重跑")
    p10.add_argument("--no-index", action="store_true", help="只生成 chunk，不写 ES")
    p10.add_argument("--no-prefix", action="store_true", help="子块正文不加标题前缀")
    p10.set_defaults(func=cmd_ingest)

    p11 = sub.add_parser("cache", help="查看/清除中间结果缓存")
    p11.add_argument("--list", action="store_true", help="列出缓存（默认行为）")
    p11.add_argument("--clear", default="", metavar="DOC_ID", help="清掉某篇论文的缓存")
    p11.add_argument("--clear-all", action="store_true", help="清掉全部缓存")
    p11.set_defaults(func=cmd_cache)

    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
