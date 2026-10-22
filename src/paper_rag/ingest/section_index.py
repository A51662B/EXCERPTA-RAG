# -*- coding: utf-8 -*-
"""阶段 3：给每个块算章节编号和章节路径。

输入：已经填好 level 的 blocks（阶段 2 的产物）
输出：原地填好 sec_no（"3.2"）和 sec_path（"III. METHOD > A. xxx"），返回同一个列表

规则
----
1. 文档题目块（idx==0 且 md_level==1）不参与章号：sec_no="0"，sec_path 是它自己的标题。
   理由：经测试，MinerU 用 `#` 标题目、`##` 标章节，若不排除，真正的第 1 章会变成第 2 章。
2. 其余块按 level 计数：进入第 L 层时，计数[L] +1，比 L 更深的计数清零。
   sec_no = 计数[1..L] 用点连接。
3. sec_path 是"祖先标题 > 自己标题"的拼接，只保留当前路径上的祖先。
"""
from __future__ import annotations

from paper_rag.models import Block

MAX_LEVEL = 9


def number_sections(blocks: list[Block]) -> list[Block]:
    """原地填写 sec_no / sec_path，返回 blocks 便于链式调用。"""
    counters = [0] * (MAX_LEVEL + 1)
    path: list[str] = []  # 当前路径上的标题

    for block in blocks:
        # 题目块：编号 0，路径是它自己
        if block.is_document_title:
            block.sec_no = "0"
            block.sec_path = block.title.strip()
            continue

        # 没有标题的块（首个标题之前的正文）不参与编号
        if not block.title.strip():
            block.sec_no = ""
            block.sec_path = ""
            continue

        level = block.level or 1
        level = max(1, min(level, MAX_LEVEL))

        counters[level] += 1
        for deeper in range(level + 1, MAX_LEVEL + 1):
            counters[deeper] = 0

        block.sec_no = ".".join(str(counters[i]) for i in range(1, level + 1))

        # 路径：截断到 level-1 层，再追加自己
        del path[level - 1:]
        path.append(block.title.strip())
        block.sec_path = " > ".join(path)

    return blocks


def is_ancestor_of(candidate: Block, target: Block) -> bool:
    """candidate 是不是 target 的上级（用于判断空块标题是否已在目标路径里）。"""
    if not candidate.sec_path or not target.sec_path:
        return False
    return target.sec_path.startswith(candidate.sec_path + " > ")
