# -*- coding: utf-8 -*-
"""ingest 子包：把文档变成可入库的切片。

各模块对应流水线的一个阶段，彼此只通过 models.Block 传递数据：

    markdown_blocks  切块        full.md -> list[Block]
    heading_levels   判层级      list[Block] -> list[int]
    section_index    章节编号    list[Block] -> 填 sec_no / sec_path
    block_merge      空块合并    list[Block] -> list[Block]
    chunk_renderer   渲染        list[Block] -> list[str]
"""
