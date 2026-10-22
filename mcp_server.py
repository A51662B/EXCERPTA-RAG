# -*- coding: utf-8 -*-
"""论文知识库 MCP 服务端（官方 SDK，mcp 2.x）。

被 agent 框架当子进程启动，通过 stdin/stdout 走 MCP 协议。

启动：python mcp_server.py        （默认 stdio，不需要端口）

用到的解释器必须装过 mcp（2.x）。AEGIS 那边用的是 LanggraphAgent 环境，
所以它 spawn 这个脚本时会用同一个解释器，天然满足。

提示词原则
----------
参数说明只讲「怎么填」，不举例具体论文里的名词（模型名、数据集名、模块名）。
一旦举例，模型会倾向于照着例子搜，等于把知识库范围缩到了那篇论文。
具体的使用策略写在 skill 里，工具描述只负责说清接口。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mcp.server.mcpserver import MCPServer  # noqa: E402
from pydantic import Field  # noqa: E402

from paper_rag import kb_tools  # noqa: E402

server = MCPServer("paper-rag")


@server.tool()
def kb_list() -> dict:
    """列出知识库里已入库的文档（标识、标题、记录条数、可检索条数）。

    用于回答「库里有哪些文档」，或在删除、限定检索范围之前获取文档标识。
    返回的标识是后续 kb_search / kb_delete 的合法入参，不要凭空构造。

    这是本地 RAG 知识库工具。使用前先通过 skilltool 读取 rag_kb 规则。
    """
    return kb_tools.kb_list()


@server.tool()
def kb_search(
    query: Annotated[
        str,
        Field(
            description=(
                "检索关键词。传 1~3 个术语，用文档里的原文用语"
                "（指标名、方法名、数据集名、章节主题词都行）。"
                "不要传完整问句，也不要用同义词替换原文术语。"
            )
        ),
    ],
    top_n: Annotated[
        int,
        Field(
            ge=1,
            le=6,
            description=(
                "返回几段。默认 2，一般足够回答一个具体问题。"
                "调大会显著增加上下文占用和响应时间，且单段可能长达数千字符。"
            ),
        ),
    ] = 2,
    doc_id: Annotated[
        str,
        Field(
            description="只在这一篇文档内检索；不填则检索全部。取值必须来自 kb_list 的返回。"
        ),
    ] = "",
) -> dict:
    """在知识库中检索与关键词相关的段落，返回按相关性排序的结果和章节引用。

    适用于回答「文档里写了什么」这类事实性问题：数值、方法细节、实验设置与结论。
    返回的每条结果带章节号与章节路径，回答时应引用它们。
    truncated 为 true 表示该段原文超长、已被截断，不要当作完整内容使用。

    这是本地 RAG 知识库的检索工具。使用前先通过 skilltool 读取 rag_kb 规则，
    里面规定了检索次数上限、查询写法和引用格式。
    """
    return kb_tools.kb_search(query=query, top_n=top_n, doc_id=doc_id)


@server.tool()
def kb_delete(
    doc_id: Annotated[
        str,
        Field(
            description=(
                "要删除的文档标识，必须来自 kb_list 的返回。"
                "传错不会误删，会返回可用标识列表供重选。"
            )
        ),
    ],
) -> dict:
    """从知识库中删除一篇文档的全部记录（切块、向量、索引一并清除）。

    这是不可逆操作。执行前应先调用 kb_list 确认标识，并已得到用户明确同意。
    这是本地 RAG 知识库工具，使用前先通过 skilltool 读取 rag_kb 规则。
    """
    return kb_tools.kb_delete(doc_id=doc_id)


@server.tool()
def kb_ingest(
    md_path: Annotated[
        str,
        Field(
            description=(
                "已解析好的 markdown 文件的绝对路径，文件必须已存在。"
                "只接受 markdown；如果是 PDF，需要先转成 markdown 再入库。"
            )
        ),
    ],
    title: Annotated[
        str,
        Field(
            description=(
                "文档标题。不填则自动取文件里的第一个标题，通常正确。"
                "标题决定文档标识，随意填写会被当成另一篇文档。"
                "只在自动取到的标题明显不对时才需要填。"
            )
        ),
    ] = "",
    force: Annotated[
        bool,
        Field(
            description=(
                "忽略缓存全部重跑。会重新产生解析与向量化费用。"
                "只在切块参数变化、或源文件重新解析过时才用；"
                "普通重试、重复入库都不要传 true。"
            )
        ),
    ] = False,
) -> dict:
    """把一篇已解析的 markdown 文档入库（切块、向量化、写索引）。

    首次入库需要数秒并产生 API 费用；同一文件重复调用会命中缓存，几乎免费。
    返回的文档标识之后可用于限定检索范围或删除。
    这是本地 RAG 知识库工具，使用前先通过 skilltool 读取 rag_kb 规则
    （其中包含 PDF 需要先转 markdown 的入库流程）。
    """
    return kb_tools.kb_ingest(md_path=md_path, title=title, force=force)


if __name__ == "__main__":
    server.run()
