# -*- coding: utf-8 -*-
"""paper_rag：本地论文知识库的 RAG 流水线。"""
from __future__ import annotations

import os
import socket

__version__ = "0.1.0"


def _prefer_ipv4() -> None:
    """把 DNS 结果里的 IPv4 排到前面。

    为什么需要：某些域名的 AAAA 记录不可达，但 Python 的 socket.create_connection
    会按 getaddrinfo 返回的顺序逐个尝试。遇到不可达的 IPv6 时，每个地址会卡满
    系统 connect 超时（Windows 上约 21 秒），两个就是 42 秒——API 本来 0.3 秒能完成
    的请求会变成 43 秒。

    这里只改顺序、不丢弃 IPv6：IPv4 优先尝试，失败了仍会回落。
    设 PAPER_RAG_PREFER_IPV4=0 可关闭。
    """
    if os.environ.get("PAPER_RAG_PREFER_IPV4", "1") in ("0", "false", "False", ""):
        return
    original = socket.getaddrinfo
    if getattr(original, "_paper_rag_ipv4_first", False):
        return

    def ipv4_first(host, port, family=0, type=0, proto=0, flags=0):
        infos = original(host, port, family, type, proto, flags)
        return sorted(infos, key=lambda i: 0 if i[0] == socket.AF_INET else 1)

    ipv4_first._paper_rag_ipv4_first = True  # type: ignore[attr-defined]
    socket.getaddrinfo = ipv4_first            # type: ignore[assignment]


_prefer_ipv4()
