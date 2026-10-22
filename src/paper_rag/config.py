# -*- coding: utf-8 -*-
"""运行期配置。

优先级：环境变量 > 项目根目录的 .env > 代码里的默认值。
不引入 python-dotenv 依赖，自己解析 .env（只支持 KEY=VALUE 这一种写法）。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

_ENV_PREFIX = "PAPER_RAG_"


def load_env_file(path: Path | None = None) -> None:
    """把 .env 里的键写进 os.environ；已存在的环境变量不覆盖。"""
    path = path or ENV_FILE
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            os.environ.setdefault(key, value.strip().strip('"').strip("'"))


def _env(*names: str, default: str = "") -> str:
    """按顺序取第一个非空的环境变量。"""
    for name in names:
        v = os.environ.get(name)
        if v and v.strip():
            return v.strip()
    return default


def _env_int(*names: str, default: int) -> int:
    v = _env(*names)
    return int(v) if v else default


# dataclass 的字段默认值在「类定义时」求值，不是实例化时。
# 所以必须在定义 Settings 之前把 .env 读进 os.environ，否则默认值取不到。
load_env_file()


def _abs(p: str | Path) -> Path:
    """相对路径按项目根目录展开，避免"在哪个目录下执行命令"影响产物位置。"""
    path = Path(p)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


@dataclass(frozen=True)
class Settings:
    """流程参数：只放流程会读到的字段。"""

    # 产物目录
    data_dir: Path = _abs(_env(_ENV_PREFIX + "DATA_DIR", default="data"))
    # 中间结果缓存：同一篇论文重跑时不重复花 LLM / embedding
    cache_dir: Path = _abs(_env(_ENV_PREFIX + "CACHE_DIR", default="data/cache"))

    # --- 切片大小 ---
    # embed_max_chars 是「编码模型能吃下的最大字符数」，它同时是切分的唯一阈值：
    #   - 一段 > embed_max_chars → 必须拆成若干 ≤ embed_max_chars 的子块
    #   - 一段 ≤ embed_max_chars → 不拆（solo）
    # 换算依据：token 上限 × 每 token 字符数 × 余量
    #   英文：512 token × 4 字符 ≈ 2048 → 取 2000
    #   中文：512 token × 1~1.5 字符 ≈ 600 → 换成中文论文时要调小
    embed_max_chars: int = _env_int(_ENV_PREFIX + "EMBED_MAX_CHARS", default=2000)
    # 子块的目标大小（软目标，可以小于等于上限；实际不会超过 embed_max_chars）
    child_target_chars: int = _env_int(_ENV_PREFIX + "CHILD_TARGET_CHARS", default=1200)

    # 检索引擎（阶段 6/7 用）
    es_url: str = _env(_ENV_PREFIX + "ES_URL", default="http://localhost:9200")
    es_index: str = _env(_ENV_PREFIX + "ES_INDEX", default="papers")

    # 嵌入模型（OpenAI 兼容接口：POST {base_url}/embeddings）
    # .env 里已有的命名：EMBEDDING = base_url，EMBEDDING_API_KEY = key
    embed_base_url: str = _env(_ENV_PREFIX + "EMBED_BASE_URL", "EMBEDDING")
    embed_api_key: str = _env(_ENV_PREFIX + "EMBED_API_KEY", "EMBEDDING_API_KEY", "DASHSCOPE_API_KEY")
    embed_model: str = _env(_ENV_PREFIX + "EMBED_MODEL", default="qwen3.7-text-embedding-flash")
    # 实测该接口一次最多 25 条，留余量取 20
    embed_batch: int = _env_int(_ENV_PREFIX + "EMBED_BATCH", default=20)
    # 向量维度：0 表示运行时探测
    embed_dims: int = _env_int(_ENV_PREFIX + "EMBED_DIMS", default=0)

    # --- 重排模型（DashScope 原生接口，不是 OpenAI 兼容路径）---
    rerank_url: str = _env(
        _ENV_PREFIX + "RERANK_URL",
        default="https://cn-beijing.maas.qianwenaiapi.com/api/v1/services/rerank/text-rerank/text-rerank",
    )
    # 优先用专门的 RERANK_API_KEY；没有就沿用 embedding 那把（同一个 key）
    rerank_api_key: str = _env(_ENV_PREFIX + "RERANK_API_KEY", "DASHSCOPE_API_KEY", "EMBEDDING_API_KEY")
    rerank_model: str = _env(_ENV_PREFIX + "RERANK_MODEL", default="qwen3.7-text-rerank")
    rerank_enabled: bool = _env(_ENV_PREFIX + "RERANK_ENABLED", default="1") not in ("0", "false", "False", "")

    # --- 检索参数 ---
    # 候选数 = 送去重排的条数，直接决定 rerank 的费用和耗时；最终条数 = 塞给 LLM 的段数
    retrieve_candidates: int = _env_int(_ENV_PREFIX + "RETRIEVE_CANDIDATES", default=10)
    retrieve_top_n: int = _env_int(_ENV_PREFIX + "RETRIEVE_TOP_N", default=2)
    retrieve_threshold: float = float(_env(_ENV_PREFIX + "RETRIEVE_THRESHOLD", default="0"))
    # 最终塞给 LLM 的文本预算（按字符估，英文 4 字符 ≈ 1 token）
    prompt_char_budget: int = _env_int(_ENV_PREFIX + "PROMPT_CHAR_BUDGET", default=24000)

    # LLM（阶段 2 用）：兼容项目里已有的 OPENAI_* / CHAT_MODEL 命名
    llm_base_url: str = _env(_ENV_PREFIX + "LLM_BASE_URL", "OPENAI_BASE_URL")
    llm_api_key: str = _env(_ENV_PREFIX + "LLM_API_KEY", "OPENAI_API_KEY")
    llm_model: str = _env(_ENV_PREFIX + "LLM_MODEL", "CHAT_MODEL", default="deepseek-flash")
    # 这个任务上 thinking 又慢又贵还更容易判错，默认关
    llm_thinking: bool = _env(_ENV_PREFIX + "LLM_THINKING", default="0") not in ("0", "false", "False", "")


def load_settings() -> Settings:
    load_env_file()
    return Settings()
