# EXCERPTA-RAG

把 MinerU 解析出的论文 markdown，变成可以问、可以引用、可以给 agent 调用的本地知识库。

PDF 解析交给 MinerU，本仓库负责**从 `full.md` 到可检索切片**的全部处理：
切块 → 判章节层级 → 编号合并 → 切父子块 → 向量化 → 分词 → 写 Elasticsearch → 混合检索 → 重排 → 按父块聚合。

```console
$ python run.py retrieve "WDCANet Vaihingen mIoU"

查询            : WDCANet Vaihingen mIoU
查询分词        : wdcanet vaihingen miou
重排模型        : qwen3.7-text-rerank
流程            : ES 召回 10 条 → 重排 10 条（送去 9097 字符）→ 截断到前 2 条 → 聚合父块后 2 条

最终 2 条，共 16997 字符：

  1. [父块] 分数 0.9985  sec 4.4  13653 字符
     路径: IV. EXPERIMENTS > D. Performance Comparison
     开头: To assess the effectiveness of the proposed WDCANet, extensive experiments were ...
     命中子块: 37f6298531633161
```

---

## 它解决什么问题

读论文时真正想知道的往往是「这个方法在某数据集上多少 mIoU」「他们为什么说前人没用」，
而通用 RAG 方案（LangChain / RAGFlow）要么太重（MySQL + Redis + worker 进程），要么切片策略
不适合论文这种强层级的文本。

这个项目只做一件事：**把一篇论文切成带章节坐标的块，检索时能指回「第 4.4 节」**。

三个设计取舍：

| 取舍 | 做法 | 为什么 |
|---|---|---|
| 切片大小由嵌入模型决定 | 超过 `EMBED_MAX_CHARS`（默认 2000 字符）就拆父子块 | 切片阈值应该等于编码上限，而不是拍一个数字 |
| 子块不复制父块正文 | 子块只存 `parent_id`，检索时按 id 批量取父块 | RAGFlow 把父块正文塞进每个子块（存 N+1 次），这里只存一份 |
| id 用内容哈希 | `chunk_id = sha1(内容 + doc_id)`，`doc_id = sha1(标题)` | 重跑是覆盖而不是重复；改文件夹名不会变成两篇论文 |

---

## 特性

- **一条命令入库**：`ingest` 串起解析后的全部处理，命中缓存时 0 次 LLM、0 次 embedding
- **可核查的检索**：每条结果带 `sec_no` / `sec_path` / `doc_name`，能直接回溯原文
- **混合检索**：向量 + 关键词（标题词加权），融合后按需重排
- **父块聚合**：命中同一节的多个子块合并成一条，分数取子块最大值
- **给 agent 用**：内置 MCP 服务端，暴露 4 个工具（列表 / 检索 / 删除 / 入库）
- **零依赖核心**：只用标准库，不需要 `elasticsearch` 客户端

---

## 快速开始

### 0. 前置条件

| 需要 | 说明 |
|---|---|
| Python 3.10+ | 核心只用标准库 |
| Elasticsearch 8.x | 本地起一个，单节点即可 |
| 嵌入模型 | 任意 OpenAI 兼容的 `/embeddings` 接口 |
| 对话模型 | 任意 OpenAI 兼容的 `/chat/completions`，用来判标题层级 |
| 重排模型（可选） | 有则排序更准；没有也能跑（`--no-rerank`） |

### 1. 起 ES

```bash
docker run -d --name es -p 9200:9200 \
  -e discovery.type=single-node \
  -e xpack.security.enabled=false \
  -e ES_JAVA_OPTS="-Xms1g -Xmx1g" \
  elasticsearch:8.15.0
```

### 2. 配置

```bash
git clone <your-repo-url> my-ragflow
cd my-ragflow
pip install -e .

cp .env.example .env
# 填入 EMBEDDING / EMBEDDING_API_KEY / OPENAI_BASE_URL / OPENAI_API_KEY
```

### 3. 入库一篇

`full.md` 是 MinerU 解析 PDF 的产物（同目录通常还有 `images/`）。

```bash
python run.py ingest "D:/papers/some-paper/full.md"
```

输出：

```
源文件          : D:/papers/some-paper/full.md
doc_id          : 2fcd744464c1
缓存状态        : levels=False  chunks=False
层级来源        : LLM（llm）
chunk           : 共 80 条（solo 5，parent 12，child 63）
写入            : 成功 80 条，失败 0 条
本次花费        : LLM 1 次，embedding 68 条，耗时 6.1s
```

再跑一次同样的命令：`本次花费 : LLM 0 次，embedding 0 条，耗时 0.8s`（命中缓存）。

### 4. 检索

```bash
python run.py retrieve "WDCANet Vaihingen mIoU"
```

```
流程 : ES 召回 10 条 → 重排 10 条（送去 9097 字符）→ 截断到前 2 条 → 聚合父块后 2 条
最终 2 条，共 16997 字符：

  1. [父块] 分数 0.9985  sec 4.4  13653 字符
     路径: IV. EXPERIMENTS > D. Performance Comparison
```

---

## 流水线

```
full.md
  │
  ├─ 1  切块             ingest/markdown_blocks.py    # / ## 为界            免费
  ├─ 2  判标题层级       ingest/heading_levels.py     LLM 一次调用            花钱
  ├─ 3  章节编号         ingest/section_index.py      1 / 2.1 / 4.4          免费
  ├─ 4  空块合并         ingest/block_merge.py        II. RELATED WORK 这类    免费
  ├─ 5  切父子块         ingest/chunk_split.py        超过阈值就拆             免费
  ├─ 6  向量化           index/embedder.py            只算子块，父块不算       花钱
  ├─ 7  分词             tokenizer.py                 挖词 + 时间戳            免费
  └─ 8  写 ES            index/es_store.py            先按 doc_id 删后写       免费
```

检索：

```
问题 → 分词 ─┬─ ES 混合检索（BM25 × 0.05 + knn × 0.95）
             │      └─ 召回 candidates 条
             ├─ 重排（rerank 模型，候选条数决定费用）
             ├─ 截断到 top_n
             ├─ 按 parent_id 聚合，分数取子块最大值
             └─ 按字符预算裁剪
```

---

## 命令

| 命令 | 作用 | 花钱 |
|---|---|---|
| `ingest <full.md>` | 一条命令跑完全流程（带缓存） | 首次花钱，之后免费 |
| `retrieve "问题"` | 检索 + 重排 + 父块聚合 | 重排 |
| `delete --list` / `--doc-name` / `--doc-id` | 查看和删除入库文档 | 免费 |
| `cache --list` / `--clear` | 查看和清理中间结果缓存 | 免费 |
| `blocks` / `levels` / `structure` / `chunks` | 单步调试 | 视步骤 |
| `embed` / `tokenize` / `index` | 单步入库 | 视步骤 |
| `search "问题"` | 只看 ES 单层结果（不重排、不聚合） | 嵌入 |

常用参数：

```bash
python run.py ingest path/to/full.md --force        # 忽略缓存重跑
python run.py retrieve "..." --top 3                # 多取几段
python run.py retrieve "..." --candidates 5         # 减少重排费用
python run.py retrieve "..." --budget 6000          # 限制塞给模型的字符数
python run.py retrieve "..." --no-rerank            # 关掉重排
```

完整说明见 [MANUAL.md](MANUAL.md)。

---

## 用于 agent（MCP）

`mcp_server.py` 是一个 MCP 服务端（stdio 传输），暴露 4 个工具：

| 工具 | 参数 | 作用 |
|---|---|---|
| `kb_list` | — | 列出已入库文档 |
| `kb_search` | `query`, `top_n`, `doc_id` | 检索，返回带章节引用的段落 |
| `kb_delete` | `doc_id` | 删除一篇文档 |
| `kb_ingest` | `md_path`, `title`, `force` | 入库一篇 markdown |

启动（客户端会把它当子进程拉起，不需要端口）：

```bash
python mcp_server.py
```

在支持 MCP 的客户端里配置：

```json
{
  "mcpServers": {
    "paper-rag": {
      "command": "python",
      "args": ["D:/path/to/my-ragflow/mcp_server.py"]
    }
  }
}
```

需要额外装 MCP SDK：`pip install "mcp>=2.0"`。

---

## 项目结构

> 仓库名是 EXCERPTA-RAG，Python 包名保留为 `paper_rag`（历史原因），
> 所以代码里看到的是 `from paper_rag import ...`。日常使用统一走 `python run.py <命令>`。

```
my-ragflow/
├── run.py                     入口（把 src 加进 sys.path，无需 PYTHONPATH）
├── mcp_server.py              MCP 服务端
├── MANUAL.md                  操作手册（参数、执行逻辑、费用）
├── assets/                    README 用图
├── src/paper_rag/
│   ├── config.py              配置（读 .env）
│   ├── models.py              Block / Chunk 数据模型
│   ├── cli.py                 子命令
│   ├── cache.py               中间结果缓存
│   ├── tokenizer.py           分词（英文 + 中文 2-gram）
│   ├── io_json.py             chunks 读写
│   ├── llm.py                 对话模型客户端
│   ├── kb_tools.py            MCP 工具的业务实现
│   ├── ingest/                写路径
│   ├── index/                 向量化 / ES / 重排
│   └── retrieve/              检索流水线
├── tests/                     134 个单元测试
└── data/                      产物（git 忽略）
```

---

## 设计说明

**为什么 `content` 字段不建索引**

ES 里 `content` 是 `text` + `index: false`，只存不搜。检索走 `content_ltks`（自己分好词、whitespace 分析器）。
这样分词规则掌握在自己手里，中文不会被 ES 切成单字，也方便以后在应用层重算分数。

**为什么 `sec_path` 用 multi-field**

`sec_path` 同时是 `text`（可部分匹配）和 `sec_path.keyword`（精确过滤）。一个字段满足「搜 performance」
和「只看 4.4 节」两种需求。

**为什么入库要按 doc_id 先删后写**

`chunk_id` 是内容哈希，切块阈值一变 id 就变，旧记录会变成孤儿。按 `doc_id` 整体删除再写，能保证库里没有残留。

**为什么加了 IPv4 优先补丁**

某些 API 域名的 AAAA 记录不可达，而 Python 的 `socket.create_connection` 会按 DNS 顺序逐个尝试，
每个 IPv6 卡满 21 秒系统超时，两个就是 42 秒。`paper_rag/__init__.py` 把 IPv4 排到前面，
只改顺序不丢弃 IPv6。实测 43.85s → 0.72s。

---

## 实测数据

20 篇遥感语义分割论文（MinerU 解析）：

| 指标 | 数值 |
|---|---|
| 记录数 | 1557 条（含父块） |
| 可检索条数 | 1288 条 |
| 单篇切块 | 26 ~ 117 条 |
| 入库耗时 | 1.7 ~ 6.1 秒/篇（首次） |
| 检索耗时 | 0.7 秒（10 条候选，含嵌入与重排） |
| 重排 10 条 | 0.26 秒 |

---

## 已知限制

| 限制 | 影响 | 状态 |
|---|---|---|
| `84.36` 被切成 `84` 和 `36` | 数字类查询噪音偏大 | 待修（改分词器即可，不需要重新向量化） |
| 同名论文会撞 `doc_id` | 两篇同名论文互相覆盖 | 用 `ingest --doc-id` 手工区分 |
| 中文按 2-gram 近似切词 | 中文论文召回率低于英文 | 建议接中文分词器 |
| 公式不渲染 | 检索结果里的 LaTeX 是原文 | 展示层的问题，本项目不处理 |
| 单步命令不写缓存 | 一步步跑不会积累缓存 | 正式流程走 `ingest` |

---

## 测试

```bash
python -m unittest discover -s tests
```

134 个用例，覆盖切块边界、层级校验、编号、合并、父子切分、分词、缓存、ES 请求载荷、检索流水线。

---
