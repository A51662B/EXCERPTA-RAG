# 论文知识库 · 操作手册

一条命令入库，一条命令提问。其他命令是排错用的。

---

## 一、三个命令就够用

```powershell
python run.py ingest "D:\...\full.md"          # ① 一篇论文入库
python run.py retrieve "你的问题"               # ② 提问（检索）
python run.py delete --list                    # ③ 看库里有什么 / 删掉
```

---

## 二、执行逻辑

### `ingest` 做了什么

```
full.md
  │
  ├─① 解析成块（免费）            extract_blocks
  │
  ├─② 判断标题层级（花钱）        LLM 一次调用
  │     └─ 结果存 levels.json
  │
  ├─③ 章节编号 + 空块合并（免费）
  │
  ├─④ 切父子块（免费）            超过 2000 字符的节 → 父块 + 子块
  │
  ├─⑤ 向量化（花钱）              只算子块和独立块，父块不算
  │
  ├─⑥ 分词（免费）                content_ltks / title_tks / create_time
  │     └─ 结果存 chunks.json
  │
  └─⑦ 写 ES（免费）               先按 doc_id 删旧记录，再批量写入
```

**缓存判断发生在最前面：**

| 缓存状态 | 会跳过 | 花费 |
|---|---|---|
| `chunks.json` 在 | ②③④⑤⑥ 全部跳过，直接 ⑦ | **0** |
| 只有 `levels.json` | 跳过 ②，重做 ③④⑤⑥ | 只花 embedding |
| 都没有 | 全跑 | LLM + embedding |

缓存位置：`data/cache/<doc_id>/<源文件哈希>/`
缓存键 = 标题哈希 + `full.md` 内容哈希（源文件一改，自动失效）

### `retrieve` 做了什么

```
问题
 ├─① 分词（免费）          "WDCANet Vaihingen mIoU 84.36" → "wdcanet vaihingen miou 84 36"
 ├─② 向量化（花钱）        一次 embedding 调用
 ├─③ ES 混合检索（免费）    BM25(文本) × 0.05  +  knn(向量) × 0.95，一次请求
 │                          召回 20 条候选
 ├─④ 重排（花钱）          20 条候选逐条打分
 ├─⑤ 过滤 + 截断（免费）    阈值过滤 → 取前 2
 ├─⑥ 聚父块（免费）        同一节的多个子块合成一条，分数取最高；回取父块正文
 └─⑦ 预算裁剪（免费）      按字符预算装，父块太大就退回命中的子块
```

---

## 三、命令参数速查

### `ingest <full.md>` —— 入库

| 参数 | 作用 | 什么时候用 |
|---|---|---|
| `--force` | 忽略缓存，全部重跑 | 改了 .env 里的切块参数、或缓存坏了 |
| `--no-index` | 只生成 chunk，不写 ES | 想先检查切块结果 |
| `--title "..."` | 手工指定标题 | 第一个标题取错了（比如取了期刊名） |
| `--doc-id "..."` | 手工指定 doc_id | 两篇论文同名，default 的哈希会撞 |
| `--no-prefix` | 子块正文不加标题前缀 | 一般不用 |

### `retrieve <问题>` —— 检索

| 参数 | 默认 | 作用 | 调整建议 |
|---|---|---|---|
| `--top N` | 2 | 最终给几条 | 想要更多上下文就调大 |
| `--candidates N` | 10 | 送去重排的条数 | **这是最贵的旋钮**；调小省钱，调大召回好 |
| `--threshold F` | 0 | 分数低于此值丢掉 | 重排分是 0~1；设 0.3 能滤掉明显不相关的 |
| `--budget N` | 24000 | 最终文本字符上限 | 嫌返回太长就设 4000，会自动退回子块 |
| `--no-rerank` | 关 | 不开重排模型 | 调试、省钱时用（排序会变差） |

### `delete` —— 删除

| 参数 | 作用 |
|---|---|
| `--list` | 列出库里所有论文（doc_id、条数、标题） |
| `--doc-name "片段"` | 按标题片段删（不区分大小写） |
| `--doc-id xxx` | 按 id 删 |
| `--all` | 删全部 |
| `--dry-run` | 预览会删什么，不执行 |
| `--yes` | 确认执行（不加则只看不做） |

### `cache` —— 缓存

| 参数 | 作用 |
|---|---|
| `--list` | 列出所有缓存（默认行为） |
| `--clear <doc_id>` | 清掉某篇的缓存 |
| `--clear-all` | 清掉全部缓存 |

### 分步命令（排错用，不常用）

| 命令 | 输入 → 输出 | 花钱 |
|---|---|---|
| `blocks <md>` | full.md → 块列表 | 免费 |
| `levels <md>` | 块 → 带层级的块 | **LLM** |
| `structure` | 块 → 编号 + 合并 | 免费 |
| `chunks` | 块 → 父子块 | 免费 |
| `embed --from <json>` | 块 → 带向量 | **embedding** |
| `tokenize --from <json>` | 块 → 带分词 | 免费 |
| `index --from <json>` | 块 → ES | 免费 |
| `search <问题>` | ES 单层检索（不重排、不聚合） | embedding |

常用共同参数：`--show N` 打印前 N 条、`--dump <路径>` 导出 JSON、`--from <json>` 从上一步产物载入。

---

## 四、配置（`.env`）

### 服务凭证（必需）

| 变量 | 作用 |
|---|---|
| `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `CHAT_MODEL` | 判标题层级用的 LLM |
| `EMBEDDING` / `EMBEDDING_API_KEY` | 嵌入模型（当前 `qwen3.7-text-embedding-flash`，1024 维） |
| `DASHSCOPE_API_KEY` | 重排模型（`qwen3.7-text-rerank`） |

### 行为参数（可选，都有默认值）

| 变量 | 默认 | 作用 | 什么时候改 |
|---|---|---|---|
| `PAPER_RAG_EMBED_MAX_CHARS` | 2000 | 切分阈值：超过就拆 | 换嵌入模型时（中文要调小到 600 左右） |
| `PAPER_RAG_CHILD_TARGET_CHARS` | 1200 | 子块目标大小 | 一般不用 |
| `PAPER_RAG_RETRIEVE_CANDIDATES` | 10 | 送回重排的条数 | 省钱/提召回 |
| `PAPER_RAG_RETRIEVE_TOP_N` | 2 | 最终条数 | 想要更多上下文 |
| `PAPER_RAG_PROMPT_CHAR_BUDGET` | 24000 | 文本预算 | 嫌返回长 |
| `PAPER_RAG_ES_URL` | `http://localhost:9200` | ES 地址 | — |
| `PAPER_RAG_ES_INDEX` | `papers` | 索引名 | 想分库时 |
| `PAPER_RAG_CACHE_DIR` | `data/cache` | 缓存目录 | 一般不用 |

改完 `.env` 后，切块相关的参数需要 `ingest --force` 才生效（因为缓存）。

---

## 五、费用一览

一次完整入库（一篇论文、80 条记录）实测：

| 步骤 | 花钱 | 用量 |
|---|---|---|
| 判层级 | 1 次 LLM 调用 | 输入约 1.9k token，输出约 50 token |
| 向量化 | 68 条 | 16883 token |
| 重排（每次提问） | 10 条 | 约 9500 字符 |
| 其余 | 免费 | 解析、切块、分词、写 ES |

**省钱要点：**
1. 同一篇论文**第二次跑免费**（缓存命中）
2. 提问时 `--candidates 10` 能把重排费用砍一半
3. 调试排序用 `--no-rerank`，只花一次 embedding

---

## 六、文件地图

项目里的 `data/` 目录：

```
data/
├── cache/                                  ← 缓存（ingest 用它省钱）
│   └── <doc_id>/<源文件哈希>/
│       ├── meta.json      版本、来源、chunk 数
│       ├── levels.json    带层级的块（省 LLM 钱）
│       └── chunks.json    带向量和分词的块（省 embedding 钱）
│
├── blocks.json                ← 调试产物：原始切块（无层级）
├── blocks_leveled.json        ← 调试产物：带层级
├── blocks_structured.json     ← 调试产物：编号 + 合并
├── chunks.json                ← 调试产物：切好的父子块
├── chunks_embedded.json       ← 调试产物：带向量
└── chunks_final.json          ← 调试产物：最终形态
```

**`cache/` 是正式流程用的，其余是 `--dump` 出来的调试快照。** 删掉不影响正式流程，只影响你回看。

---

## 七、常见操作

**重新入库同一篇论文（内容没变）**
```powershell
python run.py ingest "...\full.md"          # 命中缓存，0 花费
```

**重新入库同一篇论文（重新解析过，内容变了）**
```powershell
python run.py ingest "...\full.md"          # 源文件哈希变了 → 自动全跑
```

**改了切块参数后要重跑**
```powershell
python run.py ingest "...\full.md" --force  # 忽略缓存
```

**想看看切块结果对不对（不写库）**
```powershell
python run.py ingest "...\full.md" --no-index
python run.py chunks --from data\blocks_leveled.json --show 10
```

**清缓存**
```powershell
python run.py cache --list                  # 先看
python run.py cache --clear 2fcd744464c1    # 清一篇
python run.py cache --clear-all             # 清全部
```

**ES 里数据乱了想重建**
```powershell
python run.py index --from data\chunks_final.json --force
```

---

## 八、已知限制

| 限制 | 影响 | 缓解 |
|---|---|---|
| `84.36` 被切成 `84` + `36` | 数字类查询噪音大 | 待修（改分词器，无需重新 embedding） |
| 同名论文 `doc_id` 会撞 | 两篇同名论文会互相覆盖 | 用 `--doc-id` 手工区分 |
| `content` 字段不建索引 | 只能搜 `content_ltks` | 这是对的，和 RAGFlow 一致 |
| 只支持英文分词 | 中文按 2-gram 近似切 | 中文论文要调小 `EMBED_MAX_CHARS` |
| 单步命令不写缓存 | 一步步跑不会积累缓存 | 正式流程走 `ingest` |
