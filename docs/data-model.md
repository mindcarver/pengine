---
layout: default
title: 数据与一致性
permalink: /data-model/
---

{% include nav.md %}

# 数据与一致性

Pengine 把“能不能展示进度”和“能不能交付成品”建立在持久化证据上。内存对象、模型响应和浏览器页面都不是最终权威。

## 1. 本地目录

```text
PENGINE_DATA_DIR/
├── pengine.sqlite3       # 业务状态、审计和 LangGraph checkpoint 共用的 SQLite 文件
├── pengine.sqlite3-wal   # SQLite WAL 运行期间可能存在
├── pengine.sqlite3-shm   # SQLite WAL 运行期间可能存在
└── persona-snapshots/
    └── <snapshot_sha256>/
        ├── manifest.json
        └── paradigm.md ... l6.md
```

人格源目录 `PENGINE_PERSONA_ROOT` 是操作员管理的输入，不属于这个快照树；应用不会把生成内容写回人格源目录。

## 2. 两种状态的权威性

### 业务状态

由 `Repository` 事务管理，决定：

- creation 和 initial/revision run 的公开状态；
- job lease 和阶段尝试次数；
- 哪些阶段 checkpoint 已批准；
- 哪个设计候选/batch/episode pointer 是 active；
- 是否允许 revision、continue、repair authorization 或 end；
- 是否可组装正式 delivery。

### Agent 状态

由 `AsyncSqliteSaver` 管理，保存一个 Deep Agents thread 的消息、计划和 virtual `StateBackend` scratch。它帮助同一个 run 继续执行，但不拥有业务批准权。

```text
业务 checkpoint = 可公开、可交付的事实
LangGraph checkpoint = 可恢复、但仍可能未批准的执行上下文
```

如果 Agent thread 说“已经完成”，但业务 checkpoint 没有对应阶段，公开资源仍按未完成处理。

## 3. 逻辑表分组

当前 `src/pengine/repository.py` 的迁移头版本为 `21`。下表按领域归纳表，而不是要求调用方直接依赖内部 SQL；SQL schema 变化必须通过迁移维护。

| 领域 | 逻辑表 | 作用 |
| --- | --- | --- |
| 元数据 | `pengine_schema` | 迁移版本 |
| 作品与运行 | `creations`、`runs` | 原始输入、人格快照引用、初稿/修订 run、顺序和状态 |
| 调度 | `jobs`、`stage_attempts` | 队列、租约、下次运行时间、阶段尝试保护 |
| 业务 checkpoint | `business_checkpoints` | 通过验证的方向、设计产物、合同和阶段结果；锁定合同可绑定 `review_call_id` |
| 进度 | `run_progress`、`episode_plans`、`episode_drafts`、`episode_attempts`、`episode_attempt_cycles`、`episode_attempt_current`、`episode_timeouts` | 可恢复进度、按重写周期隔离的逐集尝试和超时证据 |
| 交付与修订 | `deliveries`、`frozen_revisions` | 成功交付、冻结 feedback、修订资格 |
| 命令幂等 | `idempotency_records` | key、scope、payload hash 和已接受响应 |
| 质量/内容 | `quality_gate_rejections`、`content_rejections` | L0/L4 拒绝、合同/连续性拒绝、证据和修复计数 |
| 模型审计 | `model_calls` | 每次模型尝试、预检、实际用量、provider 证据和血缘 |
| 设计版本 | `series_bible_candidates`、`series_bible_lineage` | SeriesBible 候选、active/stale/superseded、重建预算 |
| 剧本版本 | `script_batches`、`episode_candidates` | 设计绑定的 batch、逐集不可变候选、active pointer 和前序 hash |
| 结构审查 | `series_reviews`、`repair_authorizations` | 绑定 design/batch/prefix 的里程碑审查与一次性授权 |
| Agent 执行 | LangGraph checkpointer 表 | thread checkpoint，表结构由依赖库管理 |

## 4. Creation、Run、Job、Thread 的关系

```text
Creation 1
 ├─ Initial Run 1 ── Job 1 ── Thread A
 ├─ Frozen Revision Feedback 0..1
 └─ Revision Runs 0..N（失败重排队会保留旧 run）
             ├─ Job N ── Thread B
             └─ Delivery 0..1（成功修订）
```

- 一个 `creation` 绑定一次用户输入和一个不可变人格 snapshot；
- `run` 代表一次完整初稿/修订执行，具有自己的状态、checkpoint 和 thread；
- `job` 是可以被 Worker 租约和重新入队的调度实体，不是用户资源；
- `thread_id` 只标识 LangGraph 的执行线程，不替代 `run_id`；
- 失败修订重排队会创建新的 run/thread，但保留原失败 run 供审计。

## 5. Hash 与版本边界

### 人格

```text
per-file sha256（按 persona schema 固定顺序）
        ↓
package_sha256 = SHA256(concat(file_hashes))
        ↓
snapshot_sha256 = SHA256(domain + package_sha256 + canonical_manifest)
```

manifest 自身不放进 `package_sha256`，避免循环；但规范化 manifest 会参与 `snapshot_sha256`，所以人格身份和该 schema 的完整 Markdown 集合共同决定最终快照。新任务使用完整挂载 `soul.md` 与 `l3.md` 的 v3 八文件集合；历史 v1/v2 snapshot 仍保持原身份和 L3 摘要投影。

L4 的来源正文不进入快照：快照只保存经过授权编译的 `l4.md` 及来源指纹。旧任务继续解析创建时绑定的旧 snapshot；人格版本升级只影响之后创建的新任务。Pengine 产品参数暂作为各 persona `l4.md` 的一致投影，但其权属不因此变成创作者规则。

### 设计

SeriesBible 候选的投影、合同和设计内容都属于同一个 candidate。候选、全局审核、promotion、design epoch 必须绑定同一 candidate/hash；旧候选不能借用新候选的审核结果。

### 逐集剧本

```text
episode candidate
  = design binding
  + batch/epoch
  + episode number/version
  + predecessor hash
  + script content hash
  + state delta + folded state hash
  + review evidence + model call lineage
```

candidate 是不可变记录；active pointer 是可变索引，但提交时会做 CAS/前序校验。迟到响应会被保留为 stale/non-active，不能移动 active pointer。

## 6. 事务边界

### 创建命令

创建命令必须保证下面的资源不会出现“只有半套”的状态：

```text
idempotency record
  + creation
  + initial run
  + job
  + thread_id
```

同一幂等 key 的重放返回已保存的接受响应；同 key 不同 payload 被拒绝。

### 阶段批准

模型返回候选后，业务层按以下顺序处理：

```text
parse structured result
  → validate exact stage
  → validate language/contract/hash/ordering/reference
  → run independent review if required
  → approve_business_checkpoint
```

`business_checkpoints` 已批准记录不可被普通重试覆盖。修订或后续阶段只能基于已批准内容继续。
带 `StoryContract` 的分集大纲还必须绑定一个同 run、审核角色、分集大纲阶段、状态为
`succeeded` 且具有 `operation_id` 的真实 `review_call_id`；不能用合成 ID 补来源。

### 单集提交

单集的剧本、`EpisodeStateDelta`、折叠 `SeriesState`、semantic review、repair rounds、
hash 和生成 `call_id` 必须原子提交。生成 `call_id` 必须来自同一 `operation_id` 下的真实
成功物理调用。提交前重新验证 design epoch、batch、active predecessor 和候选版本，
防止迟到响应把旧后缀写回当前序列。

### 正式交付

正式交付前，Repository 会重建完整剧本并检查：

- 每个必需业务 checkpoint 存在；
- active SeriesBible 和 script batch 仍匹配；
- 全集内容/状态 hash 可复验；
- L0、L4 gate 证据为通过；
- 绑定当前 lineage 的最终全剧 review 为通过；
- 运行仍处于可成功的状态。

任何一项失败都不能通过“补写一个 delivery row”绕过。

## 7. 模型调用审计

`model_calls` 的一条记录至少回答：

| 字段组 | 说明 |
| --- | --- |
| 身份 | 物理 `call_id`、业务 `operation_id`、run/creation/thread、run kind |
| 路由 | role、adapter、provider、requested model、实际 `response_model_ids` |
| 血缘 | stage、episode、candidate、batch、supersedes_call_id |
| 预算 | estimated input/output/total、verified limit、preflight |
| 结果 | started/succeeded/failed/timed_out/stale/superseded/preflight_blocked |
| usage | provider 实际 input/output/cache，或 partial/unavailable |
| 诊断 | finish reason、error code/type、HTTP status、provider code、脱敏 response、安全消息 |
| 人格挂载 | persona/snapshot 身份、Soul 与 L3 的 hash、字符数、挂载路径和完整挂载状态 |
| 时间 | requested/finished/duration |

实际用量缺失时不从估算值回填。估算是“是否允许发出请求”的安全预检证据，provider usage 才是实际使用证据。`l3_full_text_mounted` 只证明运行时装配了完整文本，不声称模型已在语义上采用；审计元数据不保存 L3 正文或摘录。

`operation_id` 不是 provider request id。Worker 在进入一个受保护阶段或一集写作时创建
operation；callback 为每次真正出站调用记录唯一 `call_id`。锁定产物时，Repository 会
反查同 run/role/stage/episode/operation 的成功记录，并在终态发布前等待审计 writer 落盘。

Schema 18 同时引入两组迁移：旧 `episode_attempts` 进入 `attempt_cycle=0`，并增加
`episode_attempt_cycles`/`episode_attempt_current`；`model_calls` 增加 `operation_id`，
`business_checkpoints` 增加 `review_call_id` 以及相应索引。迁移在一个前向事务中完成。
Schema 21 以加法迁移为 `model_calls` 增加四个 L3 挂载审计字段，旧行保持原样并使用安全默认值。

L3 的正文不进入这些审计列，L4 的完整正文也不复制进 `DeliveryReport`：creation 通过
`persona_snapshot_sha256` 绑定精确人格资产，L3 只记录安全挂载元数据，最终 L4 只保存
`l4_gate` 证据。完整边界见 [L3 实际实现设计]({{ site.baseurl }}/l3-integration/) 与
[L4 实际实现设计]({{ site.baseurl }}/l4-integration/)。

## 8. SQLite、WAL 与备份

Repository 初始化时启用 WAL、`synchronous=NORMAL`、外键和 busy timeout；模型审计使用单独同步 writer，并在 run finalize 前 drain，避免和 LangGraph 异步写入形成“日志已返回但 audit 尚未落盘”的假象。

服务运行中优先使用 SQLite 自带 backup：

```bash
mkdir -p backups
sqlite3 data/pengine.sqlite3 ".backup 'backups/pengine-$(date +%Y%m%d-%H%M%S).sqlite3'"
```

同时备份 `data/persona-snapshots/`，因为数据库中的 creation 只保存 snapshot 引用。只备份 `.sqlite3` 而丢快照，重启后可能无法解析旧任务的人格上下文。

恢复前应停止 Worker/服务；恢复后用原数据目录启动，并检查：

1. `GET /personas` 能发现当前源人格；
2. 旧 creation 的 `snapshot_sha256` 仍能从 snapshot tree 解析；
3. `pengine_schema` 在支持范围内；
4. paused/running 任务的 job lease、checkpoint 和 model-call audit 可读；
5. 不要用手工 SQL 把 `run.state` 改成 `succeeded`。

## 9. 数据保留和敏感信息

V1 不自动过期故事、反馈、模型调用和快照，也没有选择性删除/导出 API。故事、persona 内容、provider 脱敏证据和本地数据库都应按敏感创作资产管理。

禁止进入 Git 的内容至少包括：

- `.env` 和 relay/API key；
- `data/`、SQLite/WAL、persona snapshots；
- live E2E 产物、完整模型 prompt/response 和含用户正文的日志；
- 个人素材、未授权人格内容、provider secret。
