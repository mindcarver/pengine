---
layout: default
title: 运行流程与恢复
permalink: /runtime/
---

{% include nav.md %}

# 运行流程与恢复

本页按真实执行顺序描述一个创作任务。界面展示的是业务状态的投影，不是模型内部思考过程。

## 1. 从命令到 Worker

```text
POST /creations
  │
  ├─ FastAPI 校验 persona_id / story / requirements / Idempotency-Key
  ├─ Repository 检查命令重放或冲突
  ├─ PersonaCatalog 在事件循环外生成/复用快照
  ├─ SQLite 事务写入 creation + initial run + job + thread_id
  └─ 返回 202 {creation_id, initial_state: queued, resource_url}
       │
       ▼
Worker lease job
  │
  ├─ 读取快照和已批准业务检查点
  ├─ 建立角色绑定模型与 model-call audit
  ├─ 以同一个 thread_id 调用 Deep Agents supervisor
  └─ 每个受保护阶段通过 StageGuard → 候选校验 → 业务检查点提交
```

请求返回 `202` 只代表命令已排队。调用方应轮询同一个 `GET /creations/{creation_id}` 资源，不应创建新任务来“查询进度”。

## 2. 阶段映射

### 内部阶段（9 个）

```text
loading_persona
  → selecting_l0_variant
  → generating_story_outline
  → generating_character_relationships
  → generating_episode_outline
  → generating_episode_scripts
  → accepting_l0
  → accepting_l4
  → assembling_delivery
```

### 对外进度阶段（6 个）

| 对外 `current_stage` | 覆盖的内部阶段 | 完成条件 |
| --- | --- | --- |
| `determining_direction` | `loading_persona`、`selecting_l0_variant` | 快照可读，L0 变体与理由已批准 |
| `generating_story_outline` | `generating_story_outline` | 故事大纲 checkpoint 已批准 |
| `generating_character_relationships` | `generating_character_relationships` | 人物小传和关系逻辑已批准 |
| `generating_episode_outline` | `generating_episode_outline` | 分集大纲、剧情合同、独立审查和 hash 已锁定 |
| `generating_episode_scripts` | `generating_episode_scripts` | 所有集候选均 active，连续状态可从前缀重放 |
| `final_review` | `accepting_l0`、`accepting_l4`、`assembling_delivery` | 两个 gate、全剧审查和完整交付事务均通过 |

界面的“人物与关系”和“成品审核”是用户阶段标签；它们不等于一个单独的模型调用。`FinalReviewProgress` 会分别显示 `l0`、`l4` 的 `pending/running/passed/paused/failed`。

## 3. 一次初稿成功路径

### 3.1 人格和方向

1. `PersonaCatalog` 只从有效人格目录发现可选项。
2. 创建事务固定 `persona_id`、版本和 `snapshot_sha256`。
3. Worker 从快照构造只读阶段上下文。
4. `story_architect` 选择 L0 变体并返回理由。
5. 业务层检查结构化结果后写入方向 checkpoint。

### 3.2 故事设计

1. `story_architect` 生成故事大纲。
2. `story_architect` 生成角色和关系逻辑。
3. `episode_planner` 生成分集大纲和 `StoryContract`。
4. 合同经过确定性校验，再由绑定该候选的 `canon_reviewer` 独立审核。
5. 合同及其 Markdown 投影、hash 一起写入 approved checkpoint。
6. 设计同步阶段将投影组装为一个 `SeriesBible` candidate；不允许把不同候选的故事大纲、人物关系和分集大纲拼在一起。

### 3.3 逐集写作

对每一集 N，Writer 的输入包含：完整 active SeriesBible 投影、锁定合同、当前集计划与义务、active 前缀 1..N-1 的完整剧本、折叠后的 `SeriesState` 和有界 WriterNotes。摘要不能替代已锁定前缀。

```text
script_writer
  → script + EpisodeStateDelta
  → deterministic contract/state validation
  → episode_reviewer semantic continuity review
  → optional bounded episode_repair
  → atomic candidate commit
```

只有提交成功的集才会进入 active pointer。当前集没提交时，API 不展示它的正文；已提交的前缀在刷新、结束或失败后仍可只读查看。

### 3.4 最终闸门和交付

1. Worker 从每个已锁定集重建完整剧本 checkpoint，复验合同 hash、剧本 hash 和状态 hash。
2. `quality_reviewer` 对完整内容生成 L0 和 L4 证据。
3. `series_reviewer` 对绑定当前设计/batch/active-prefix 的结构审查做最终分类。
4. 两个质量 gate 和绑定全剧审查都通过后，Repository 在同一业务边界内写入 `Delivery` 并把 run 置为 `succeeded`。
5. `ContentPackage` 固定包含：故事大纲、人物小传、关系逻辑、分集大纲、分集剧本；`DeliveryReport` 单独保存人格快照、L0/L4 证据、归属声明和修订反馈覆盖。

没有通过最终闸门时，设计包或单集候选可以作为可读进度/证据存在，但不能被当作正式交付。

## 4. Run 状态机

```text
                         ┌───────────────┐
                         │    queued     │
                         └──────┬────────┘
                                ▼
                         ┌───────────────┐
                         │    running    │
                         └──┬────┬───┬────┘
                            │    │   │
         首次暂时中断/超时 ──┘    │   └─ 用户结束 / 终态错误
                                 │          ▼
                                 │       ended / failed
                                 ▼
                         ┌───────────────┐
                         │ auto_resuming │
                         └──────┬────────┘
                                │ 同一 run + thread_id
                                └──────────────► running

        同一用户阶段第二次共享中断 / 内容修复耗尽 / 等待授权
                                ▼
                         ┌───────────────┐
                         │    paused     │
                         └──────┬────────┘
                                │ continue / authorize-repair
                                └──────────────► queued

        全部 checkpoint + final gates 通过 → succeeded
        final L0/L4 不通过 → quality_rejected（保留证据，可只重试最终审核）
```

资源中还会暴露 `recovery_state`、`recovery_reason`、`can_continue` 和 `can_end`。恢复理由包括 `run_timeout`、`relay_interruption`、`content_rejected`、`episode_error`、`context_budget` 和 `repair_authorization`。

## 5. 哪些错误可以恢复

恢复分类是在真实异常边界上作出的，不是所有异常都重试。

| 类别 | 例子 | 自动行为 | 操作员动作 |
| --- | --- | --- | --- |
| 暂时 relay/网络 | 请求开始后的连接、DNS、TLS、读取超时或重置；relay `429/502/503/504` | 首次在同一 run/thread 上进入 `auto_resuming`，遵守至少 10 秒或更长 `Retry-After` | 若同一用户阶段再次共享中断，`Continue` 或 `End` |
| 语法正确地址但连接失败 | 主机名解析/连接失败，无法证明一定短暂 | 按受限 transport 路径计入调用预算，耗尽后失败 | 修正 relay 配置后新建任务 |
| 配置/安全错误 | 缺 URL/key、非 loopback HTTP、证书校验失败 | 不自动降级，不切换模型 | 修正 `.env` 后重新运行 |
| 身份/协议错误 | provider 回报了错误模型、OpenAI/Anthropic 协议不匹配、结构化输出无效 | 终止为安全错误 | 检查 relay adapter、模型路由和响应合同 |
| 上下文预算 | 未设置可信上限，或序列化请求 + 保留输出超出上限 | 请求前阻断，0 outbound call，运行暂停为 `context_budget` | 增加已验证上限、缩小上下文或结束任务 |
| 内容审查不通过 | 合同、单集连续性、结构性里程碑失败 | 只做有界内容修复；预算耗尽后 `paused` | 需要 `authorize-repair` 才能消费一次授权周期，或保留并结束 |
| L0/L4 最终拒绝 | 质量闸门返回 rejected | `quality_rejected`，不重跑前面内容 | `retry-final-review` 只重跑同一最终审核，或结束 |
| checkpoint/图执行故障 | thread checkpoint 缺失、递归上限、未知内部错误 | `failed`，不把未验证 thread 内容升级为批准内容 | 保留错误和证据，修复运行环境后新建/重试适用命令 |

普通阶段和受保护子调用有最多三次尝试边界；第四次调用会在模型请求前被拒绝。Relay 恢复次数、阶段尝试次数和内容修复次数是不同预算，不能相互挪用。

## 6. 暂停后的三种动作

### Continue

只适用于运行时/relay/timeout 等可继续路径。它重新排队同一个 run 和同一个 `thread_id`，不会改变已批准 checkpoint，也不能花掉内容修复预算。

### Authorize repair

只适用于内容审查给出的 `repair_authorization`。授权绑定设计候选、batch、影响集数、review id、证据和参考上下文量，只允许一次 generation + review cycle。参考上下文量只统计暂停时的活动设计投影与保留前缀；这些文本不保证全部进入重建设计或每次逐集调用，因此它既不是下限，也不是整轮用量或费用预测。若仍有硬约束冲突，会按最新审查证据再次暂停，不会无限自动循环。

### End

终止当前 run，但保留已提交 checkpoint、逐集草稿、模型调用和审查证据。`ended` 不是成功，也不会使未提交正文变成正式交付。

## 7. 重启恢复

进程重启后使用同一个 `PENGINE_DATA_DIR`：

1. Repository 初始化 SQLite；当前实现会按 `pengine_schema` 版本顺序执行迁移。
2. 过期 job lease 回到队列，已暂停/已结束任务不会被误重新入队。
3. Worker 读取现有 run、业务检查点、阶段尝试和同一 `thread_id`。
4. LangGraph 从 SQLite checkpoint 恢复 Agent；业务层重新注入已批准内容。
5. 已批准阶段跳过生成；发生在外部模型调用中的已记账 attempt 不会被重启“补回”。
6. 如果 thread checkpoint 不存在或损坏，运行失败并保留原因。

恢复关注的是**同一个运行**，而不是“重新提交同一输入创建一个看起来相似的任务”。这也是为什么创建、run、job、thread_id 和 checkpoint 都要持久化。

## 8. 模型上下文预算和调用审计

每次出站模型调用前，audit handler 对实际序列化的 system prompt、消息、tools/schema 和保留输出进行确定性估算，并与角色的已验证上下文上限比较：

```text
estimated_total = serialized_messages + serialized_tools + reserved_output
if limit is missing or estimated_total > limit:
    write preflight_blocked model_call
    do not send provider request
```

估算值和 provider 实际用量严格分开：

- provider 报告 input/output/cache 用量时原样保存；
- provider 没有报告时，状态为 `unavailable` 或 `partial`；
- 不能用本地估算值伪造实际 token 使用量；
- 每个 call 有独立 `call_id`、角色、adapter/provider/model、阶段、集数、候选/batch 血缘、耗时、finish reason 和安全错误。

这套记录同时服务于 UI 用量面板、SQLite `model_calls` 表和结构化日志。迟到、被取代、超时和预检拦截的调用也保留，避免只看“最后一次成功请求”。

## 9. 修订流程

```text
initial succeeded
       │
       ▼
POST /creations/{id}/revision
       │ 首次反馈冻结
       ▼
revision run（新的 thread_id，同一 persona snapshot）
       ├─ succeeded → revision entitlement 永久关闭
       ├─ failed → 只允许相同 feedback 重排队为新的 revision attempt
       ├─ paused → Continue / End
       ├─ quality_rejected → 只重试最终审核
       └─ ended → 永久终止
```

修订是完整工作流，不修改初稿；初稿在修订运行期间仍保持可读。失败修订不消耗“唯一一次修订”资格，但重新排队必须使用 byte-for-byte 相同的冻结反馈。
