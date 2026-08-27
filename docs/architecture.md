---
layout: default
title: 架构与设计
permalink: /architecture/
---

{% include nav.md %}

# 架构与设计

这页描述 Pengine 的责任边界和不变量。它解释“为什么这样拆”，不重复 OpenAPI 的每一个字段。

## 1. 一句话模型

Pengine 是一个**本地、单进程、单 Worker、SQLite 持久化、双角色模型路由、同源 Web + JSON API** 的模块化单体。

```text
输入：persona_id + story + requirements
  │
  ├─ 人格包校验 → 内容寻址快照
  ├─ Agent 有序生成 → 结构化候选
  ├─ 业务层确定性校验 → 独立审核 → 版本化提交
  ├─ 剧情合同 + SeriesBible → 逐集 script batch
  ├─ L0 / L4 / whole-series gates
  └─ 完整 ContentPackage + DeliveryReport
输出：可查询 creation resource（初稿，最多一个冻结修订）
```

最重要的因果链不是“模型返回了文本”，而是：

```text
候选结果 → schema / hash / 引用 / 顺序 / 状态校验
        → 绑定对应设计和前序版本
        → 业务事务提交为 approved checkpoint
        → 进入下一阶段或最终交付
```

模型可以失败、超时、返回结构错误或提出不一致候选；这些都不能直接改变公开业务状态。

## 2. 组件拓扑

<div class="diagram">

<pre><code>
┌───────────────────────────────┐
│ 浏览器：同源 Web 工作台         │
│ 选择人格、创建、轮询、控制、阅览 │
└──────────────┬────────────────┘
               │ loopback HTTP
┌──────────────▼────────────────┐
│ FastAPI                        │
│ 参数校验 / DomainError / JSON  │
│ 静态资源 / OpenAPI             │
└──────┬─────────────────┬─────┘
       │ 查询/命令         │ 启动/停止
       ▼                  ▼
┌──────────────┐   ┌────────────────────┐
│ Repository   │   │ Embedded Worker    │
│ SQLite 事务  │◀──│ lease / stage guard │
│ 业务权威状态 │   │ invoke / recovery   │
└──────┬───────┘   └─────────┬──────────┘
       │                      │
       │                  ┌───▼─────────────────┐
       │                  │ Deep Agents          │
       │                  │ workflow_supervisor  │
       │                  │ StateBackend         │
       │                  └───┬─────────────────┘
       │                      │ synchronous subagents
       │        ┌─────────────┼────────────────┐
       │        ▼             ▼                ▼
       │  story_architect  episode_planner  script_writer
       │        │             │                │
       │        └─────────────┼────────────────┘
       │                      ▼
       │              quality / canon / series review
       │                      │
       │           ┌──────────▼──────────┐
       │           │ role-bound relay    │
       │           │ generation / review │
       │           └─────────────────────┘
       │
       └─ SQLite also stores model_calls and LangGraph checkpointer tables
</code></pre>

</div>

### 组件责任矩阵

| 组件 | 拥有的责任 | 明确不拥有的责任 |
| --- | --- | --- |
| Web 工作台 | 表单、状态轮询、只读草稿阅览、继续/授权/结束命令 | 不推断阶段、不决定审批、不执行模型 |
| FastAPI | 参数验证、HTTP 序列化、幂等命令、稳定错误、资源查询 | 不编排创作、不解析人格正文、不决定质量 |
| `Repository` | SQLite 事务、创建/运行/任务/检查点/交付/幂等记录、迁移 | 不拥有人格源目录，不代替 Agent 生成内容 |
| `Worker` | 租约、重启协调、阶段尝试预算、Agent 调用、恢复分类 | 不暴露 HTTP 语义，不拥有模型厂商协议细节 |
| `workflow_supervisor` | 在一个 Agent thread 内按顺序委派阶段 | 不直接批准业务检查点，不创建 revision entitlement |
| 同步子 Agent | 产出结构化候选、审核证据或有界修复 | 不直接写数据库、不修改人格、不解锁已批准内容 |
| 人格加载器 | manifest/schema/hash/结构验证、快照、只读投影、L5/L6 有界检索 | 不自动学习或写回人格 |
| Relay 客户端 | 角色到 adapter/provider/model 的绑定、上下文预检、响应身份审计、安全错误映射 | 不决定重试策略，不在角色间回退 |
| LangGraph checkpointer | Agent 线程、消息、计划、虚拟 scratch | 不决定公开状态、阶段次数或交付有效性 |

V1 进程只运行一个 Worker，并一次处理一个创作任务。Deep Agents 和 LangGraph 是嵌入库，不是独立部署的服务；外部消息队列、分布式 Worker、缓存和远程 Agent 都不在当前边界内。

## 3. Agent 拓扑与模型路由

| Agent | 主要工作 | 路由 | 是否带专用 skill |
| --- | --- | --- | --- |
| `workflow_supervisor` | 读取已批准上下文，严格按阶段委派，汇总完成 | generation | 否，只有通用编排能力 |
| `story_architect` | L0 选择、故事大纲、人物小传、关系逻辑 | generation | 否 |
| `episode_planner` | 分集大纲 + `StoryContract` | generation | 否 |
| `script_writer` | 每集完整剧本 + `EpisodeStateDelta` | generation | 否 |
| `quality_reviewer` | L0/L4 闸门证据、修订反馈覆盖 | review | 否 |
| `canon_reviewer` | 故事候选/剧情合同的独立审查 | review | `canon-review` |
| `series_reviewer` | 结构性里程碑和全剧审查，分类为通过/设计缺陷/剧本缺陷 | review | 审核范围由运行时注入 |
| `episode_repair` | 只修复当前未锁定集 | generation | `continuity-repair` |
| `story_repair` | 只修复未锁定的人物/关系候选 | generation | `story-repair` |

角色绑定由运行时建立，不能用环境变量把审核角色切换成生成角色。当前配置合同是：

```text
generation = ChatAnthropic / Anthropic Messages / claude-opus-5 或 claude-sonnet-5
review     = ChatDeepSeek  / deepseek-v4-flash
          | ChatOpenAI    / gpt-5.5 或 gpt-5.6-terra
          | ChatAnthropic / claude-opus-5 或 claude-sonnet-5
```

两个客户端共用 relay URL 和 key，但每个响应还必须回报与该角色配置一致或在显式官方别名映射中的单一模型身份。当前请求 `gpt-5.5` 时允许 `gpt-5.5` 与固定快照 `gpt-5.5-2026-04-23`，不使用前缀或通配符。身份缺失、不在显式集合或出现多个值时，本次响应会被丢弃；所有通过或拒绝的实际 `response_model_ids` 都原样写入 SQLite 与 Langfuse，拒绝时运行暂停为 `relay_identity_mismatch`，核验 Relay 后才能人工继续。

模型调用预算与 LangGraph recursion limit 分开计算。默认每个普通阶段最多保留生成调用
`48` 次、审核调用 `32` 次；剧本阶段另有全剧生成 `192`、审核 `128` 的总上限，同时
仍受单集对应角色上限约束。预算在出站前原子保留，超限不会触发 provider 请求。

## 4. 人格包：从 persona v3 到不可变上下文

一个可选择人格必须包含 `manifest.json` 和以下固定文件：

```text
paradigm.md  project.md  l0.md  soul.md
l3.md        l4.md      l5.md  l6.md
```

### 校验和快照过程

1. `PersonaCatalog.discover()` 扫描人格根目录。
2. 只接受固定文件集合、UTF-8 文本、合法 JSON、符合 `persona-package.schema.json` 的 manifest。
3. 每个 Markdown 文件的声明 SHA-256 必须匹配实际内容，并且标题/状态/归属结构必须通过。
4. 按 schema 对应的固定文件顺序拼接各文件 hash，形成 `package_sha256`。
5. 将规范化 manifest 与 `package_sha256` 放入域分离 hash，形成 `snapshot_sha256`。
6. 创建任务时复制到 `data/persona-snapshots/<snapshot_sha256>/`；任务只引用快照 hash，不引用会变化的源路径。

人格上下文按职责投影：

- `/persona/` 是只读虚拟上下文；
- `project`、完整 L0、完整 Soul、完整 L3 和按阶段选择的 L4 内容进入工作上下文；Project 是权威路由和工作纪律，不是新的事实来源或 Gate；
- 内容生成和修复的实际 provider 请求会从该快照完整、逐字、恰好一次地内联 Project，不依赖模型主动 `read_file`；缺失或空 Project 在出站前失败关闭；
- Supervisor 不带入完整 Project；Reviewer 只接收独立审核边界，不能以人格或 Project 相似度通过或拒绝候选；纯结果序列化重试只有在明确禁止改写内容时才省略全文；
- v3 的 Soul 与 L3 读取同一份完整文本，不摘要、不检索、不切片、不静默截断；L3 只约束创作决策方式，不能重选 L0、覆盖合同或成为 Reviewer Gate；
- L0 选择与 L0 Gate 的 L4 投影只有 L4-A；四个生成阶段的单文件投影固定为 `L4-A + 全阶段通则 + 当前阶段规则 + Pengine 参数`；`accepting_l4` 才读取完整 L4；
- Worker 把七个专业阶段的投影以不同 `/persona/l4/<stage>.md` 路径预装进同一只读 VFS，当前阶段选择由 prompt 约束，不是按子 Agent 隔离的文件 ACL；Story/Outline 两条直接补丁调用不运行该 VFS，也不直接内联 L3/L4 正文，只执行 Reviewer 已确认的有界补丁；
- 只有明确标为创作者确认的 L4 硬规则可成为阻断依据。创作建议只指导生成，题材适配和未经确认的 AI/市场转译不进入 Gate；6 集、时长和场数属于 Pengine 默认产品参数，用户或锁定合同优先；
- 历史 v1/v2 snapshot 继续使用原 L3 摘要以及各自的 L1/L2 或 Soul 路径，但 v1/v2 source package 不再用于新任务；
- L5/L6 走有界检索，受结果数量和字符数限制；
- `/workspace/` 是该 Agent thread 的临时 scratch；
- 不配置 `StoreBackend`，不会形成跨任务可写记忆。

更细的代码路径、阶段矩阵和验收证据见 [L3 实际实现设计]({{ site.baseurl }}/l3-integration/)
与 [L4 实际实现设计]({{ site.baseurl }}/l4-integration/)。两份接入设计当前不展开 L5/L6。

## 5. 剧情合同和 SeriesBible

### `StoryContract` 是什么

分集大纲不只有面向人的 Markdown，还会产生一份版本化剧情合同。当前流程先提交紧凑的
`OutlineSeasonMap`，由大纲决定连续的 1–4 集自然组；每组先生成并锁定 Markdown 正文，
再从该固定正文提取小型连续性 Sidecar。引擎使用 SeasonMap 坐标、Markdown 逐集正文和
Sidecar 确定性构造计划、事实、时间线、线索和分集义务。Sidecar 失败或恢复时复用已锁定
Markdown；每组独立审核、按哈希提交，最终由确定性组装器产生完整合同。
合同是后续写作的机器边界，至少覆盖：

- 角色与关系；
- 类型化事实和单位；
- 时间顺序；
- 角色知道什么、不知道什么；
- 线索生命周期（只在声明悬疑类型时启用相关约束）；
- 每集剧情义务。

合同在写入业务检查点前，必须经过确定性校验和绑定同一候选的 `canon_reviewer`。修复只能产生有界 patch；不能让修复悄悄改变已锁定的上游内容。

数据 schema 只要求 `episode_count >= 1`，并不等于任意集数都已通过生产验收。自然分组已经
移除单次全量结构化输出风险，但 60–100 集仍需分别验证累计连续性账本、模型调用预算、跨组
终审、逐集写作总耗时和恢复行为；没有相应真实隔离证据时，文档不宣称可靠支持。

### SeriesBible 的原子候选

当 L0、故事大纲、人物小传、关系逻辑和分集大纲都具备且合同存在时，系统会把它们组装为一个 `SeriesBible` 候选。候选内的投影、合同、hash 和版本必须属于同一候选，不能让 API/界面看到跨版本拼接的“四件套”。

候选提升为 `active` 需要同时满足：

1. schema、引用、唯一性、顺序和显式算术通过；
2. 该候选自身的类型激活规则通过；
3. 绑定该 `candidate_id` 和内容 hash 的全局设计审核通过；
4. 事务/CAS 提升成功，且没有被更新的 active 指针取代。

人物投影（小传覆盖契约 cast）不再做事后确定性校验：该约束由大纲生成提示词前置承担（小传是整季 cast 权威来源，大纲阶段不得引入无小传角色），生成后纯靠模型理解（#222）。设计缺陷最多自动触发一次完整重建；之后要生成明确的 `repair_authorization`，由操作员授权一次生成+审核周期。迟到、过期或被替代的候选只作为审计证据保留，不能重新移动 active 指针。`relay_unavailable` 与 `stage_validation_failed` 的终态 run 可由操作员经 retry 复活：已 approve 的 checkpoint 保持完整，worker 经重启恢复路径幂等重入。

## 6. 逐集候选和连续性状态

写作阶段不是把所有剧本放进一个可变字符串，而是维护一个绑定设计的 script batch：

```text
SeriesBible(candidate_id, content_hash, design_epoch)
        │
        ▼
ScriptBatch(batch_id, batch_epoch)
        │
        ├─ episode 1 candidate v1 → active pointer 1
        ├─ episode 2 candidate v1 → active pointer 2
        └─ episode N candidate vK → active pointer N
```

每个 `EpisodeCandidate` 绑定：

- 设计候选、设计 epoch、batch 和 batch epoch；
- 集数与候选版本；
- 前一个 active 候选的 hash；
- 生成 `call_id`；该 ID 必须是与同一 `operation_id` 对应的真实成功物理调用；
- 完整剧本内容和内容 hash；
- `EpisodeStateDelta` 与折叠后的 `SeriesState`；
- 语义审查证据和修复轮次。

改写第 N 集时，1..N-1 保持 active；N..末集全部 supersede；状态严格从保留前缀重新折叠，不能把旧后缀的事实、知识、线索或 delta 带回新上下文。设计 epoch 改变时，整个旧 batch 失效并从第 1 集重建。

剧本正文仍是 `ScriptWriterResult.content` 中的自由文本；结构化字段负责集数、状态增量、
证据和审查结论。确定性层只对显式锁定事实和证据目标做机器校验。审核不得把姓名/别名/
职业/泛称等说话人标签、冒号排版、片尾标记、算式，或故事世界中的 JSON、代码、模型、AI
题材本身当成违规；私有运行泄漏必须给出能对应到运行时来源的上下文证据。

## 7. 两套检查点为什么必须分开

| 检查点 | 保存内容 | 谁能使用它推进业务 |
| --- | --- | --- |
| 业务检查点 | 已批准的 L0、故事/人物/关系/分集大纲、剧情合同、逐集锁定内容、交付 | 只有它能推进公开阶段、组装交付和计算完成状态 |
| LangGraph checkpoint | Agent messages、supervisor plan、subagent 结果、虚拟 scratch、`thread_id` | 只用于继续同一个 Agent thread |

如果两者不一致，业务检查点优先；Worker 会把已批准结果重新注入运行上下文，而不是把 Agent thread 中未经批准的文本当成成品。若 thread checkpoint 缺失或不可读，运行安全失败，不会凭空生成“已批准”的状态。

## 8. 安全与信任边界

- API 默认只能绑定 `127.0.0.1`、`localhost` 或其他 loopback 地址。
- Relay URL 必须是 HTTPS；只有 loopback relay 可使用 HTTP。
- API key 使用 `SecretStr`，不应提交到 Git；日志只允许安全错误和脱敏 provider 证据。
- Agent 的虚拟路径与主机文件系统隔离；权限规则只允许需要的只读人格路径和 thread scratch。
- Relay 会收到故事、要求、人格上下文和生成阶段需要的内容；这不是本地隐私隔离的替代物。
- V1 没有身份认证，所以不要把服务绑定到局域网或公网。

## 9. 明确的非目标

以下事项不是“隐藏开关”，而是当前设计边界：

- 公共多租户服务、账号体系、权限模型；
- 自动人格学习、跨任务共享记忆、人格源文件写回；
- Agent 访问本地主机文件、shell 或任意工具；
- 局部片段作为正式交付；
- 静默的模型/供应商 fallback；
- 自动删除、选择性导出或快照垃圾回收；
- 只靠最终审核重试来修复未改变的草稿；
- 通过 `HTTP 200` 或“模型支持 JSON”推断真实 provider 协议兼容。
