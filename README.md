<p align="center">
  <img src="docs/assets/pengine-retro-workbench.jpg" alt="Pengine 复古创作工作台：人格档案、创作终端、专业 Agent 与成片胶卷" width="100%">
</p>

<h1 align="center">PENGINE</h1>

<p align="center">
  <strong>本地短剧创作引擎 · 第一版</strong><br>
  把一套可验证的编剧人格、一个故事和创作要求，变成完整短剧交付。
</p>

<table align="center">
  <tr>
    <td align="center"><strong>运行</strong><br>本地单操作员</td>
    <td align="center"><strong>接口</strong><br>仅限回环地址</td>
    <td align="center"><strong>人格</strong><br>不可变快照</td>
    <td align="center"><strong>修订</strong><br>一次提交后冻结</td>
  </tr>
</table>

Pengine V1 是一个带同源 Web 原型的本地短剧创作 Agent。它通过
[Deep Agents](https://github.com/langchain-ai/deepagents) 与 LangGraph，
让阶段化专业 Agent 按固定流程协作；业务状态、检查点和交付物全部落在本地
SQLite。模型请求共用一组 relay URL 与密钥，但按角色固定为两条路由：生成与创作修复
使用 Anthropic Messages 的 `claude-opus-5`，审核使用 DeepSeek OpenAI-compatible 的
`deepseek-v4-flash`。

Web 原型只服务本机单操作员，并以约 1.8 秒轮询展示后端确认的七阶段进度、
已运行时长和超时恢复操作；V1 没有身份认证、公共部署、多用户隔离、SSE /
WebSocket 或跨项目可写记忆。

<h2 align="center">01 · 系统架构</h2>

```text
┌──────────────┐     HTTP / 127.0.0.1      ┌─────────────────┐
│    操作员     │ ─────────────────────────▶│  FastAPI 接口   │
└──────────────┘                           └────────┬────────┘
                                                  │ 命令 / 查询
                    ┌─────────────────────────────▼──────────────┐
                    │             SQLite 状态库                   │
                    │ 创作 · 运行 · 任务 · 闸门 · 交付            │
                    │ 业务检查点 · LangGraph 检查点               │
                    └───────────────┬─────────────────────────────┘
                                    │ 持久任务租约
┌──────────────┐   校验 / 快照  ┌─────▼──────────┐   调用    ┌──────────────┐
│   人格目录    │ ────────────▶│   内嵌 Worker  │ ─────────▶│ Deep Agents  │
└──────────────┘               └─────┬──────────┘           │ 监督 Agent   │
                                    │                       └──────┬───────┘
                                    │ 只读上下文                    │
                              ┌─────▼──────┐              ┌───────▼────────┐
                              │ L0–L6 文件 │              │ 创作与审查 Agent│
                              └────────────┘              └───────┬────────┘
                                                                  │
                                                           ┌──────▼──────┐
                                                           │ 双协议模型中继│
                                                           └─────────────┘
```

| 组件 | 负责什么 |
|---|---|
| 同源 Web 工作台 | 七阶段进度、初稿/修订结果和暂停任务操作 |
| 接口服务（FastAPI） | 参数校验、幂等命令、状态、恢复操作与结果查询 |
| 人格加载器 | 校验九文件人格包，生成内容寻址的不可变快照 |
| 状态仓库 | 管理创作、运行、任务、反馈、检查点与交付物 |
| 内嵌任务器 | 租约、重启恢复、阶段尝试预算、工作流调度 |
| Agent 编排层 | 由监督 Agent 编排同步创作 Agent 与技能化审查/修复子代理 |
| 模型中继客户端 | 共用 URL/密钥，按角色固定调用 Opus 生成路由与 DeepSeek 审核路由 |

系统采用模块化单体：一个进程、一个 Worker、一次处理一个创作任务，不依赖外部
消息队列或远程 Agent。

<h2 align="center">02 · 九文件人格</h2>

每个人格目录必须包含一个 `manifest.json` 和九个 UTF-8 Markdown 文件：

```text
persona/
├── manifest.json    # 身份、版本、文件哈希
├── paradigm.md      # 总纲：层级定义、仲裁、纪律
├── project.md       # 当前人格的完整工作说明
├── l0.md            # 内核：变体、雷区、温度
├── l1.md            # 来源画像
├── l2.md            # 第二套画像坐标
├── l3.md            # 创作方法与认知路径
├── l4.md            # 价值观与短剧技艺
├── l5.md            # 作品与经历
└── l6.md            # 外部技法条目
```

源文件始终只读。创建任务时，Pengine 会复制并验证人格包：

- `package_sha256` 标识九个 Markdown 内容；
- `snapshot_sha256` 同时纳入规范化 manifest 身份；
- 旧任务永远引用原快照，新版本只影响新任务；
- L5/L6 只按需、限量检索，不整文件注入模型上下文。

规范见
[`persona-package.schema.json`](contracts/persona-package.schema.json)。

<h2 align="center">03 · 创作流水线</h2>

```text
载入人格
  └─▶ 选择 L0
       └─▶ 故事大纲
            └─▶ 人物小传
                 └─▶ 关系逻辑
                      └─▶ 分集大纲
                           └─▶ 剧情合同双检并锁定
                                └─▶ 分集剧本
                                └─▶ L0 闸门
                                     └─▶ L4 闸门
                                          └─▶ 组装交付
```

四个主创 Agent 分工明确：

- `story_architect`：L0 选择、故事大纲、人物小传、关系逻辑；
- `episode_planner`：分集大纲；
- `script_writer`：分集剧本；
- `quality_reviewer`：L0/L4 验收证据与修订反馈覆盖。

`workflow_supervisor`、三个创作 Agent、故事／分集大纲补丁生成和 `episode_repair`
固定使用生成路由；`quality_reviewer`、`canon_reviewer` 与 `episode_reviewer` 固定使用
审核路由。两个角色不互换，也不会在一路失败时回退到另一路。

分集大纲批准前，`episode_planner` 同步产出结构化、带版本的剧情合同。
确定性校验与加载 `canon-review` skill 的独立审查子代理均通过后，合同及其
SHA-256 才会写入业务检查点并锁定。后续每集只读取该合同、上一集折叠后的
`series_state` 和已锁剧本；`episode-continuity-review` 独立审查通过后，剧本、
`episode_state_delta`、新状态及其哈希才会原子提交。Skill 只加载到对应审查或
修复子代理，不作为监督 Agent 的全局提示。

系统同时保留两类检查点：

- **LangGraph checkpoint**：保存 Agent 线程、消息与临时工作区；
- **Business checkpoint**：保存已批准的阶段结果，是推进状态和组装交付的唯一依据。

普通生成阶段最多尝试三次。合同或单集内容审查最多修复两轮；仍不通过时，系统
保留具体证据并暂停，让操作员继续重生当前未锁内容或结束任务。Relay／网络恢复
次数与内容修复次数分别记录，互不消耗。只有全部分集锁定、聚合哈希复验通过，
且 L0/L4 闸门通过的完整内容包才会公开。

<h2 align="center">04 · 一次创作，一次修订</h2>

1. 创建请求使用调用方生成的 `Idempotency-Key`，返回异步资源地址；
2. 初稿成功后，系统开放一次修订机会；
3. 首次反馈一经接受即冻结；
4. 修订会使用同一人格快照完整重跑工作流，不修改初稿；
5. 失败修订只能用完全相同的反馈重试；成功后修订入口永久关闭。

机器接口以 [`openapi.json`](contracts/openapi.json) 为准。

<h2 align="center">05 · 快速启动</h2>

要求：Python `3.12`、[`uv`](https://docs.astral.sh/uv/)，以及一个能以同一组 URL／密钥
同时提供 Anthropic Messages 与 DeepSeek OpenAI-compatible chat completions 的 relay。
仓库已内置四套临时原型人格包。
这四套人格当前统一采用 6 集原型基线，不代表创作者人格定稿。

```bash
uv sync --locked --all-groups
cp .env.example .env
```

编辑 `.env`：

```dotenv
PENGINE_PERSONA_ROOT=./personas
PENGINE_DATA_DIR=./data
PENGINE_HOST=127.0.0.1
PENGINE_PORT=8000
PENGINE_RELAY_BASE_URL=https://your-relay.example/v1
PENGINE_RELAY_API_KEY=replace-with-your-key
PENGINE_GENERATION_MODEL_ID=claude-opus-5
PENGINE_REVIEW_MODEL_ID=deepseek-v4-flash
PENGINE_GENERATION_MAX_OUTPUT_TOKENS=128000
# PENGINE_REVIEW_MAX_OUTPUT_TOKENS=...
# 已验证的上下文窗口（tokens）。预检会把完整序列化请求 + 保留输出与此上限比较；
# 未设置时 fail closed，任何真实模型请求都不会发出。
PENGINE_GENERATION_CONTEXT_LIMIT_TOKENS=200000
PENGINE_REVIEW_CONTEXT_LIMIT_TOKENS=64000
```

API 只允许绑定回环地址。Relay URL 必须使用 HTTPS；只有 `localhost`、
`127.0.0.1` 和 `::1` 可使用 HTTP。`PENGINE_RELAY_BASE_URL` 与
`PENGINE_RELAY_API_KEY` 同时交给两个客户端；该地址必须同时接受 Anthropic Messages
和 OpenAI-compatible 请求。`PENGINE_GENERATION_MODEL_ID` 必须是 `claude-opus-5`，
`PENGINE_REVIEW_MODEL_ID` 必须是 `deepseek-v4-flash`。URL、密钥或任一模型 ID 缺失时，
工作流会 fail closed，不会降级成单模型，也不会跨角色回退。

生成路由通过 `ChatAnthropic` 调用 Anthropic Messages；
`PENGINE_GENERATION_MAX_OUTPUT_TOKENS` 默认使用 Opus 5 支持的最大值 128000，且不能
配置为更大的值。审核路由通过原生
`ChatDeepSeek` 调用 OpenAI-compatible API，关闭 thinking 并串行调用工具；未设置
`PENGINE_REVIEW_MAX_OUTPUT_TOKENS` 时 Pengine 不额外添加输出上限。旧的
`PENGINE_RELAY_ADAPTER`、`PENGINE_RELAY_MODEL_ID` 和
`PENGINE_RELAY_MAX_OUTPUT_TOKENS` 已不再生效，设置它们不能配置或覆盖任一路由。
每次响应还必须报告与角色配置一致的模型 ID，否则按协议不兼容失败。密钥不得提交到仓库。

### 模型上下文预算与用量观测

每次真实模型请求发出前，Pengine 都会把**实际序列化**的 system prompt、messages、
tools/schema 与完整规范上下文，加上该路由的**保留输出**，估算为 token 数并与该路由
**已验证的上下文上限**（`PENGINE_GENERATION_CONTEXT_LIMIT_TOKENS` /
`PENGINE_REVIEW_CONTEXT_LIMIT_TOKENS`）比较。超出上限、或该路由没有可信的已验证上限时，
请求**不会发出**，任务会安全暂停（`context_budget`），已批准内容与已提交草稿保持不变。
估计值与 provider 实际用量是两个独立字段：provider 报告 input/output/cache 用量时按原值
持久化；缺失时显示为 `unavailable`，绝不从估计值回填。

每次尝试的调用（生成、审核、修复、被预检拦截的尝试）都会以唯一的 `call_id` 记录角色、
adapter/provider/model、阶段、分集、候选与批次血缘，以及估计值、实际或不可用用量、
耗时、结束原因与结果，并同时写入结构化日志、SQLite 的 `model_calls` 表、创作资源
（`RunProgress.model_calls`）与工作台用量面板；失败、超时、被取代、过期与被拦截的调用
都保留各自的分类并计入本轮与整轮合计。

确认 `PENGINE_PERSONA_ROOT=./personas` 后启动：

```bash
uv run pengine
```

原型界面：`http://127.0.0.1:8000`

交互文档：`http://127.0.0.1:8000/docs`

<h2 align="center">06 · 最短接口路径</h2>

```bash
# 1. 查看可用人格
curl --fail-with-body http://127.0.0.1:8000/personas

# 2. 创建完整短剧
curl --fail-with-body -X POST http://127.0.0.1:8000/creations \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: creation-001' \
  -d '{"persona_id":"your-persona-id","story":"一个人回乡面对旧事。","requirements":"生成完整短剧。"}'

# 3. 查询状态与结果
curl --fail-with-body \
  http://127.0.0.1:8000/creations/REPLACE_WITH_CREATION_ID

# 4. 初稿成功后提交唯一一次修订
curl --fail-with-body -X POST \
  http://127.0.0.1:8000/creations/REPLACE_WITH_CREATION_ID/revision \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: revision-001' \
  -d '{"feedback":"让结尾的代价更明确，同时保留原有情绪底色。"}'
```

轮询同一个 creation 资源。运行状态会依次使用 `queued`、`running`，首次整体
超时或可恢复的 relay／网络中断进入 `auto_resuming`；同一阶段第二次共享中断进入
`paused`。可恢复中断仅包括请求开始后的临时连接、DNS、TLS、读取超时或重置，以及 relay 的
429／502／503／504；首次 relay 自动恢复至少等待 10 秒，并遵从更长的 `Retry-After`。
启动时可验证的 relay 配置错误、认证、参数或协议错误会立即终止；证书校验失败也会终止。
语法正确的地址发生 DNS／连接失败时，系统无法可靠区分短暂网络故障与错误主机名，因此按上述
可恢复路径计入三次调用上限，耗尽后终止，不会无限重试。
成品审核未通过会进入
`quality_rejected`：已提交的工作区和审核证据会保留，可只重试 L0／L4 审核（每关最多
三次），不会重跑前面的创作阶段。终态为 `succeeded`、`failed` 或用户主动选择的
`ended`。

剧情合同或单集连续性在两轮修复后仍未通过会进入 `paused`，恢复原因是
`content_rejected`。界面显示独立审查证据；继续只会重新生成当前未锁内容，已经
锁定的合同和分集不会静默改变。

暂停后可用同一套幂等命令继续或结束初稿／修订（把 `initial` 换成 `revision`
即可操作修订）：

```bash
curl --fail-with-body -X POST \
  http://127.0.0.1:8000/creations/REPLACE_WITH_CREATION_ID/runs/initial/continue \
  -H 'Idempotency-Key: continue-001'

curl --fail-with-body -X POST \
  http://127.0.0.1:8000/creations/REPLACE_WITH_CREATION_ID/runs/initial/retry-final-review \
  -H 'Idempotency-Key: final-review-001'

curl --fail-with-body -X POST \
  http://127.0.0.1:8000/creations/REPLACE_WITH_CREATION_ID/runs/initial/end \
  -H 'Idempotency-Key: end-001'
```

<h2 align="center">07 · 数据与恢复</h2>

```text
PENGINE_DATA_DIR/
├── pengine.sqlite3      # 业务状态 + LangGraph checkpoints
└── persona-snapshots/   # 不可变人格快照
```

进程中断后，使用同一数据目录重启即可。过期租约会重新入队；已批准的业务检查点
不会重新生成；现有 LangGraph `thread_id` 会继续用于恢复。首次整体墙钟超时或限定的
relay／网络中断也会按同一原则自动继续；同一用户阶段或首个未完成分集的第二次共享
中断时，SQLite 会冻结时长并等待操作员继续或结束。请求前可验证的配置、证书校验、
鉴权、参数／协议不兼容、结构化输出、检查点缺失、图递归、质量拒绝和未知错误仍是终态
失败，不会伪装成可恢复中断；语法正确地址的 DNS／连接失败则受三次调用上限约束。
剧情合同和每集状态增量也持久化在 SQLite；汇总剧本检查点必须带有合同哈希、每集
内容哈希和连续状态哈希，缺失或冲突时不能进入 L4。

数据不会自动过期。故事、反馈、生成内容和备份都应按敏感内容保护。

<h2 align="center">08 · 验证与边界</h2>

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv build --no-sources
```

自动化测试使用非生产人格与确定性假模型，只证明工程行为。真实双路 relay 的协议、
工具调用、结构化输出、角色／模型身份审计，以及不同真实人格是否产生可辨识差异，仍需
按 [`tests/live/README.md`](tests/live/README.md) 单独验收。

更完整的设计依据见 [`.scd/architecture.md`](.scd/architecture.md)；后端边界见
[Issue #1](https://github.com/mindcarver/pengine/issues/1)，可运行前端原型见
[Issue #9](https://github.com/mindcarver/pengine/issues/9)，长任务进度与恢复见
[Issue #10](https://github.com/mindcarver/pengine/issues/10)。
[Issue #37](https://github.com/mindcarver/pengine/issues/37) 记录剧情合同硬锁与逐集连续性
审查的验收契约。
