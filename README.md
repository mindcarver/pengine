<p align="center">
  <img src="docs/assets/pengine-retro-workbench.jpg" alt="Pengine 复古创作工作台：人格档案、创作终端、四个专业 Agent 与成片胶卷" width="100%">
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

Pengine V1 是一个本地短剧创作 Agent 后端。它通过
[Deep Agents](https://github.com/langchain-ai/deepagents) 与 LangGraph，
让四个专业 Agent 按固定阶段协作；业务状态、检查点和交付物全部落在本地
SQLite，模型请求则通过 Anthropic Messages 兼容 relay 发出。

V1 没有前端、身份认证、公共部署、多用户隔离或跨项目可写记忆。

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
                              │ L0–L6 文件 │              │ 4 个专业 Agent │
                              └────────────┘              └───────┬────────┘
                                                                  │
                                                           ┌──────▼──────┐
                                                           │  模型中继   │
                                                           └─────────────┘
```

| 组件 | 负责什么 |
|---|---|
| 接口服务（FastAPI） | 参数校验、幂等命令、状态与结果查询、稳定错误 |
| 人格加载器 | 校验九文件人格包，生成内容寻址的不可变快照 |
| 状态仓库 | 管理创作、运行、任务、反馈、检查点与交付物 |
| 内嵌任务器 | 租约、重启恢复、阶段尝试预算、工作流调度 |
| Agent 编排层 | 由监督 Agent 编排四个同步专业 Agent |
| 模型中继客户端 | 以固定超时和安全错误映射调用模型 |

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
                           └─▶ 分集剧本
                                └─▶ L0 闸门
                                     └─▶ L4 闸门
                                          └─▶ 组装交付
```

四个 Agent 分工明确：

- `story_architect`：L0 选择、故事大纲、人物小传、关系逻辑；
- `episode_planner`：分集大纲；
- `script_writer`：分集剧本；
- `quality_reviewer`：L0/L4 验收证据与修订反馈覆盖。

系统同时保留两类检查点：

- **LangGraph checkpoint**：保存 Agent 线程、消息与临时工作区；
- **Business checkpoint**：保存已批准的阶段结果，是推进状态和组装交付的唯一依据。

每阶段最多尝试三次。只有通过结构校验和 L0/L4 闸门的完整内容包才会公开。

<h2 align="center">04 · 一次创作，一次修订</h2>

1. 创建请求使用调用方生成的 `Idempotency-Key`，返回异步资源地址；
2. 初稿成功后，系统开放一次修订机会；
3. 首次反馈一经接受即冻结；
4. 修订会使用同一人格快照完整重跑工作流，不修改初稿；
5. 失败修订只能用完全相同的反馈重试；成功后修订入口永久关闭。

机器接口以 [`openapi.json`](contracts/openapi.json) 为准。

<h2 align="center">05 · 快速启动</h2>

要求：Python `3.12`、[`uv`](https://docs.astral.sh/uv/)、一个兼容 Anthropic
Messages 的 relay，以及至少一个人格包。

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
PENGINE_RELAY_BASE_URL=https://your-relay.example
PENGINE_RELAY_API_KEY=replace-with-your-key
PENGINE_RELAY_MODEL_ID=your-model-id
```

API 只允许绑定回环地址。Relay URL 必须使用 HTTPS；只有 `localhost`、
`127.0.0.1` 和 `::1` 可使用 HTTP。密钥不得提交到仓库。

把人格包放入 `PENGINE_PERSONA_ROOT` 的直接子目录，然后启动：

```bash
uv run pengine
```

默认地址：`http://127.0.0.1:8000`

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

轮询同一个 creation 资源，直到 `initial.state` 或 `revision.state` 进入
`succeeded` / `failed`。

<h2 align="center">07 · 数据与恢复</h2>

```text
PENGINE_DATA_DIR/
├── pengine.sqlite3      # 业务状态 + LangGraph checkpoints
└── persona-snapshots/   # 不可变人格快照
```

进程中断后，使用同一数据目录重启即可。过期租约会重新入队；已批准的业务检查点
不会重新生成；现有 LangGraph `thread_id` 会继续用于恢复。

数据不会自动过期。故事、反馈、生成内容和备份都应按敏感内容保护。

<h2 align="center">08 · 验证与边界</h2>

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv build --no-sources
```

自动化测试使用非生产人格与确定性假模型，只证明工程行为。真实 relay 的工具调用、
结构化输出，以及不同真实人格是否产生可辨识差异，仍需单独 UAT。

更完整的设计依据见 [`.scd/architecture.md`](.scd/architecture.md)；产品边界见
[Issue #1](https://github.com/mindcarver/pengine/issues/1)。
