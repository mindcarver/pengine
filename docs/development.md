---
layout: default
title: 开发与开源
permalink: /development/
---

{% include nav.md %}

# 开发与开源

这页是贡献者从零开始验证项目的入口。所有命令默认在仓库根目录执行。

## 1. 安装

要求：

- Python `>=3.12,<3.13`；
- [`uv`](https://docs.astral.sh/uv/)；
- 若执行真实模型路径，需要 OpenRouter 或一个能接受所选生成与审核模型协议的 relay；
- 若启用可选 Langfuse tracing，需要 Docker Compose 和本地 Langfuse 服务。

```bash
uv sync --locked --all-groups
cp .env.example .env
```

`.env` 不应提交。默认配置绑定 `127.0.0.1:8000`，数据目录是 `./data`，人格根目录是 `./personas`。

## 2. 配置模型路由

最小真实配置：

```dotenv
PENGINE_PERSONA_ROOT=./personas
PENGINE_DATA_DIR=./data
PENGINE_HOST=127.0.0.1
PENGINE_PORT=8000
PENGINE_RELAY_BASE_URL=https://openrouter.ai/api/v1
PENGINE_RELAY_API_KEY=replace-me
PENGINE_GENERATION_MODEL_ID=deepseek/deepseek-v4-flash
PENGINE_REVIEW_MODEL_ID=deepseek/deepseek-v4-flash
PENGINE_GENERATION_MAX_OUTPUT_TOKENS=128000
PENGINE_REVIEW_MAX_OUTPUT_TOKENS=128000
PENGINE_GENERATION_CONTEXT_LIMIT_TOKENS=1048576
PENGINE_REVIEW_CONTEXT_LIMIT_TOKENS=1048576
PENGINE_STAGE_MODEL_CALL_LIMIT=48
PENGINE_STAGE_REVIEW_CALL_LIMIT=32
PENGINE_SCRIPT_STAGE_MODEL_CALL_TOTAL_LIMIT=192
PENGINE_SCRIPT_STAGE_REVIEW_CALL_TOTAL_LIMIT=128
```

配置规则：

- `.env.example` 默认使用 OpenRouter 的单一模型 `deepseek/deepseek-v4-flash`
  同时承担生成与审核；两个完整 slug（含 `z-ai/glm-5.3-flash`）都允许用于任一角色
  并走 OpenAI-compatible Chat Completions；GLM 保留强制推理，DeepSeek 关闭推理；
- 可选 `PENGINE_OPENROUTER_PROVIDER`（逗号分隔供应商列表）通过 provider.order 让 OpenRouter 大冷 prefill 调用优先走实测快的上游（Issue #285）；回退保持开启，工具兼容性抖动时退回默认路由而非 404；留空保持默认负载均衡；
- 兼容白名单仍保留 `deepseek-v4-flash`、`gpt-5.5`、`gpt-5.6-terra`、
  `claude-opus-5`、`claude-sonnet-5` 及既有 OpenRouter Claude slug；不能仅凭配置通过
  就声称 provider 能力已验证；
- generation/review 上下文上限必须是实际验证过的 token window；没有可信上限时请求前阻断；
- relay URL 必须使用 HTTPS，loopback relay 才允许 HTTP；
- URL、key、任一角色模型缺失时，服务可以启动，但真实模型工作流不会静默降级；
- 旧的单路由变量不能覆盖角色路由：`PENGINE_RELAY_ADAPTER`、`PENGINE_RELAY_MODEL_ID`、`PENGINE_RELAY_MAX_OUTPUT_TOKENS` 不属于当前配置合同；
- stage call limit 是出站模型调用预算，不是 LangGraph recursion limit 或三次业务尝试；
  剧本阶段同时受单集角色上限和整轮 generation/review 总上限约束；
- Langfuse 是可选观测，不改变业务状态权威性：

```dotenv
PENGINE_LANGFUSE_ENABLED=false
PENGINE_LANGFUSE_HOST=http://127.0.0.1:3001
PENGINE_LANGFUSE_PUBLIC_KEY=
PENGINE_LANGFUSE_SECRET_KEY=
```

启动：

```bash
uv run pengine
```

开发期间可以直接访问 `/docs` 查看运行时生成的 OpenAPI UI；仓库中固定的 `contracts/openapi.json` 才是提交和审查时的机器合同。

## 3. 验证分层

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv build --no-sources
```

验证含义不同：

| 检查 | 能证明什么 | 不能证明什么 |
| --- | --- | --- |
| Ruff | Python 静态规则和格式 | provider 协议兼容 |
| 普通 pytest | 确定性业务、API、迁移、恢复和假模型行为 | 真实 relay、真实模型质量 |
| `uv build` | 包和构建产物可生成 | 页面已部署 |
| live E2E | 具体 relay、模型身份、工具调用、结构化输出和真实运行证据 | 所有 provider 或所有人格都能成功 |
| 浏览器验收 | 同源工作台能读取真实 API 投影 | 仅凭 UI 通过就证明后端最终交付 |

真实模型测试是显式 opt-in：见 [`tests/live/README.md`](https://github.com/mindcarver/pengine/blob/main/tests/live/README.md)。它们可能产生费用和敏感创作数据，应在隔离数据目录执行，不能把 live artifact 放进 Git。

## 4. 新增人格包

人格不是随便放几个 Markdown 文件：

1. 复制一个完整 persona v3 八文件目录；
2. 修改 manifest 的身份、版本和每个文件 hash；
3. 重新计算 `package_sha256`；
4. 用 [`contracts/persona-package.schema.json`](https://github.com/mindcarver/pengine/blob/main/contracts/persona-package.schema.json) 校验；
5. 确保必需标题、状态标记和归属信息完整；
6. 确保 `soul.md` 与 `l3.md` 均已确认、各自未超过 8,000 字符，且包内不存在 `l1.md` 或 `l2.md`；
7. L4 必须把创作者确认硬规则、确认建议和 Pengine 产品参数分区；未经确认的 AI/市场转译、长剧体量和题材适配判断不得混入硬 Gate；
8. 原始访谈、Markdown/DOCX 或其他 L9 来源不进入 Git、snapshot、Prompt、SQLite 或观测系统；运行资产只保留已授权编译内容和来源指纹；
9. `project.md` 是运行宪章：修改时必须保留必需标题、更新文件与包 hash，并验证创作/修复请求全文内联、Supervisor/Reviewer 隔离、旧 snapshot 不变；
10. 先运行 `tests/test_personas.py`、`tests/test_bundled_personas.py`、`tests/test_agents.py` 和完整测试；Project 行为的真实模型验证使用 `PENGINE_RUN_PROJECT_ABC=1 uv run pytest -m live_model tests/live/test_project_persona_e2e.py -vv -s`；
11. 不把包含个人隐私、未授权作品或 provider 机密的人格包提交到公开仓库。

创建任务会把有效包复制为内容寻址 snapshot。不要在运行中的任务里替换 snapshot 目录，也不要手工编辑 SQLite 中的 snapshot hash。

L3/L4 的当前实现合同分别记录在 [L3 实际实现设计]({{ site.baseurl }}/l3-integration/) 与
[L4 实际实现设计]({{ site.baseurl }}/l4-integration/)。修改这两层时至少运行：

```bash
uv run pytest tests/test_personas.py tests/test_agents.py tests/test_worker.py \
  tests/test_model_calls.py tests/test_repository.py tests/test_unified_integration.py -q
```

真实模型对照是显式 opt-in，分别使用 `PENGINE_RUN_L3_ABC=1` 和 `PENGINE_RUN_L4_ABC=1`；
接入设计当前不包含 L5/L6，不能顺手扩大为检索或学习闭环改造。

## 5. 修改代码的最小闭环

```text
先找真实运行路径
  → 写能复现行为的测试
  → 只改请求范围内的文件
  → 跑 focused tests
  → 跑 Ruff / 全量 pytest / build
  → 检查 OpenAPI、README 和 Pages 是否仍与代码一致
```

高风险变化要额外检查：

- auth、loopback 绑定和 secrets；
- SQLite schema/migration 与备份恢复；
- retry 分类、模型路由和结构化输出；
- business checkpoint、active pointer 和 stale-write fence；
- 浏览器只读展示是否误报未提交内容为正式交付；
- 物理 `call_id`/`operation_id`/checkpoint `review_call_id` 是否仍能证明产物来源；
- 长篇幅改动是否误把当前一次性全量 `EpisodePlannerResult` 宣称为 60–100 集可靠支持。

## 6. GitHub Pages 发布

站点源文件在 `docs/`，发布工作流在 `.github/workflows/pages.yml`。工作流做三件事：

1. 以 Jekyll 构建 `docs`；
2. 上传 GitHub Pages artifact；
3. 只有 push 到 `main` 或手动触发时部署，Pull Request 只构建不部署。

仓库管理员首次启用时：

1. 打开 GitHub 仓库的 **Settings → Pages**；
2. 将发布源选择为 **GitHub Actions**；
3. 确认 `github-pages` environment 的保护规则；
4. 合并到 `main` 后查看 Actions 的 Pages workflow 输出 URL。

官方流程要求 Pages 部署 job 具备 `pages: write` 与 `id-token: write`，并通过 `github-pages` environment 部署 artifact；本仓库工作流按这一约定配置。[GitHub 官方文档](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)

本地不需要安装 Ruby 才能修改 Markdown。若需要预览 Jekyll，使用与 Pages action 兼容的 Jekyll 环境；提交前至少检查 front matter、相对链接和 workflow YAML。

## 7. 开源边界

代码公开不等于把所有运行数据公开。建议按以下边界审查 PR：

| 可以提交 | 不应提交 |
| --- | --- |
| 源代码、测试、脱敏 fixture、API 合同、架构文档 | `.env`、API key、relay URL 中的凭据 |
| 不含个人数据的示例人格或明确授权内容 | `data/`、SQLite/WAL、persona snapshots |
| 可复现的确定性假模型测试 | 真实用户故事、完整 live prompt/response、provider 账单/密钥 |
| 经过授权的设计素材 | 未授权作品、个人信息、内部评审记录 |

V1 仍是本地单操作员系统；开源仓库中的文档、测试和 Pages 不应被解释为公共部署已经具备认证、隔离或生产数据保护能力。

## 8. 贡献流程

```text
Issue / 设计说明
  → 小范围分支
  → 回归测试与独立审查
  → 更新 contracts/README/Pages（若行为发生变化）
  → PR
  → CI：ruff + pytest + build
  → 合并 main 后自动发布 Pages
```

提交 PR 时请说明：

- 真实修改的运行路径和不变量；
- 新增/修改的测试；
- 是否改变 API、SQLite schema、迁移、模型路由、重试分类或数据边界；
- 哪些验证实际运行过，哪些需要真实 provider 或浏览器人工验收；
- 是否包含需要额外授权的 persona、图片或生成内容。

## 9. 维护检查清单

每次涉及工作流或文档的变更，至少检查：

- [ ] `README.md` 的启动命令和 Pages 链接仍正确；
- [ ] `contracts/openapi.json` 与 `src/pengine/api.py` 路由一致；
- [ ] 内部阶段和公开 `UserStage` 的映射一致；
- [ ] README/Pages 没有把草稿、HTTP 200 或单次模型成功写成最终交付；
- [ ] 恢复说明区分 relay、内容修复、质量拒绝、checkpoint 和配置错误；
- [ ] 新增的敏感 fixture 未进入仓库；
- [ ] CI 与 Pages workflow 的权限最小化；
- [ ] build、测试和实际浏览器验收证据已记录。

## 10. 结构化输出的校验与修复模式（Issue #285 实战经验）

模型的结构化输出只是候选。跨字段约束（角色名/事实 ID 唯一、义务与事实集精确对应、
引用闭合）**无法**靠 JSON Schema、约束解码或 `response_format` 保证——它们只能保证
语法正确；语义约束必须生成后用确定性校验器把关。这一层的工程经验：

- **修复环必须回喂错误原文**（Instructor 模式）。校验失败后"盲重跑"在确定性上下文上
  会复现同一个错误（实测 flash 同错三连败）；把校验器的逐字错误（含字段名）附进重试
  请求，模型一修即过。season map（`generate_outline_season_map`）与大纲组 sidecar
  （`bounded_current_group_repair`）都走这条路径。
- **纯重复先确定性去重**。同 ID 且全量内容一致的重复条目（如同一 fact 在组内列了两
  次）是语义空操作，用 `drop_identical_group_registrations` 直接丢弃（并只在 fact
  完全无剩余拷贝时才清理义务引用），不要浪费有界修复轮次。同 ID 不同内容是真冲突，
  保留给错误回喂环处理。
- **错误信息要透传**。把底层 `ValidationError` 的字段细节吞成通用文案（如
  "未通过确定性校验"）会让运维与修复环都失去目标；`safe_message` 与日志都应携带
  `{exc}` 原文（AgentProtocolError / internal_error 均适用）。
- **传输层隐形重试**。交互式 coding agent（Codex `stream_max_retries` 默认 5）对
  "首 token 前死亡"的流式调用做带退避的整请求透明重发，用户只看到转圈——这是它们
  "怎么接中转都不中断"的核心。pengine 在 `_SerialChatOpenAI._astream` 内实现了等价
  层（`PENGINE_STREAM_MAX_RETRIES`，默认 2）：仅重试未交付任何输出的失败（stall /
  零 chunk 干净 EOF），部分输出已到达消费者后绝不重试，业务层确定性不受影响、也不
  占阶段尝试预算。上游侧参照：DeepSeek 官方 API 以 keep-alive 注释 + 30 分钟宽限
  维持慢流，聚合路由器的处理窗则短得多——慢流对哑代理中转通常无害。
- **常热前缀（Claude Code 模式）**。交互式 agent 的会话前缀永远命中缓存，因此
  prefill 秒回、永远远离聚合路由器的 ~300 秒处理上限。批量管线在重调用（season map）
  前显式发一个同前缀、最小输出的预热请求（`warm_prompt_cache`，
  `PENGINE_PROMPT_CACHE_WARMUP`）复刻同一条件——注意只有具备隐式缓存亲和的上游有效
  （DeepSeek flash slug 上实测 Alibaba/SiliconFlow 命中 12288/12547 tokens，Novita/
  DeepInfra 不命中），供应商偏好需与之配对。
- **模型档位决定 schema 纪律**。flash 档在 9 人 cast 的转录任务上反复违反唯一性约束，
  pro 档一次通过；重结构化阶段考虑用高档模型（分阶段路由），轻阶段用 flash 控制成本。

失败分类速查：`structured_output_invalid`（带错误原文，可反馈修复）→
`content_rejected`（有界修复耗尽，可 continue）→ `attempts_exhausted`（阶段预算耗尽，
终态）；`internal_error` 意味着异常逃逸出修复环，属于缺陷，应修环而不是调模型。
