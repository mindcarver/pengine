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
- 若执行真实模型路径，需要一个能同时接受 Anthropic Messages 生成路由和所选审核
  模型协议的 relay；
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
PENGINE_RELAY_BASE_URL=https://relay.example/v1
PENGINE_RELAY_API_KEY=replace-me
PENGINE_GENERATION_MODEL_ID=claude-opus-5
PENGINE_REVIEW_MODEL_ID=deepseek-v4-flash
PENGINE_GENERATION_MAX_OUTPUT_TOKENS=128000
PENGINE_GENERATION_CONTEXT_LIMIT_TOKENS=200000
PENGINE_REVIEW_CONTEXT_LIMIT_TOKENS=64000
PENGINE_STAGE_MODEL_CALL_LIMIT=48
PENGINE_STAGE_REVIEW_CALL_LIMIT=32
PENGINE_SCRIPT_STAGE_MODEL_CALL_TOTAL_LIMIT=192
PENGINE_SCRIPT_STAGE_REVIEW_CALL_TOTAL_LIMIT=128
```

配置规则：

- generation model 必须是 `claude-opus-5`，生成输出上限不能超过 `128000`；
- `.env.example` 选择 `deepseek-v4-flash`；Settings 允许的审核模型完整集合是
  `deepseek-v4-flash`、`gpt-5.5`、`gpt-5.6-terra` 和 `claude-opus-5`，分别走
  DeepSeek、OpenAI 和 Anthropic 客户端；不能仅凭配置通过就声称 provider 能力已验证；
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

1. 复制一个完整 persona v2 八文件目录；
2. 修改 manifest 的身份、版本和每个文件 hash；
3. 重新计算 `package_sha256`；
4. 用 [`contracts/persona-package.schema.json`](https://github.com/mindcarver/pengine/blob/main/contracts/persona-package.schema.json) 校验；
5. 确保必需标题、状态标记和归属信息完整；
6. 确保 `soul.md` 已确认、未超过 8,000 字符，且包内不存在 `l1.md` 或 `l2.md`；
7. 先运行 `tests/test_personas.py`、`tests/test_bundled_personas.py` 和完整测试；
8. 不把包含个人隐私、未授权作品或 provider 机密的人格包提交到公开仓库。

创建任务会把有效包复制为内容寻址 snapshot。不要在运行中的任务里替换 snapshot 目录，也不要手工编辑 SQLite 中的 snapshot hash。

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
