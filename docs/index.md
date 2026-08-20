---
layout: default
title: Pengine 工程文档
permalink: /
---

{% include nav.md %}

<p><span class="label">V1 · LOCAL-FIRST</span> <span class="label">DESIGN DOCS</span></p>

# Pengine

## 本地短剧创作 Agent 的工程说明

Pengine 把一个自由表达的故事想法、短剧要求和一份版本化人格包，变成一套经过结构化审查、连续性验证和最终闸门验收的短剧交付。它不是一个“调用一次模型再打印文本”的脚本，而是一个带有持久状态、可恢复长任务、内容证据和版本边界的本地模块化单体。

这套站点回答六个问题：

1. 系统由哪些组件组成，谁拥有哪一种状态？
2. 一个创作请求如何从 HTTP 命令走到 Worker、Agent、模型 relay 和 SQLite？
3. 剧情合同、SeriesBible、逐集状态和最终交付为什么要分开？
4. 任务中断、模型超时、内容不通过或重启时，哪些东西会保留、哪些东西会重试？
5. 公开 HTTP API 的请求、状态、错误和幂等规则是什么？
6. 怎样本地运行、测试、提交贡献，并把这份文档发布为 GitHub Pages？

## 先读哪一页

| 目标 | 页面 | 你会看到什么 |
| --- | --- | --- |
| 了解整体设计 | [架构与设计](architecture/) | 组件边界、Agent 拓扑、人格包、SeriesBible、逐集候选与安全边界 |
| 了解一次运行 | [运行流程与恢复](runtime/) | 从 `POST /creations` 到交付、状态机、重启、超时、修复和修订 |
| 追踪 L3 实现 | [L3 实际实现设计](l3-integration/) | 完整挂载、阶段方法、直接补丁边界、安全审计和验收 |
| 追踪 L4 实现 | [L4 实际实现设计](l4-integration/) | 阶段投影、硬规则/建议/参数、Reviewer、最终 Gate 和持久化 |
| 接入本地服务 | [HTTP API](api/) | 10 个公开端点、请求示例、幂等键、状态和稳定错误 |
| 研究持久化 | [数据与一致性](data-model/) | SQLite 逻辑表、哈希、检查点、候选提升、审计和迁移 |
| 参与开源 | [开发与开源](development/) | 安装、环境变量、测试、Pages 发布、贡献边界和敏感数据规则 |

## 设计摘要

<div class="diagram">

<pre><code>
操作员 / 同源 Web 工作台
          │  HTTP + Idempotency-Key
          ▼
      FastAPI API  ──────────────── 查询/命令 ───────────────┐
          │                                                  │
          │ 入队                                               │
          ▼                                                  ▼
   SQLite 业务状态  ◀──────────  Embedded Worker ─────── LangGraph checkpoint
   creation / run / job             │                         │
   checkpoint / delivery             │                         └─ thread / scratch
   model_calls / review              │
                                     ▼
                              Deep Agents supervisor
                                     │ 有序委派
          ┌──────────────────────────┼──────────────────────────┐
          ▼                          ▼                          ▼
    生成路由                     审核路由                    内容闸门
  ChatAnthropic        DeepSeek / OpenAI / Anthropic    L0 / L4 / final
  claude-opus-5          由 review_model_id 选择
          │                          │
          └────────────── 同一 relay URL / key ──────────────┘
</code></pre>

</div>

核心原则是：**模型提出候选，业务层验证和提交；SQLite 的业务检查点决定公开进度，Agent checkpoint 只负责继续 Agent 线程。**

## 当前实现边界

<div class="callout warning">
<strong>这不是公共 SaaS。</strong> V1 只绑定回环地址，面向单个本地操作员；没有认证、多用户隔离、公共部署、流式传输、远程队列、跨创作可写记忆、任务列表/导出/删除 API，也不会把主机文件系统暴露给 Agent。
</div>

模型 relay 共享一组 URL 和 API key，但角色路由固定分离：生成和创作修复只使用
Anthropic Messages 的 `claude-opus-5`；审核模型允许 `deepseek-v4-flash`、`gpt-5.5`、
`gpt-5.6-terra` 或 `claude-opus-5`，并分别选择 DeepSeek、OpenAI 或 Anthropic 客户端。
两类角色不互换，也不在一路失败时自动回退到另一路。

当前仓库内置的四套人格包是可运行的原型数据，不等于生产人格定稿。人格源文件只读；创建任务会建立不可变快照，后续修改源目录不会改变已经创建的任务。

<div class="callout warning">
<strong>长篇幅边界：</strong>分集大纲已按自然组拆分为 Markdown 正文和独立 Sidecar，
不再依赖一次全量结构化结果。这降低了单组格式失败的损失，但不代表 60–100 集的累计上下文、
调用预算、跨组一致性和总耗时已经完成生产验收。
</div>

## 最短启动路径

环境要求：Python `3.12`、[`uv`](https://docs.astral.sh/uv/)，以及同时支持 Anthropic
Messages 生成路由和所选审核模型协议的 relay。

```bash
uv sync --locked --all-groups
cp .env.example .env
# 编辑 .env，至少填写 relay URL、API key、两个角色模型和已验证上下文上限
uv run pengine
```

启动后：

- Web 工作台：[`http://127.0.0.1:8000`](http://127.0.0.1:8000)
- FastAPI Swagger：[`http://127.0.0.1:8000/docs`](http://127.0.0.1:8000/docs)
- 机器合同：[contracts/openapi.json](https://github.com/mindcarver/pengine/blob/main/contracts/openapi.json)
- 人格合同：[contracts/persona-package.schema.json](https://github.com/mindcarver/pengine/blob/main/contracts/persona-package.schema.json)

如果没有真实 relay 配置，服务可以启动和运行确定性测试，但真实创作请求必须 fail closed；不能把“服务能打开”当成“模型能力已经验证”。

## 一次成功运行的判定

只有下面的证据同时成立，运行才会公开为 `succeeded`：

- 选定的人格包已经校验并快照化；
- 故事大纲、人物/关系和分集大纲已形成同一份有效设计候选；
- 剧情合同通过确定性校验和绑定的独立审核；
- 每一集都绑定合同、前序剧本和折叠后的 `SeriesState`，并通过确定性合同/状态检查；
- 声明的结构里程碑与最终完整前缀通过绑定的系列级语义审查；
- 完整剧本批次通过绑定的全剧结构审核；
- `quality_reviewer` 给出 L0 和 L4 的通过证据；
- 最终交付包、交付报告和运行成功状态在一个业务提交边界内落盘；
- 模型调用的角色、模型身份、上下文预检、实际用量或 `unavailable` 状态都留下可查询记录。
- 锁定合同、分集候选和结构审核都绑定到真实成功的物理 `call_id`，而不是合成来源。

这意味着：HTTP `202`、进度增长、单个草稿产生或某次模型请求返回 `200`，都只能证明阶段性进展，不能单独证明成品交付。

## 证据与源代码索引

| 事实 | 当前权威来源 |
| --- | --- |
| HTTP 路径、请求/响应 schema、错误 | [`contracts/openapi.json`](https://github.com/mindcarver/pengine/blob/main/contracts/openapi.json) |
| 人格 manifest、v1/v2/v3 文件集合与 hash | [`contracts/persona-package.schema.json`](https://github.com/mindcarver/pengine/blob/main/contracts/persona-package.schema.json) 与 `src/pengine/personas.py` |
| HTTP 适配与状态查询 | `src/pengine/api.py` |
| 业务状态、事务、幂等、迁移 | `src/pengine/repository.py` |
| 租约、恢复、阶段推进和交付提交 | `src/pengine/worker.py` |
| Agent 拓扑、结构化结果和技能范围 | `src/pengine/agents.py` |
| 剧情合同、系列状态和连续性规则 | `src/pengine/continuity.py`、`src/pengine/series_bible.py` |
| 模型预检、用量、relay 身份审计 | `src/pengine/model_calls.py`、`src/pengine/relay.py` |
| 可重复行为和回归边界 | `tests/` |

更完整的设计决策仍保留在仓库的 [`.scd/architecture.md`](https://github.com/mindcarver/pengine/blob/main/.scd/architecture.md)；这套 Pages 负责把它整理成面向贡献者和使用者的可导航说明。

## 文档阅读约定

- “已实现”只指当前代码和测试能支撑的行为；“未来”或“非 V1”是明确的设计边界，不是已有功能。
- `creation` 是作品资源；`run` 是一次初稿或修订执行；`job` 是 Worker 的可租约任务；`thread_id` 是 LangGraph Agent 线程。四者不能混为一谈。
- “候选”表示模型或修复阶段产生的待验证内容；“已批准检查点”才有资格推进业务状态；“交付”只代表最终闸门之后的完整包。
- 页面中的字段名以 Python schema 和 OpenAPI 为准；如果文档描述与合同冲突，应先修正文档或代码中的不一致，而不是靠调用方猜测。
