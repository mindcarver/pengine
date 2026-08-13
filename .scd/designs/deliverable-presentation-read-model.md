---
managed_by: scd-architecture
status: ready
sources:
  - ../architecture.md
  - ../ux/deliverables-reading-room.md
  - ../../contracts/openapi.json
---

# 结果与边界

Pengine 增加一个独立、只读、可降级的成品展示投影，为故事大纲、人物小传、人物关系、分集大纲和分集剧本提供稳定导航单位。现有正式 `ContentPackage` 五个原文字符串继续是交付权威；展示投影不改写、摘要、补全或重新审核内容，也不参与运行完成判定。

本设计只支持已成功的 initial 或 revision run。读取展示投影不会启动模型、创建任务、修复内容、推进阶段、冻结反馈或修改 SQLite。创作中仍使用现有 `RunDraftSnapshot`。

HTTP 语法由 canonical `contracts/openapi.json` 唯一定义。本文只记录职责、数据流、验证、兼容性和取舍，不复制字段表。

# 现有上下文

- `GET /creations/{creation_id}` 同时承担运行轮询和最终结果读取；前端当前每 1.8 秒轮询该资源。
- `ContentPackage` 固定包含五个非空自由文本字段，正式原文保存在 `deliveries.content_package_json`。
- 故事大纲、人物小传和人物关系的生成结果目前只有整段文本；没有可供 UI 可靠消费的章节、人物或关系列表。
- 分集大纲同时有完整文本和持久化 `episode_plans`；分集剧本同时有完整聚合文本和持久化 `episode_drafts`。
- 创作中的分集剧本已经按 `episode_number` 导航；正式成品却只读取聚合字符串。
- 当前输出允许不同题材、语言内表达和剧本格式，不能用作品、人格、中文序号或固定标题正则推断结构。

# 领域与职责变更

## 权威数据与投影

| 事实 | 所有者 | 作用 |
| --- | --- | --- |
| 正式五类原文 | `deliveries.content_package_json` | 交付权威、完整原文回退 |
| 已批准分集计划与逐集剧本 | business checkpoint、`episode_plans`、`episode_drafts` | 阶段权威、生成分集展示项的证据 |
| 展示 manifest | `deliveries` 的可空展示列 | 只读导航投影，不推进业务状态 |
| 当前文稿、当前人物/关系/集、滚动位置 | 浏览器 | 单客户端 UI 状态，不回写服务端 |

展示 manifest 必须绑定 run、schema version 和五类原文 SHA-256。任一文稿的绑定或项目验证失败，只降级该文稿；不得把展示损坏解释为交付失败。

## 组件职责

### 阶段生产者

现有生成结果增加最小导航提示，不增加新 Agent 或独立模型调用：

- story architect 为故事大纲返回有序章节锚点；为人物小传返回有序人物锚点；为人物关系返回有序关系锚点；
- episode planner 在现有逐集计划上返回显示标签；
- script writer 为当前集返回集名和可选场次锚点。

锚点必须是生成原文中的完整行；它只定位内容边界，不规定某种中文、Markdown 或剧本格式。稳定 item id、ordinal 和 hash 由运行时生成，不能让模型决定。

### Presentation compiler

新增内部纯函数组件，输入已批准原文、导航提示、episode plans 和 episode drafts，输出 schema v1 manifest。它负责：

1. 逐文稿验证锚点按原文顺序存在；
2. 以相邻锚点切分内容，要求标签可在对应内容中核对；
3. 验证 item id 唯一、ordinal 从 1 连续；
4. 验证 episode number 与已批准计划/草稿连续且一致；
5. 验证场次属于对应单集原文；
6. 计算原文和每个展示项的 SHA-256；
7. 单个文稿无法验证时生成 `source` 模式，不猜测或丢弃原文。

Compiler 不调用模型、不读取 persona、不解释文学语义。人物分组、关系分组和章节层级来自同阶段受验证的导航提示；缺失或非法时该文稿回退 source 模式。

### Repository

在 delivery 成功事务中，先按现有规则验证正式交付，再编译并写入可空 `presentation_manifest_json` 和对应 SHA-256。manifest 与 delivery 同事务提交；manifest 失败不得阻止已通过全部业务 Gate 的正式交付，改为写入 source-mode manifest 或 `NULL`。

读取时 Repository 重新校验 schema version、manifest hash 和五类 source hash。损坏文稿现场降级 source 模式，并记录不含正文的诊断事件。

### API

新增 `getDeliveryPresentation` 查询，不扩展现有 `getCreation`：

- 避免轮询资源长期携带重复展示数据；
- initial/revision 使用同一读取语义；
- 只接受成功且存在 formal delivery 的 run；
- 历史 delivery 没有 manifest 时返回 HTTP 200；若已有连续且验证通过的 `episode_plans` / `episode_drafts`，只读投影为按集导航，其他文稿保持 `source`，且不回填数据库、不调用模型；
- 创建不存在返回 `creation_not_found`；指定 run 尚无正式交付返回 `presentation_not_available`。

API 不暴露内部 checkpoint、数据库行、提示词或模型导航提示，只返回经过 Compiler 验证的读取模型。

### Web 消费者

进入成品阅览室时只请求一次 presentation；切换初稿/修订稿时请求对应 run kind。UI 按每个文稿的 `mode` 决定结构化导航或完整原文回退，不自行解析标题、人物、关系、集或场。

# 流程与失败行为

## 新交付

1. 各生成阶段在原有结构化结果中携带最小导航提示。
2. 现有 reviewer、repair、business checkpoint 和最终 Gate 继续只决定内容是否批准。
3. Delivery assembler 生成现有五类原文。
4. Presentation compiler 对最终已批准文本编译展示 manifest。
5. Repository 在同一成功事务中持久化 delivery 和 manifest，然后把 run 置为 succeeded。
6. Web 先从现有 Creation resource 确认 succeeded，再调用只读 presentation 查询。

修复如果改变了锚点行，修复结果必须同步返回更新后的导航提示；否则 Compiler 会把受影响文稿降级 source，而不是沿用陈旧边界。

## 历史交付

不批量回填，不重新调用模型。读取历史 succeeded run 时，从正式 `ContentPackage` 构造五个 source-mode 文稿。即使数据库保留 `episode_plans`/`episode_drafts`，读取路径也不现场生成或回填 manifest；只有新交付在成功事务中编译并保存的 manifest 可以启用 structured 模式。

## 部分损坏

- 单项 hash、锚点、序号或 episode 对齐失败：该文稿 source 模式，整体状态 `partial`；
- 五项全部缺少或失败：整体状态 `source`；
- 五项全部结构化：整体状态 `complete`；
- delivery 原文不存在或 run 未成功：409，不生成假展示；
- manifest JSON 无法解析或 schema version 不支持：整份 manifest 忽略，五项 source 模式。

客户端不重试确定性的 source/partial 降级；只有普通网络或服务不可用错误按现有读取策略重试。

# 共享契约变更

正式契约是 canonical `contracts/openapi.json`：

- 新增一个 GET 操作，不修改现有操作、响应或错误语义；
- 响应固定包含五类文稿，每类同时携带完整原文、原文 hash、mode 和类型化 items；
- structured 模式要求至少一个 item，source 模式要求 items 为空；
- episode items 明确 episode number，scene items 可为空；
- 关系展示只包含关系条目和完整文字，不定义节点/边图协议；
- 代表性 complete 与 source fallback 示例内嵌于契约并参与模式验证。

后端 Pydantic 模型、运行时路径和 canonical OpenAPI 必须保持一致；前端只消费该只读端点，不维护第二份 schema。

# 数据、兼容性与迁移

- SQLite 使用单向、可空的 additive migration；现有 delivery 行不回填、不重写。
- 新列只保存展示 manifest 和完整 manifest hash；正式原文仍在原列。
- 新 endpoint 是兼容性新增；现有 `getCreation`、ContentPackage、revision 和轮询消费者零变化。
- schema version 从 1 开始；未知版本 fail closed 到 source 模式，不能尽力解析。
- 发布顺序：数据库/Repository/Compiler → API 与 canonical OpenAPI → Web 消费者。
- 回滚顺序：Web 退回现有原文阅读 → 停用新 endpoint → 保留可空列。旧二进制忽略新增列，不需要降级迁移。
- 如果未来需要关系图、搜索索引或富文本编辑，必须新增版本或独立投影，不能扩张 v1 条目的语义。

# 安全、可靠性与运维

- 仍只绑定 loopback；无新增身份验证、跨用户或外部传输边界。
- endpoint 只读且无幂等键；任何 GET 都不得产生模型调用或数据库写入。
- 响应包含与现有 Creation 成品相同的创作正文，不扩大数据类别。
- 诊断只记录 creation id、run kind、schema version、文稿 key、降级原因和 hash；不记录正文或导航内容。
- 编译工作发生在 delivery 提交前；GET 只做有界 JSON/hash 校验和映射，不做与正文长度平方相关的扫描。
- 五类原文必须始终可回退，因此展示投影故障不影响作品可读性。

# 备选方案与决策

## 在现有 Creation resource 中直接增加 presentation

拒绝。该资源同时承担运行轮询，会放大响应并把只在 succeeded 后需要的数据带入所有状态；还会扩大现有联合类型的兼容面。

## 前端按中文序号、人物名或剧本格式解析

拒绝。它对题材、语言、人格和模型格式敏感，且会把推断出的结构伪装成权威数据。

## 读取时调用模型生成目录

拒绝。GET 会产生费用、延迟、失败和不确定副作用，破坏只读与可复现性。

## 只修复分集剧本导航

拒绝。它能复用现有逐集记录，但不能解决用户明确指出的故事、人物、关系和分集大纲层级问题；五套不一致的临时逻辑也会增加长期熵。

## 将 presentation 作为第六类正式交付

拒绝。展示是可重建投影，不应参与内容审核、交付完整性或创作所有权。

# 验证

- 目标 OpenAPI 必须通过格式感知 lint；内嵌 structured/source 示例必须通过 `DeliveryPresentation` schema。
- 契约必须证明 source 模式 items 为空、structured 模式 items 非空、未知字段被拒绝。
- 实施时验证现有 OpenAPI 操作集合除一个新增 GET 外不变，现有测试和消费者继续通过。
- Repository 测试覆盖新交付完整投影、历史 NULL、单项 hash 损坏、未知 schema version 和 source fallback。
- API 测试覆盖 200 complete/partial/source、404 creation 和 409 presentation unavailable，并断言 GET 零写入、零模型调用。
- Web 验收引用 `../ux/deliverables-reading-room.md` 的桌面、390 px、键盘、恢复位置与逐集阅读映射。

# 待定事项

无。关系图、搜索、编辑和历史模型回填均明确不在 v1 展示契约中。
