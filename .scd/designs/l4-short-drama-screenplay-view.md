---
managed_by: scd-architecture
status: ready
sources:
  - https://github.com/mindcarver/pengine/issues/109
  - ../architecture.md
  - ../../contracts/persona-package.schema.json
---

# 结果与边界

Pengine 在不改变 persona schema、公共 API、数据库、Agent 数量或工作流阶段的前提下，把经确认的守拙剧本观接入现有 L4 链路。

设计只改变四件事：

1. 明确运行时 L4 的规则资格与权属；
2. 把长剧经验按戏剧功能编译为短剧规则；
3. 让现有阶段审核结果同时保存适用 L4 硬规则的证据；
4. 把当前参数段明确降格为 Pengine 默认产品规则的兼容投影。

原始 L9 文档不是运行时协议。Issue #109 是产品行为权威，persona snapshot 中的 `l4.md` 是单次创作的运行时权威，现有 `StoryContract`、`SeriesBible`、`SeriesState` 和 business checkpoint 继续拥有剧情事实与进度。

## 现有上下文

- persona schema v3 固定八个 Markdown 文件，`l4.md` 已参与 package hash 和 immutable snapshot。
- `src/pengine/personas.py::_stage_l4_context` 当前始终装载 L4-A，并按故事、人物关系、分集大纲、分集剧本切片 L4-B；`accepting_l4` 读取完整 L4。
- 当前参数段进入所有四个生成阶段；用户未指定集数时，现有 supervisor 会采用 L4 基线。
- 故事、人物关系和分集大纲已有 Canon 审核结果及有界修复；剧本已有确定性逐集校验、里程碑/终集系列审核；最终已有 `quality_reviewer` L4 Gate。
- `personas/shouzhuo/l4.md` 仍明确声明为项目方 AI 原型，不能作为守拙本人确认的 L4 定稿。
- 当前 `quality_reviewer` 已限制：只有用户要求、明确 persona gate 规则、锁定合同/SeriesBible、冻结反馈或输出协议的直接冲突才可拒绝；普通风格和审美不足不能拒绝。

## 领域与职责变更

### 规则资格

运行时 L4 只允许三类内容：

| 类别 | 权威主体 | 运行作用 | 能否单独拒绝 |
| --- | --- | --- | --- |
| 守拙确认硬规则 | 守拙或授权内容负责人 | 生成约束、阶段审核、最终 L4 Gate | 能，必须给出规则与候选的直接冲突证据 |
| 守拙确认创作建议 | 守拙或授权内容负责人 | 生成偏好、修复参考 | 不能 |
| Pengine 默认产品参数 | Pengine 产品合同 | 用户未明确时提供默认集数、时长和场数 | 能，但必须以产品规则名义，不能写成守拙观点 |

AI、市场或编辑推导的未确认候选不属于运行时 L4。它们可以保留在受控研究来源或 L6 候选材料中，但不得装载到 `/persona/l4.md`，也不得出现在 Gate 证据里。

### L4 文件职责

守拙 `l4.md` 维持现有固定文件与必需标题，正文采用以下语义结构：

```text
# 守拙 L4 剧本观与短剧化规则
## L4-A 价值观
  - 来源指纹、确认状态、归属、权限说明
  - 硬规则
  - 已确认创作建议
## L4-B 短剧技艺
### 全阶段通则
  - 硬规则
  - 已确认创作建议
### 故事大纲
### 人物小传
### 人物关系逻辑
### 分集大纲
### 分集剧本
## 分环节标准参数
  - 所有者：Pengine
  - 默认值与用户覆盖规则
```

只有使用“硬规则”语义明确声明的条款属于 Gate。建议使用“优先、可以、建议”等非强制措辞。未经确认内容不得靠“候选”“参考”标签混入运行文件，因为最终审核会读取完整 L4。

### 所有权与优先级

创作规则冲突仍按现有优先级处理：

```text
用户要求 / 已批准 checkpoint / 已锁定 Canon 与合同
  > L0
  > L4 已确认硬规则
  > L3
  > Soul
```

生产参数使用独立覆盖链：

```text
用户明确参数
  > 已锁定任务生产参数
  > Pengine 默认产品参数投影
```

L4 创作建议不能覆盖以上任何硬约束。Pengine 参数也不能反向解释为创作者价值判断。

## 流程与失败行为

### 离线编译与发布

1. 人工核对受控 L9 来源及 SHA-256；来源不一致则停止。
2. 按 Issue #109 的保留、转译、排除清单编译守拙 L4；不运行时解析原文。
3. 在 L4-A 写入来源指纹、确认状态、归属和 Gate 权限。
4. 更新 `project.md` 的 L4 状态与摘要，不复制 L4 正文。
5. 更新 persona version、逐文件 hash 与 package hash；创建新任务时生成新的 immutable snapshot。
6. 旧任务继续解析原 snapshot，不回填、不迁移、不静默升级。

### 运行时投影

`_stage_l4_context` 保持现有切片，并新增“全阶段通则”为四个生成阶段的共同切片。每个生成阶段只接收：

```text
L4-A + 全阶段通则 + 当前阶段规则 + Pengine 参数投影
```

`accepting_l4` 继续读取完整 L4。`selecting_l0_variant` 与 `accepting_l0` 不使用 L4-B 重新选择或解释 L0。

### 审核证据复用

不新增审核 Agent 或持久化结构。现有结果字段承担 L4 证据：

| 阶段 | 现有审核载体 | L4 责任 |
| --- | --- | --- |
| 故事大纲 | `consistency_review` | 检查 L4-A、全阶段及故事大纲硬规则 |
| 人物小传/关系 | `consistency_review` | 检查 L4-A、全阶段、人物和关系硬规则 |
| 分集大纲 | `contract_review` | 检查 L4-A、全阶段及分集大纲硬规则；需要时用现有有界修复 |
| 分集剧本 | 现有里程碑/终集 `BoundStructuralReview` | 在已触发的结构审核点检查剧本硬规则；非里程碑集仍由生成约束、确定性合同校验和最终 Gate 覆盖 |
| 最终 L4 | `QualityReviewerResult.evidence` | 对完整交付执行唯一聚合 L4 Gate |

阶段 reviewer 只能把明确硬规则的直接冲突写入 blocking issues。确认建议、普通格式偏好和审美意见仍属于非阻断内容，不得塞进 blocking issue 规避现有 schema。

passing 阶段证据使用稳定标签 `L4硬规则：`；最终 passing evidence 使用：

```text
L4-A：
短剧硬规则：
产品参数：
```

应用只确定性校验标签存在、stage 与 `passed` 协议，不用关键词扫描解释文学质量。标签后的语义判断继续由 reviewer 承担。

### 失败与恢复

- L4 含待确认、无归属或来源指纹不匹配：不得发布新的守拙 persona version。
- Stage reviewer 以建议或审美拒绝：视为 reviewer 协议错误，不能批准 business checkpoint。
- passing review 缺少要求的 L4 证据标签：视为结构化协议错误，走现有有界结构修复/失败路径，不伪造通过证据。
- 用户参数与默认值冲突：采用用户/锁定任务参数；不得产生 L4 拒绝。
- 新版 persona 出现质量回归：停止让新任务选择该 source version；已创建任务仍保持原 snapshot，可用旧 persona source version 重新创建新任务，不改历史任务。

## 共享契约变更

无新的前端、HTTP、数据库、事件或插件边界，因此不新增 OpenAPI/JSON Schema。

现有 `contracts/persona-package.schema.json`、manifest v3 文件集合、public API 与 SQLite schema 均保持不变。L4 标题结构仍由现有 persona loader 与聚焦测试执行；新增的“全阶段通则”和审核证据标签是 Pengine 内部 Prompt/loader 协议，不对外暴露。

若未来把 Production Profile 独立为前后端共享资源，则必须另行定义机器可读契约；本设计不预留未使用字段。

## 数据、兼容性与迁移

- 不执行数据库迁移。
- 不新增 persona schema v4：文件集合、投影机制与公共兼容性均未改变，内容版本由 manifest `version` 和 hash 表达。
- 历史 v1/v2/v3 snapshot 继续按其原正文、hash 和投影恢复。
- 仅守拙 persona 内容升级；三个其他 persona 的创作内容不改。
- 当前四个 persona 中重复的 6 集、约 2 分钟、2–3 场参数暂时保留为兼容投影。测试要求这些默认值一致，并要求每份参数段明确归属 Pengine。
- 一旦出现第二种产品制式、persona 参数需要分化，或同一默认值发生两处修改，本设计的暂存方案失效，必须提取单一 Production Profile 权威源。

## 安全、可靠性与运维

- 原始 L9 Markdown/DOCX 保持工作区受控资料，不进入 Git、snapshot、Prompt、SQLite、日志或 Langfuse。
- 运行资产只记录来源 SHA-256，不记录未采用原文或个人信息。
- model-call 和 Langfuse 已有 persona schema/id/version/snapshot 元数据足以定位实际使用的 L4 版本；不新增正文日志。
- 审核证据可证明模型返回的判断，但不能证明规则本身由创作者确认；确认权威来自 Issue #109、运行文件状态和 snapshot hash。

## 备选方案与决策

### 直接装载原始 L4

拒绝。它混合长剧尺度、创作者判断、AI/市场补写和产品参数，无法保持权属或 Gate 边界。

### 新增 persona schema v4

暂不采用。当前没有文件集合、历史投影或外部契约变化；用内容 version/hash 和现有结构验证即可表达本次增量。若以后需要逐条机器化 authority metadata，再以独立需求评估 schema 演进。

### 立即建设 Production Profile

暂不采用。当前只有一个实际默认制式，新增子系统的成本高于消除的风险。用明确归属和跨 persona 一致性测试约束过渡状态；触发第二制式时再提取。

### 新增 L4 Reviewer 或阶段

拒绝。现有 Canon、系列和 quality reviewer 已覆盖所需边界；复用现有结果能提供证据并避免新的状态、重试和恢复路径。

### 对剧本执行关键词/格式扫描

拒绝。心理活动、旁白、对白格式和叙事符号必须结合具体作品判断；确定性代码只验证稳定协议、锁定数据和证据标签。

## 验证

### 已完成的设计证据

- 当前基线 `038bc594af9385a61e779614888da4f797c19ffc` 已核对 persona v3、L4 阶段切片、最终 Gate、immutable snapshot 和现有 review payload。
- 受控来源 SHA-256 已核对为 `ef6c2125cc330209d2ec16761e9a8e4daa5235d5004815ea5c99dbe650833d40`。
- Issue #109 已保存确认后的产品行为、范围、失败场景和 A1-A12 验收。
- 设计不改变共享机器契约，因此无待解析的新 OpenAPI、JSON Schema、数据库 migration 或消费者代码生成。

### 实施交接验证

实施至少运行：

```bash
uv run pytest \
  tests/test_personas.py \
  tests/test_bundled_personas.py \
  tests/test_agents.py \
  tests/test_worker.py \
  tests/test_series_review.py \
  tests/test_contracts.py
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv build --no-sources
```

真实模型 A/B/C 使用隔离 persona、临时 SQLite、相同模型与参数，分别验证硬规则冲突、仅建议偏差和用户参数覆盖。A1-A12 全部有直接证据后才能关闭 Issue #109。

## 待定事项

无。Production Profile 的提取条件已经定义，不阻塞当前设计或兼容性。
