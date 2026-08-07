---
layout: default
title: HTTP API
permalink: /api/
---

{% include nav.md %}

# HTTP API

Pengine 的机器接口是本地 JSON HTTP API。完整字段合同在 [`contracts/openapi.json`](https://github.com/mindcarver/pengine/blob/main/contracts/openapi.json)，FastAPI 运行时也会在 `/docs` 提供交互式 Swagger UI。

## 1. 连接与约定

| 项目 | V1 约定 |
| --- | --- |
| 默认地址 | `http://127.0.0.1:8000` |
| 绑定范围 | 只允许 loopback；不要绑定到局域网/公网 |
| 内容类型 | 请求体使用 `application/json` |
| 长任务 | 创建和控制命令返回 `202 Accepted`，通过资源 GET 轮询 |
| 幂等 | 所有改变状态的命令要求 `Idempotency-Key`，同 key + 同 payload 重放原响应 |
| 冲突 | 同 key + 不同 payload 返回 `idempotency_conflict` |
| 错误体 | `{ "code": "稳定代码", "message": "安全说明" }` |
| 认证 | V1 没有认证，只可在本机使用 |

`Idempotency-Key` 长度为 1–128 个字符。它不是 creation id，也不是 provider request id；它只标识一次调用方命令。模型调用的 `call_id` 由服务内部生成并在资源里展示。

## 2. 端点总览

| 方法 | 路径 | 作用 | 成功响应 |
| --- | --- | --- | --- |
| `GET` | `/personas` | 返回当前有效可选人格包 | `200 PersonaList` |
| `POST` | `/creations` | 排队一轮初稿全流程 | `202 CreationAccepted` |
| `GET` | `/creations/{creation_id}` | 查询初稿、修订状态、进度、草稿、证据和交付 | `200 CreationResource` |
| `POST` | `/creations/{creation_id}/revision` | 冻结并排队唯一修订，或重排队相同反馈的失败修订 | `202 RevisionAccepted` |
| `POST` | `/creations/{creation_id}/runs/{run_kind}/continue` | 继续暂停的初稿/修订 run | `202 RunControlAccepted` |
| `POST` | `/creations/{creation_id}/runs/{run_kind}/retry-final-review` | 只重跑被拒绝的 L0/L4 最终审核 | `202 RunControlAccepted` |
| `POST` | `/creations/{creation_id}/runs/{run_kind}/authorize-repair` | 授权一次绑定血缘的内容修复周期 | `202 RunControlAccepted` |
| `POST` | `/creations/{creation_id}/runs/{run_kind}/end` | 结束可控制的暂停/拒绝 run | `202 RunControlAccepted` |

其中 `{run_kind}` 只能是 `initial` 或 `revision`；`{creation_id}` 是 UUID。

## 3. 获取人格

```bash
curl --fail-with-body \
  http://127.0.0.1:8000/personas
```

响应只包含已通过校验的可选包，不会把无效人格目录当成可选项：

```json
{
  "items": [
    {
      "persona_id": "wuzhen",
      "display_name": "雾枕",
      "version": "0.1.0",
      "snapshot_sha256": "<64 位小写 sha256>"
    }
  ]
}
```

`snapshot_sha256` 是服务实际会绑定到创作任务的人格身份。源目录后续变化不会替换已有任务的快照。

## 4. 创建初稿

```bash
curl --fail-with-body -X POST \
  http://127.0.0.1:8000/creations \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: creation-20260807-001' \
  -d '{
    "persona_id": "wuzhen",
    "story": "一个离开故乡多年的人回乡处理旧屋。",
    "requirements": "创作一部完整短剧，保留人物选择的代价。"
  }'
```

返回：

```json
{
  "creation_id": "018f6d49-2e10-7b21-8f40-5f5162b9d181",
  "initial_state": "queued",
  "resource_url": "/creations/018f6d49-2e10-7b21-8f40-5f5162b9d181"
}
```

调用方应保存 `creation_id` 或 `resource_url`，然后轮询 GET。创建命令会先执行人格快照处理；快照不成功时不会创建一个缺失人格证据的任务。

## 5. 查询资源

```bash
curl --fail-with-body \
  http://127.0.0.1:8000/creations/018f6d49-2e10-7b21-8f40-5f5162b9d181
```

`CreationResource` 的稳定结构是：

```json
{
  "creation_id": "<uuid>",
  "persona": {
    "persona_id": "wuzhen",
    "display_name": "雾枕",
    "version": "0.1.0",
    "snapshot_sha256": "<sha256>"
  },
  "initial": { "state": "running", "...": "见 OpenAPI" },
  "revision": { "state": "unavailable", "feedback_locked": false },
  "created_at": "2026-08-07T00:00:00Z",
  "updated_at": "2026-08-07T00:00:05Z"
}
```

运行中的 `initial`/`revision` 状态会携带：

- `progress.current_stage` 和 `completed_stages`；
- 已运行秒数、`recovery_state`、`recovery_reason`；
- L0/L4 最终审核子状态；
- 总集数、已完成集数、当前集和已提交 `episode_drafts`；
- `model_calls` 的估算/实际用量、角色、模型、结束原因和安全错误；
- `can_continue`、`can_end` 和暂停/拒绝证据；
- 只有 `succeeded` 才有完整 `delivery`。

不应根据 `current_stage` 自己推断正文已经成功；以 `business_checkpoints` 映射出的 `completed_stages` 和最终状态为准。

## 6. 修订

初稿必须是 `succeeded` 才能提交修订：

```bash
curl --fail-with-body -X POST \
  http://127.0.0.1:8000/creations/018f6d49-2e10-7b21-8f40-5f5162b9d181/revision \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: revision-20260807-001' \
  -d '{
    "feedback": "让结尾的代价更明确，同时保留原有情绪底色。"
  }'
```

第一次成功接收后，反馈立即冻结：

- queued/running/auto_resuming 的修订不能再次提交；
- failed 修订只能以完全相同的 feedback 重排队；
- 重排队会创建新的 revision-attempt run，旧失败证据保留；
- ended 修订不能重排队；
- succeeded 修订永久关闭修订入口；
- 初稿交付不会被修订覆盖，直到修订自身成功。

## 7. 运行控制

### Continue

```bash
curl --fail-with-body -X POST \
  http://127.0.0.1:8000/creations/CREATION_ID/runs/initial/continue \
  -H 'Idempotency-Key: continue-001'
```

只对可继续暂停有效。相同 key 重放成功响应；不同 key 不能绕过当前运行状态或内容预算。

### Retry final review

```bash
curl --fail-with-body -X POST \
  http://127.0.0.1:8000/creations/CREATION_ID/runs/initial/retry-final-review \
  -H 'Idempotency-Key: final-review-001'
```

只允许 `quality_rejected` 的 run 使用。它不重跑前面的生成阶段、不改变草稿、不重新生成内容，只重新执行被拒绝的最终质量 gate；最终审核尝试次数耗尽后拒绝命令。

### Authorize repair

```bash
curl --fail-with-body -X POST \
  http://127.0.0.1:8000/creations/CREATION_ID/runs/initial/authorize-repair \
  -H 'Idempotency-Key: repair-authorization-001'
```

只允许存在当前有效 repair authorization 时使用。授权对象会绑定设计候选、batch、影响集数、review id 和证据；它只消耗一次生成+审查周期。通用 Continue 不能代替它。

### End

```bash
curl --fail-with-body -X POST \
  http://127.0.0.1:8000/creations/CREATION_ID/runs/initial/end \
  -H 'Idempotency-Key: end-001'
```

结束会保留已提交的业务 checkpoint、草稿、review 和 model-call audit，但不会生成 delivery，也不会把当前未提交候选视为正式内容。

## 8. 状态值

### Run

资源中的 run 状态可能是：

`queued` → `running` → `auto_resuming` / `paused` / `ended` / `failed` / `quality_rejected` / `succeeded`

`quality_rejected` 是“成品存在但最终质量 gate 没过”的特殊可审计状态；它和 `failed` 不同，因为它允许只重试最终审核。

### Revision

修订资源在初稿成功前为 `unavailable`，之后可能为 `available`、`queued`、`running`、`auto_resuming`、`paused`、`ended`、`failed`、`quality_rejected` 或 `succeeded`。第一次接受 feedback 后 `feedback_locked` 为 `true`。

### Model call

每个调用的 `status` 可能为 `started`、`succeeded`、`failed`、`timed_out`、`stale`、`superseded` 或 `preflight_blocked`；`usage.status` 为 `reported`、`partial` 或 `unavailable`。这两种状态不能混读：一个调用成功但 provider 缺 usage，仍然可能是 `succeeded + unavailable`。

## 9. 稳定错误代码

| HTTP/代码 | 含义 | 调用方建议 |
| --- | --- | --- |
| `422 invalid_request` | JSON 字段缺失、空白或类型不合法 | 修正 payload |
| `404 persona_not_found` | persona_id 不存在 | 重新 GET `/personas` |
| `503 persona_package_unavailable` | 人格包存在但未通过加载/快照条件 | 修复人格包，不要猜测正文 |
| `404 creation_not_found` | creation_id 不存在 | 检查保存的资源地址 |
| `409 idempotency_conflict` | 同 key 的 payload 不同 | 使用原 payload 或新的 key |
| `409 revision_not_allowed` | 初稿未成功或修订已关闭 | 读取完整资源状态 |
| `409 revision_feedback_locked` | feedback 已冻结且新值不同 | 只能使用原 feedback |
| `409 run_not_controllable` | 当前状态不允许该控制命令 | 读取 `can_continue/can_end` |
| `409 repair_authorization_stale` | 授权已过期或血缘不再 active | 重新读取暂停证据 |
| `409 series_bible_rebuild_exhausted` | 自动设计重建预算已用尽 | 等待明确授权，不要 generic continue |
| `503 service_unavailable` | 未暴露内部细节的服务级错误 | 记录 code，检查服务日志和数据库 |

服务不会把 DomainError 的未知内部 code 直接暴露成 HTTP 500；对外会收敛到稳定的 `CommandError` 合同。消息是安全说明，不应包含 API key、完整模型 prompt 或用户敏感正文。

## 10. 接入方伪代码

```text
personas = GET /personas
accepted = POST /creations with unique Idempotency-Key

loop:
    resource = GET accepted.resource_url
    display resource.initial.progress

    if resource.initial.state == succeeded:
        read resource.initial.delivery
        break
    if resource.initial.can_continue:
        ask operator, then POST continue (same run)
    if repair authorization is present:
        show evidence, then require explicit authorize-repair
    if resource.initial.can_end:
        allow operator to POST end
```

不要轮询一个新的 creation id 来处理 timeout；不要从 UI 文案推断最终交付；不要复用创建 key 发送不同 story；不要把未批准 draft 当成正式 `delivery`。
