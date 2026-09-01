---
managed_by: scd-architecture
status: ready
sources:
  - ../architecture.md
  - ../../contracts/openapi.json
  - ../ux/account-access.md
---

# 结果与边界

本设计为服务器部署的 Pengine 增加最小账户边界。认证服务负责密码、会话和当前用户；创作仓储负责在每个作品读写命令中执行所有者过滤。它不增加角色、团队、分享、密码找回、删除或并行 Worker。

## 现有上下文

- `src/pengine/api.py` 的现有作品端点没有身份上下文。
- `src/pengine/repository.py` 使用 SQLite，当前 schema 版本为 32，`creations` 无所有者列；`idempotency_records` 的幂等键全局唯一。
- 现有进程继续只监听回环地址；静态 Web 与 API 同源。

## 领域与职责变更

| 事实 | 权威所有者 | 不变量 |
| --- | --- | --- |
| 账户 | `users` | `username` 去除首尾空白后非空、最多 64 个字符，按原值唯一；只保存 Argon2id 密码哈希。 |
| 登录会话 | `sessions` | Cookie 只含高熵不透明 token；数据库仅保存其哈希、账户、过期时间和撤销时间。 |
| 作品归属 | `creations.owner_id` | 每条作品恰有一个账户所有者；创建时从认证上下文写入，后续不得转移。 |
| 我的创作 | Repository 查询投影 | 只返回当前账户的作品，按 `updated_at` 倒序；故事文本是摘要字段，不伪装成作品标题。 |

认证边界在 API middleware/dependency：除注册、登录和登出外，现有业务端点必须存在有效会话。作品读取和每个 run/revision 控制命令再以 `creation_id + owner_id` 查询；不存在和非所有者统一为既有 `creation_not_found` 404，不能泄露他人作品是否存在。Persona 列表也只对已登录用户提供。

## 流程与失败行为

1. 未登录访问 Web：只显示登录/注册入口；业务 API 返回 `authentication_required` 401。
2. 注册：校验用户名和密码，唯一冲突返回 `username_taken` 409；成功后创建账户和会话，浏览器直接进入工作台。
3. 登录：密码不匹配与不存在用户统一返回 `invalid_credentials` 401；成功后替换当前会话 Cookie。
4. 会话：固定 7 天有效；过期、撤销或无效 token 返回 401，Web 保留未提交表单内容并返回登录页。
5. 退出：撤销当前 session 并清除 Cookie；重复退出仍为成功。
6. 创建作品：以当前 `owner_id` 持久化；所有原有 Worker、检查点和全局单队列语义不变。

## 共享契约变更

规范来源仍为 `contracts/openapi.json`。实现已新增并由 FastAPI 生成后回写该文件：

- `POST /auth/register`、`POST /auth/login`、`POST /auth/logout`；
- `GET /me`；
- `GET /creations`（当前账户的分页简表）；
- 对所有既有业务操作声明 401，并将所有者缺失保持为既有 404。

`GET /creations` 的条目至少含 creation ID、编剧人格显示名、初稿/修订状态、创建/更新时间和受限长度的故事摘要；不新增虚假的 `title` 字段。幂等记录必须按 `(owner_id, idempotency_key)` 作用域隔离，避免不同账户相互冲突或重放。

## 数据、兼容性与迁移

- 一次性 v33 migration 先清除所有历史创作派生数据：作品、run、任务、检查点、交付、修订、创作相关审计与旧幂等记录；保留人格包、运行配置和 SQLite 文件本身。
- 同一 migration 创建 `users`、`sessions`，向 `creations` 添加 `owner_id`。SQLite 的原位 `ALTER TABLE` 保留可空物理列以兼容仓储级历史测试，生产创建入口始终从强制认证上下文写入 owner；因升级先清空作品，不存在无法归属的旧行。
- 现有幂等表保持不变，API 在进入仓储前将账户 UUID 编入内部幂等键，因此外部相同 key 可由不同账户独立使用且不会互相重放。
- migration 必须在单个 `BEGIN IMMEDIATE` 事务内完成；失败回滚，成功后 schema version 才前进。运行前要求可恢复的数据库备份，回滚发布通过恢复备份，而不是降级 schema。
- 发布顺序：备份 → 后端 migration/认证/授权 → canonical OpenAPI → Web 消费者 → 私有 HTTPS 反向代理验收。旧前端不可与新后端共存；根 HTML 继续 `no-store` 以降低陈旧前端风险。

## 安全、可靠性与运维

- 反向代理以 HTTPS 向内部用户服务；应用保持回环监听，管理端口不得直接暴露。
- 会话 Cookie 为 `Secure`、`HttpOnly`、`SameSite=Lax`、path `/`，且 API 不启用跨源凭据 CORS。
- 密码使用参数化 Argon2id 哈希；注册、登录与写入端点不记录密码、Cookie 或 Authorization 值。
- 密码错误、用户不存在和非所有者作品访问均使用统一可观察错误，避免枚举。
- 账户数量增加不改变单 Worker 全局队列；若未来对公网开放、计费或并发成为问题，重新评审邀请制、速率限制和每用户配额。

## 备选方案与决策

- 将 `creation_id` 存在浏览器 localStorage 作为归属：拒绝，客户端状态不能承担服务器授权。
- 把登录 token 放入 localStorage：拒绝，改用 HttpOnly Cookie 以降低脚本窃取风险。
- 为历史创作创建首个账户认领：拒绝，服务器注册顺序会导致错误归属；用户确认清空历史作品。
- 立即引入管理员、邮箱和密码找回：拒绝，超出已确认 MVP。

## 验证

- 迁移测试：含历史创作的数据库升级后作品相关表为空、账户表可用、失败时事务完整回滚。
- API/契约测试：未登录 401；跨账户每个作品 GET/控制端点均为 404；同账户可读写；相同幂等键可由不同账户独立使用。
- 安全测试：数据库没有明文密码或会话 token；Cookie 属性正确；日志脱敏。
- 部署验收：HTTPS 登录、退出、会话过期和反向代理到回环服务均可观察验证。

## 待定事项

无。实际服务器域名、证书与反向代理产品属于部署参数，不改变本设计边界。设计已由实现、运行时 OpenAPI 与账户隔离测试验证，可交付 QuickDev。
