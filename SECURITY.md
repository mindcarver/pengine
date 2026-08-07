# 安全说明

## V1 部署边界

Pengine V1 默认只绑定 loopback，且没有身份认证。请不要将 `uv run pengine` 直接暴露到局域网或公网；它不是多用户服务。

## 不要公开的内容

- `.env`、relay/API key 和 Langfuse secret；
- `data/`、SQLite/WAL、persona snapshots；
- 真实用户故事、完整模型请求/响应和 live E2E artifact；
- 未授权的人格、作品、图片或个人信息。

## 报告方式

请不要在公开 Issue 中粘贴 secret、完整 prompt、用户正文或数据库副本。若发现凭据泄露，先立即撤销/轮换凭据，再通过仓库维护者指定的私下渠道报告，并提供最小化复现信息：影响版本、触发路径、错误分类和不含敏感数据的日志片段。

## 已知边界

Relay 会收到运行所需的故事、人格上下文和阶段内容；本地 SQLite 的保留也没有自动过期。部署者需要自行评估 provider 的数据处理政策、备份保护和访问控制。
