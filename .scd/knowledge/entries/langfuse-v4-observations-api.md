---
status: active
scope: project
origin: 2026-08-08 全流程排障（Langfuse 轨迹核查）
updated: 2026-08-08
---

# 本项目自建 Langfuse v4 用 v2 observations API 查询轨迹

- Trigger: 需要用 API 查询本地 Langfuse（UI http://127.0.0.1:3001）里 pengine 的模型调用轨迹。
- Guidance: v3 风格的 `/api/public/traces` 返回 404；应使用 `/api/public/v2/observations?fromStartTime=<ISO8601>`（basic auth，pk/sk 见本仓 `.env` 与 `docker-compose.langfuse.yml`，勿外传）。项目为 events_only 采集模式，直接按 observation 维度查即可。
- Boundary: 仅适用于本项目 docker-compose 自建的 v4 实例；Langfuse Cloud 或其他版本的路径可能不同。
- Evidence: 实测 `/api/public/traces` → 404，`/api/public/v2/observations` → 200 并返回本次 run 的 100+ 条 observation。
- Source: docker-compose.langfuse.yml；2026-08-08 curl 实测。
