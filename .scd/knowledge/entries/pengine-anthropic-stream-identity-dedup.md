---
status: active
scope: project
origin: Issue #123 / PR #124
updated: 2026-08-13
---

# pengine Anthropic 流身份必须在聚合前去重

- 触发条件：流式 generation 调用被暂停为 `relay_identity_mismatch`，持久化身份呈同一模型名重复拼接，例如 `claude-opus-5claude-opus-5`。
- 指引：不要放宽允许身份或在聚合后拆分字符串。先检查原始流块；Pengine 必须在 `_SerialChatAnthropic` 的同步与异步流边界，仅移除后续块中完全相同的 `model` 或 `model_name`，再交给现有身份门验证。
- 边界：只适用于同一次流中多个块报告完全相同身份。不同身份、缺失身份或单个块直接报告拼接身份仍必须 fail-closed。
- 证据：修复前调用持久化 `["claude-opus-5claude-opus-5"]` 并暂停；重复流块测试复现字符串拼接。修复后相同身份聚合为单个 `claude-opus-5`，错误边界测试继续拒绝，原 run 续跑后的两个同阶段调用均以 `["claude-opus-5"]` 成功。
- 来源：`src/pengine/relay.py`、`tests/test_relay.py`、Issue #123、PR #124、合并提交 `4c0202f`。
