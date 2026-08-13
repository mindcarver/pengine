# 知识索引

- [pengine 语义审查子智能体没有内部模型调用上限](entries/pengine-subagent-review-loop-not-capped.md) - 子智能体评审循环卡住、review 大量但无 rewrite、canon_reviewer 不收敛、c+r 阶段阻塞
- [pengine relay 对非流式请求有约 300 秒硬超时](entries/pengine-relay-non-streaming-300s-timeout.md) - 生成调用恰好 ~300s 死于 408 timeout_error、请求超时、长调用反复重试失败、streaming=True
- [pengine Anthropic 流身份必须在聚合前去重](entries/pengine-anthropic-stream-identity-dedup.md) - relay_identity_mismatch、模型名重复拼接、claude-opus-5claude-opus-5、重复 message_start
- [pengine 阶段 attempt 跨恢复周期累计，3 次即终态失败且不可 continue](entries/pengine-stage-attempts-exhausted-not-continuable.md) - attempts_exhausted、paused/continue 反复中断、failed run 无法恢复、MAX_STAGE_ATTEMPTS
- [本项目自建 Langfuse v4 用 v2 observations API 查询轨迹](entries/langfuse-v4-observations-api.md) - Langfuse API 404、/api/public/traces 不可用、查询模型调用轨迹、observations
- [pengine 语言修复三站点中仍有两处是全量指纹校验](entries/pengine-language-repair-remaining-fingerprint-sites.md) - 语言修复被拒、锁定字段被改写整体拒绝、语义审查/修复子代理修复循环
