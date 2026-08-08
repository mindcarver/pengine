---
status: active
scope: project
origin: live-e2e 诊断（issue #78）
updated: 2026-08-07
---

# pengine 语义审查子智能体没有内部模型调用上限

- Trigger（触发）：pengine 通过子智能体调用的语义审查者（如 canon_reviewer）对一个大而复杂的候选产物进行评审时，始终无法返回合法的结构化结果；评审循环表现为大量 review 模型调用却从不触发 rewrite 或写 checkpoint，整个阶段卡住。
- Guidance（结论与做法）：子智能体自身的收敛循环并不被 pengine 约束。`_STRUCTURED_VALIDATION_FAILURE_LIMIT = 3`（agents.py:780）只限制 pengine 在 `_call_structured_stage` 里首次生成的结构化输出重试，并不约束子智能体评审者的内部重试。当评审者无法收敛时，应怀疑候选产物超出了模型稳定输出结构化数据的能力，进而拆分评审任务（例如按段落分别评审），或在 `_invoke_semantic_reviewer` 外层加显式调用上限。outline 阶段（单一小段落）通常 1-2 次 review 即收敛；c+r 阶段（双段落、需交叉一致性）则不然。
- Boundary（不适用情形）：仅适用于 story-artifact 评审/修复循环 `_generate_consistent_story_artifact` 中通过子智能体调用的语义审查者。不适用于 `_call_structured_stage` 的首次生成——后者自带错误反馈重试与上限。
- Evidence（最小证据）：live e2e 运行 20260806T145935Z-08717da0——c+r 阶段 1 次生成（成功）、25 次 review 调用（均 finish_reason=tool_calls，最大 11942 tok）、0 次 rewrite、checkpoint 未写入；同一流程 outline 阶段正常收敛。issue #78（OPEN）。
- Source（来源）：src/pengine/agents.py:780、~2576、~3263；gh issue 78；.artifacts/live-e2e/20260806T145935Z-08717da0/
