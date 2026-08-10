---
status: active
scope: project
origin: 2026-08-08 语言门修复（run 8fc9078a 失败链分析）
updated: 2026-08-08
---

# pengine 语言修复三站点中仍有两处是全量指纹校验

- Trigger: 语言修复被拒绝，报错含"语言修复改变了已锁定的结构化事实或审核结论"之类的锁定字段校验失败。
- Guidance: 只有 `_call_structured_stage`（src/pengine/agents.py）改成了"确定性剥离注释 → 模型仅翻译 → 结构保持合并"（`_strip_language_glosses` / `_merge_language_repair`，模型翻错锁定字段只会被忽略、不会整体拒绝）。另两处——`_invoke_semantic_reviewer`（agents.py ~3560）和修复子代理路径（~3717）——仍是全量指纹比对：修复模型重写任何锁定字段就整体拒绝并重试。在这两处看到该报错反复出现时，应移植合并式修复，而不是调 prompt 劝模型别改。
- Boundary: 语言门中性值判定修复（language.py，无字母的时间/区间值不再误判）已大幅降低这两处的触发概率；未触发就无需改动。
- Evidence: run 8fc9078a 的失败链（"21:00—22:00" 假阳性 → 修复即重写 → 指纹拒绝 → 循环耗尽）；`_call_structured_stage` 新合并逻辑及 tests/test_agents.py 中的行为测试（333 通过）。
- Source: src/pengine/agents.py（`_merge_language_repair` 约 1273-1370 行、except 分支 3519-3643 行、~3560、~3717）。
