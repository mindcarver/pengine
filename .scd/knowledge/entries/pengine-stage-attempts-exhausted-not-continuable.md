---
status: active
scope: project
origin: 2026-08-08 全流程排障（run 803d1a84）
updated: 2026-08-08
---

# pengine 阶段 attempt 跨恢复周期累计，3 次即终态失败且不可 continue

- Trigger: run 因 relay 中断进入 paused，continue 后同一阶段再次中断；或试图对 state=failed 的 run 调 continue。
- Guidance: `MAX_STAGE_ATTEMPTS = 3`（src/pengine/repository.py:96）按阶段委派计数且跨 pause/resume 累计——同一阶段被打断 3 次即 `attempts_exhausted`，run 进入终态 failed。failed 是死路：`continue_run`（src/pengine/repository.py:5586）只接受 paused。因此同一阶段出现第 2 次中断时应先修根因（如超时、上游故障），不要继续 continue 烧掉最后一次 attempt；已 failed 只能新建 creation 重跑。
- Boundary: 不适用于内容审核拒绝通道（pause_content_rejection，另有 repair_rounds 语义）。
- Evidence: run 803d1a84 同一阶段 3 次 relay 408 → failure code `attempts_exhausted`（attempt_count=3）→ state failed，continue 端点不可用。
- Source: src/pengine/repository.py:96、:5586；run 803d1a84 的失败载荷。
