---
status: active
scope: project
origin: 2026-08-08 全流程排障（run 803d1a84 / 5fdcfa79）
updated: 2026-08-08
---

# pengine relay 对非流式请求有约 300 秒硬超时

- Trigger: generation 模型调用在恰好 ~300.3s 处失败，relay 返回 408 `timeout_error`（safe_message "请求超时"）；同一阶段重试仍在相同时点死亡。
- Guidance: 这是 relay 对单个非流式响应的硬上限，不是偶发网络问题。长调用（分集大纲/剧本，正常耗时可 >240s）必然撞死。解法是让 generation 适配器走流式：`_SerialChatAnthropic(streaming=True)`（src/pengine/relay.py:987 附近），SSE 持续有字节即不触发该超时；`model_timeout_seconds` 随之从整响应超时变为块间超时。流式聚合对 `ainvoke` 调用方透明，但模型身份并非天然安全：重复身份起始块会被 LangChain 拼接。generation 流必须保留 `_SerialChatAnthropic` 的同值身份去重，并让不同、缺失或单块拼接身份继续 fail-closed。
- Boundary: 不适用于 <240s 的短调用（非流式也安全）；review 路由（gpt-5.5 / `_SerialChatOpenAI`）未观察到此问题，未改流式。
- Evidence: run 803d1a84 在 generating_episode_outline 三次 attempt 均死于 ~300.3s/408；改 `streaming=True` 后 run 5fdcfa79 一笔 405.3s 调用成功，全流程 6 阶段跑通（79 次调用 0 失败）。
- Source: src/pengine/relay.py（generation 适配器 `streaming=True` 及注释）；两次 run 的 model_calls 失败载荷。
