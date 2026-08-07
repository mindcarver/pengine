# 贡献指南

感谢贡献 Pengine。项目当前是本地单操作员短剧创作引擎；贡献应保持这个边界清晰，并优先补齐可验证的工程行为。

## 开始之前

- 先读 [GitHub Pages 工程文档](https://mindcarver.github.io/pengine/)。
- 对 API、SQLite、模型路由、重试分类或敏感数据的改动，先在 Issue 中说明不变量和影响范围。
- 不要提交 `.env`、`data/`、真实模型 prompt/response、用户故事或未授权 persona 内容。

## 本地验证

```bash
uv sync --locked --all-groups
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv build --no-sources
```

真实模型和浏览器验收属于额外证据，不能用确定性假模型测试代替；见 `tests/live/README.md`。

## PR 要求

请说明：

1. 真实运行路径、修改原因和不变量；
2. 新增或调整的测试；
3. API/schema/migration/model-route/retry 行为是否变化；
4. 哪些验证已执行，哪些需要真实 relay 或人工浏览器验收；
5. 是否需要同步更新 `README.md`、`contracts/` 或 `docs/`。

保持变更小而完整：不要重构无关模块，不要用 UI 文案替代后端状态，不要把 HTTP 成功或草稿产生写成正式交付。
