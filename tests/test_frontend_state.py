import json
import subprocess
from pathlib import Path


def test_revision_submission_controls_are_single_use() -> None:
    script_path = Path(__file__).parents[1] / "src" / "pengine" / "web" / "app.js"
    assertions = """
Object.assign(elements, {
  feedback: { disabled: false, value: "加强结尾行动。", focus() {} },
  "revision-button": { disabled: false, textContent: "" },
  "revision-form": { setAttribute() {} },
  "revision-message": { textContent: "" },
  "feedback-state": { textContent: "" },
  "revision-description": { textContent: "" },
});

state.creation = { revision: { state: "available" } };
state.pendingFeedback = "加强结尾行动。";
setRevisionBusy(false);
if (!elements.feedback.disabled) throw new Error("feedback was unlocked");
if (!elements["revision-button"].disabled) throw new Error("button was unlocked");
if (elements["revision-button"].textContent !== "意见已冻结") {
  throw new Error("accepted revision was presented as submit-ready");
}

state.pendingFeedback = "";
setRevisionBusy(false);
if (elements.feedback.disabled) throw new Error("available feedback stayed locked");
if (elements["revision-button"].disabled) throw new Error("available button stayed locked");

state.creation = { revision: { state: "failed" } };
setRevisionBusy(false);
if (!elements["revision-button"].disabled) throw new Error("failed revision was retryable");
if (elements["revision-button"].textContent !== "修订失败") {
  throw new Error("failed revision exposed a retry label");
}

state.creation = { revision: { state: "succeeded" } };
setRevisionBusy(false);
if (elements["revision-button"].textContent !== "修订已完成") {
  throw new Error("completed revision exposed a submit label");
}

const event = { preventDefault() {} };
let postCount = 0;
apiRequest = async () => {
  postCount += 1;
  await Promise.resolve();
  return {};
};
refreshCreation = async () => true;
state.creation = { revision: { state: "available" } };
state.creationId = "creation-id";
state.pendingFeedback = "";
await Promise.all([handleRevision(event), handleRevision(event)]);
if (postCount !== 1) throw new Error(`expected one POST, received ${postCount}`);

apiRequest = async () => {
  throw new ApiError("连接中断", "network_error", 0);
};
refreshCreation = async () => false;
state.creation = { revision: { state: "available" } };
state.pendingFeedback = "";
await handleRevision(event);
if (!state.pendingFeedback) throw new Error("ambiguous submission was unlocked");
if (!elements["revision-button"].disabled) {
  throw new Error("ambiguous submission exposed another POST");
}

apiRequest = async () => {
  throw new ApiError("请求被拒绝", "invalid_request", 422);
};
refreshCreation = async () => true;
state.creation = { revision: { state: "available" } };
state.pendingFeedback = "";
await handleRevision(event);
if (state.pendingFeedback) throw new Error("definitive rejection stayed frozen");
if (elements["revision-button"].disabled) {
  throw new Error("definitively rejected submission stayed locked");
}
"""
    harness = f"""
const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync({json.dumps(str(script_path))}, "utf8");
const context = {{
  document: {{ addEventListener() {{}} }},
  window: {{}},
  crypto: {{ randomUUID() {{ return "test-id"; }} }},
  console,
}};
const result = vm.runInNewContext(
  source + "\\n(async () => {{" + {json.dumps(assertions)} + "}})()",
  context,
);
Promise.resolve(result).catch((error) => {{
  console.error(error);
  process.exitCode = 1;
}});
"""

    subprocess.run(
        ["node", "-e", harness],
        check=True,
        capture_output=True,
        text=True,
    )


def test_terminal_initial_failure_guides_a_new_submission_without_a_request() -> None:
    root = Path(__file__).parents[1]
    script_path = root / "src" / "pengine" / "web" / "app.js"
    page = (root / "src" / "pengine" / "web" / "index.html").read_text()
    assert 'id="failure-guidance"' in page
    assert 'id="start-new-creation"' in page

    assertions = """
const storage = new Map([[STORAGE_KEY, "failed-creation"]]);
let requestCount = 0;
let storyFocused = false;
let formScrolled = false;
Object.assign(elements, {
  "task-waiting": { hidden: false },
  "failure-panel": { hidden: true },
  "result-workspace": { hidden: false },
  "failure-label": { textContent: "" },
  "failure-title": { textContent: "" },
  "failure-message": { textContent: "" },
  "failure-guidance": { hidden: true, textContent: "" },
  "failure-code": { textContent: "" },
  "failure-actions": { hidden: true },
  "run-progress": { hidden: false },
  "delivery-section": { hidden: false },
  "creation-message": { textContent: "" },
  "creation-form": {
    scrollIntoView() { formScrolled = true; },
  },
  story: {
    focus() { storyFocused = true; },
  },
});
window.localStorage = {
  getItem(key) { return storage.get(key) || null; },
  setItem(key, value) { storage.set(key, value); },
  removeItem(key) { storage.delete(key); },
};
apiRequest = async () => {
  requestCount += 1;
  return {};
};
state.creationId = "failed-creation";
state.creation = { initial: { state: "failed" }, revision: { state: "unavailable" } };
showFailure(
  { code: "internal_error", message: "The workflow failed safely." },
  "初稿生成失败",
  { canStartNewCreation: true },
);
if (elements["failure-guidance"].hidden) throw new Error("initial failure hid recovery guidance");
if (!elements["failure-guidance"].textContent.includes("不会自动重试")) {
  throw new Error("initial failure did not explain terminal state");
}
if (elements["failure-actions"].hidden) throw new Error("initial failure hid start action");

startNewCreation();
if (state.creationId || state.creation) throw new Error("failed creation stayed selected");
if (storage.has(STORAGE_KEY)) throw new Error("failed creation id stayed in storage");
if (requestCount !== 0) throw new Error("start action made a request");
if (!storyFocused || !formScrolled) throw new Error("start action did not return focus to story");
if (!elements["creation-message"].textContent.includes("重新填写故事")) {
  throw new Error("start action did not explain the next step");
}

showFailure({ code: "internal_error", message: "failed" }, "修订生成失败");
if (!elements["failure-guidance"].hidden || !elements["failure-actions"].hidden) {
  throw new Error("non-initial failure exposed initial-run action");
}

renderProgress = () => {};
renderWorkspace = () => false;
state.activeDraftRunKind = "";
state.creationId = "terminal-creation";
state.creation = {
  persona: { display_name: "测试人格", version: "1" },
  initial: {
    state: "failed",
    failure: { code: "internal_error", message: "failed" },
  },
  revision: { state: "unavailable" },
};
renderCreation();
if (!elements["result-workspace"].hidden) {
  throw new Error("failed initial run exposed unapproved drafts");
}
state.creation.initial = { state: "ended" };
renderCreation();
if (!elements["result-workspace"].hidden) {
  throw new Error("ended initial run exposed unapproved drafts");
}

renderWorkspace = () => true;
state.creation = {
  persona: { display_name: "测试人格", version: "1" },
  initial: { state: "succeeded" },
  revision: {
    state: "failed",
    progress: {
      current_stage: "generating_episode_outline",
      completed_stages: ["determining_direction", "generating_story_outline"],
      elapsed_seconds: 1391,
      recovery_reason: "none",
      final_review: { l0: "pending", l4: "pending" },
      model_calls: [],
      can_continue: false,
      can_end: false,
    },
    failure: {
      code: "structured_output_invalid",
      message: "模型未返回有效的结构化结果。",
      failed_stage: "generating_episode_outline",
      attempt_count: 1,
    },
  },
};
renderCreation();
if (elements["failure-panel"].hidden) {
  throw new Error("failed revision stayed hidden behind the progress card");
}
if (elements["failure-title"].textContent !== "修订生成失败") {
  throw new Error("failed revision did not use a terminal title");
}
if (!elements["failure-message"].textContent.includes("失败阶段：生成分集大纲")) {
  throw new Error("failed revision did not expose its failed stage");
}
if (elements["failure-code"].textContent !== "错误代码：structured_output_invalid") {
  throw new Error("failed revision did not expose its error code");
}
if (elements["failure-guidance"].hidden !== true || elements["failure-actions"].hidden !== true) {
  throw new Error("failed revision exposed an unsupported recovery action");
}
if (elements["result-workspace"].hidden) {
  throw new Error("failed revision hid the retained readable workspace");
}
"""
    harness = f"""
const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync({json.dumps(str(script_path))}, "utf8");
const context = {{
  document: {{ addEventListener() {{}} }},
  window: {{}},
  crypto: {{ randomUUID() {{ return "test-id"; }} }},
  console,
}};
const result = vm.runInNewContext(
  source + "\\n(async () => {{" + {json.dumps(assertions)} + "}})()",
  context,
);
Promise.resolve(result).catch((error) => {{
  console.error(error);
  process.exitCode = 1;
}});
"""

    subprocess.run(
        ["node", "-e", harness],
        check=True,
        capture_output=True,
        text=True,
    )


def test_external_relay_failure_offers_inline_retry_control() -> None:
    root = Path(__file__).parents[1]
    script_path = root / "src" / "pengine" / "web" / "app.js"
    page = (root / "src" / "pengine" / "web" / "index.html").read_text()
    assert 'id="retry-run"' in page

    assertions = """
Object.assign(elements, {
  "task-waiting": { hidden: false },
  "failure-panel": { hidden: true },
  "result-workspace": { hidden: true },
  "failure-label": { textContent: "" },
  "failure-title": { textContent: "" },
  "failure-message": { textContent: "" },
  "failure-guidance": { hidden: true, textContent: "" },
  "failure-code": { textContent: "" },
  "failure-actions": { hidden: true },
  "retry-run": { hidden: true, disabled: false },
  "quality-rejection-details": { hidden: true },
});

state.creationId = "failed-relay-creation";
state.creation = { initial: { state: "failed" }, revision: { state: "unavailable" } };

showFailure(
  { code: "relay_unavailable", message: "The model relay request failed (HTTP 402)." },
  "初稿生成失败",
  { canStartNewCreation: true, canRetry: true },
);
if (elements["retry-run"].hidden) throw new Error("retryable failure hid the retry control");
if (!elements["failure-guidance"].textContent.includes("原样重试")) {
  throw new Error("retryable failure did not explain the inline retry");
}

showFailure(
  { code: "internal_error", message: "The workflow failed safely." },
  "初稿生成失败",
  { canStartNewCreation: true },
);
if (!elements["retry-run"].hidden) {
  throw new Error("non-retryable failure exposed the retry control");
}
if (!elements["failure-guidance"].textContent.includes("不会自动重试")) {
  throw new Error("non-retryable failure lost its terminal guidance");
}

state.runControlBusy = true;
showFailure(
  { code: "relay_unavailable", message: "failed" },
  "初稿生成失败",
  { canStartNewCreation: true, canRetry: true },
);
if (!elements["retry-run"].disabled) throw new Error("busy workbench left retry enabled");
state.runControlBusy = false;

let posted = "";
apiRequest = async (url, options) => {
  posted = `${url}:${options.method}:${options.headers["Idempotency-Key"]}`;
  return { run_state: "queued" };
};
let refreshed = false;
refreshCreation = async () => {
  refreshed = true;
};
renderProgress = () => {};
renderQualityRejectionControls = () => {};
await handleRunControl("retry", {
  runKind: "initial",
  messageElement: elements["failure-guidance"],
});
const expectedPost =
  "/creations/failed-relay-creation/runs/initial/retry:POST:web-run-initial-retry-test-id";
if (posted !== expectedPost) {
  throw new Error(`retry control posted unexpectedly: ${posted}`);
}
if (!elements["failure-guidance"].textContent.includes("正在从已批准进度重试")) {
  throw new Error("retry control did not announce the bounded retry");
}
if (!refreshed) throw new Error("retry control did not refresh the creation");
if (state.runControlBusy) throw new Error("retry control stayed busy");
"""
    harness = f"""
const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync({json.dumps(str(script_path))}, "utf8");
const context = {{
  document: {{ addEventListener() {{}} }},
  window: {{}},
  crypto: {{ randomUUID() {{ return "test-id"; }} }},
  console,
}};
const result = vm.runInNewContext(
  source + "\\n(async () => {{" + {json.dumps(assertions)} + "}})()",
  context,
);
Promise.resolve(result).catch((error) => {{
  console.error(error);
  process.exitCode = 1;
}});
"""

    subprocess.run(
        ["node", "-e", harness],
        check=True,
        capture_output=True,
        text=True,
    )


def test_workbench_uses_four_gated_creation_scenes() -> None:
    root = Path(__file__).parents[1]
    script_path = root / "src" / "pengine" / "web" / "app.js"
    page = (root / "src" / "pengine" / "web" / "index.html").read_text()
    assert 'id="flow-select-writer"' in page
    assert 'id="flow-tell-story"' in page
    assert 'id="flow-create"' in page
    assert 'id="flow-read-deliverables"' in page
    assert 'id="selection-view"' in page
    assert 'id="brief-view"' in page
    assert 'id="current-work-view"' in page
    assert 'class="series-drawer"' not in page
    assert page.index('id="selection-view"') < page.index('id="brief-view"')
    assert page.index('id="brief-view"') < page.index('id="current-work-view"')

    assertions = """
function sceneElement() {
  return {
    hidden: false,
    disabled: false,
    dataset: {},
    attrs: {},
    setAttribute(name, value) { this.attrs[name] = value; },
  };
}
const selectionView = sceneElement();
const briefView = sceneElement();
const currentView = sceneElement();
const progressScene = sceneElement();
const selectButton = sceneElement();
const briefButton = sceneElement();
const progressButton = sceneElement();
const readingButton = sceneElement();
const placeholder = sceneElement();
Object.assign(elements, {
  "selection-view": selectionView,
  "brief-view": briefView,
  "current-work-view": currentView,
  "progress-scene": progressScene,
  "flow-select-writer": selectButton,
  "flow-tell-story": briefButton,
  "flow-create": progressButton,
  "flow-read-deliverables": readingButton,
  "current-work-placeholder": placeholder,
  "brief-persona": { textContent: "" },
  "delivery-title": { textContent: "" },
});

state.creationId = "";
state.creation = null;
state.selectedPersonaId = "";
state.workspaceView = "reading";
renderWorkspaceViews();
if (state.workspaceView !== "selection") {
  throw new Error("empty workbench skipped writer selection");
}
if (selectionView.hidden || !briefView.hidden || !currentView.hidden) {
  throw new Error("empty workbench exposed the wrong scene");
}
if (!briefButton.disabled || !progressButton.disabled || !readingButton.disabled) {
  throw new Error("empty workbench enabled a gated scene");
}

state.selectedPersonaId = "wuzhen";
if (!setWorkspaceView("brief")) throw new Error("selected writer could not open story brief");
if (!selectionView.hidden || briefView.hidden || !currentView.hidden) {
  throw new Error("story brief did not replace writer selection");
}
if (briefButton.disabled || briefButton.attrs["aria-current"] !== "page") {
  throw new Error("story brief was not marked current");
}

state.creationId = "creation-id";
if (!setWorkspaceView("progress")) throw new Error("creation could not open live progress");
if (!selectionView.hidden || !briefView.hidden || currentView.hidden || progressScene.hidden) {
  throw new Error("live progress did not replace the brief");
}
if (readingButton.disabled) {
  // Expected until a formal result exists.
} else {
  throw new Error("unapproved work enabled deliverable reading");
}

state.creation = {
  initial: { state: "succeeded", result: { content_package: { story_outline: "ready" } } },
  revision: { state: "unavailable" },
};
if (!setWorkspaceView("reading")) throw new Error("formal delivery could not open reading scene");
if (currentView.hidden || !progressScene.hidden) {
  throw new Error("reading scene kept live progress visible");
}
if (readingButton.disabled || readingButton.attrs["aria-current"] !== "page") {
  throw new Error("formal delivery did not enable reading scene");
}
"""
    harness = f"""
const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync({json.dumps(str(script_path))}, "utf8");
const context = {{
  document: {{ addEventListener() {{}} }},
  window: {{}},
  crypto: {{ randomUUID() {{ return "test-id"; }} }},
  console,
}};
const result = vm.runInNewContext(
  source + "\\n(() => {{" + {json.dumps(assertions)} + "}})()",
  context,
);
Promise.resolve(result).catch((error) => {{
  console.error(error);
  process.exitCode = 1;
}});
"""

    subprocess.run(
        ["node", "-e", harness],
        check=True,
        capture_output=True,
        text=True,
    )


def test_creation_submission_opens_live_creation_scene() -> None:
    root = Path(__file__).parents[1]
    script_path = root / "src" / "pengine" / "web" / "app.js"

    assertions = """
const storage = new Map();
window.localStorage = {
  getItem(key) { return storage.get(key) || null; },
  setItem(key, value) { storage.set(key, value); },
  removeItem(key) { storage.delete(key); },
};
Object.assign(elements, {
  story: { value: "一段待创作的故事", focus() {} },
  requirements: { value: "" },
  "creation-message": { textContent: "" },
  "create-button": {
    disabled: false,
    querySelector() { return { textContent: "" }; },
  },
  "creation-form": { setAttribute() {} },
});
state.selectedPersonaId = "wuzhen";
state.workspaceView = "brief";
let deliveryFocused = false;
apiRequest = async (path, options) => {
  if (path !== "/creations" || options.method !== "POST") {
    throw new Error("creation used the wrong endpoint");
  }
  return { creation_id: "new-creation-id" };
};
renderCreation = () => {};
refreshCreation = async () => true;
focusDelivery = () => { deliveryFocused = state.workspaceView === "progress"; };

await handleCreate({ preventDefault() {} });
if (state.creationId !== "new-creation-id") throw new Error("creation id was not retained");
if (storage.get(STORAGE_KEY) !== "new-creation-id") throw new Error("creation id was not saved");
if (state.workspaceView !== "progress") throw new Error("submission did not open live creation");
if (!deliveryFocused) throw new Error("submission did not focus live creation");
"""
    harness = f"""
const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync({json.dumps(str(script_path))}, "utf8");
const context = {{
  document: {{ addEventListener() {{}} }},
  window: {{}},
  crypto: {{ randomUUID() {{ return "test-id"; }} }},
  console,
}};
const result = vm.runInNewContext(
  source + "\\n(async () => {{" + {json.dumps(assertions)} + "}})()",
  context,
);
Promise.resolve(result).catch((error) => {{
  console.error(error);
  process.exitCode = 1;
}});
"""

    subprocess.run(
        ["node", "-e", harness],
        check=True,
        capture_output=True,
        text=True,
    )


def test_quality_rejection_retains_workspace_and_retries_the_final_review() -> None:
    root = Path(__file__).parents[1]
    script_path = root / "src" / "pengine" / "web" / "app.js"
    page = (root / "src" / "pengine" / "web" / "index.html").read_text()
    assert 'id="quality-rejection-details"' in page
    assert 'id="retry-final-review"' in page
    assert 'id="end-quality-rejected-run"' in page
    assert "按证据修复并重新审核" in page

    assertions = """
const requests = [];
Object.assign(elements, {
  "delivery-section": { hidden: true },
  "task-waiting": { hidden: false },
  "failure-panel": { hidden: true },
  "result-workspace": { hidden: true },
  "failure-label": { textContent: "" },
  "failure-title": { textContent: "" },
  "failure-message": { textContent: "" },
  "failure-guidance": { hidden: true, textContent: "" },
  "quality-rejection-details": { hidden: true },
  "quality-rejection-stage": { textContent: "" },
  "quality-rejection-evidence": { textContent: "" },
  "quality-rejection-repair": { textContent: "" },
  "quality-rejection-attempt": { textContent: "" },
  "failure-code": { textContent: "" },
  "failure-actions": { hidden: false },
  "quality-rejection-actions": { hidden: true },
  "retry-final-review": { hidden: false, disabled: false },
  "end-quality-rejected-run": { disabled: false },
  "quality-rejection-action-message": { textContent: "" },
});
renderProgress = () => {};
renderWorkspace = () => true;
state.creationId = "creation id";
state.activeDraftRunKind = "";
state.creation = {
  persona: { display_name: "测试人格", version: "1" },
  initial: {
    state: "quality_rejected",
    drafts: {
      artifacts: [{ stage: "generating_story_outline", content: "保留的故事大纲" }],
    },
    quality_rejection: {
      stage: "accepting_l0",
      evidence: "人物动机没有落实到行动。",
      attempt_count: 2,
      can_retry: true,
      repair_plan: null,
      repair_state: "available",
    },
  },
  revision: { state: "unavailable" },
};
renderCreation();
if (elements["failure-panel"].hidden) throw new Error("quality rejection stayed hidden");
if (elements["result-workspace"].hidden) {
  throw new Error("quality rejection hid persisted workspace");
}
if (!elements["failure-title"].textContent.includes("L0 创作内核")) {
  throw new Error("L0 rejection was not identified");
}
if (!elements["quality-rejection-evidence"].textContent.includes("人物动机没有落实到行动")) {
  throw new Error("reviewer evidence was not rendered");
}
if (!elements["quality-rejection-repair"].textContent.includes("绑定到具体剧集原文")) {
  throw new Error("legacy evidence-binding repair was not explained");
}
if (elements["quality-rejection-attempt"].textContent !== "审核尝试：第 2 次") {
  throw new Error("attempt count was not rendered");
}
if (!elements["failure-actions"].hidden) {
  throw new Error("quality rejection exposed start-new-creation");
}
if (elements["quality-rejection-actions"].hidden) {
  throw new Error("quality rejection hid retry and end actions");
}

apiRequest = async (path, options) => {
  requests.push({ path, options });
  await Promise.resolve();
  return {};
};
refreshCreation = async () => true;
await Promise.all([
  handleQualityRejectionControl("retry-final-review"),
  handleQualityRejectionControl("retry-final-review"),
]);
if (requests.length !== 1) throw new Error(`expected one retry POST, received ${requests.length}`);
if (requests[0].path !== "/creations/creation%20id/runs/initial/retry-final-review") {
  throw new Error(`wrong retry endpoint: ${requests[0].path}`);
}
if (requests[0].options.method !== "POST") throw new Error("retry was not a POST");
if (
  !requests[0].options.headers["Idempotency-Key"]?.startsWith(
    "web-run-initial-retry-final-review-",
  )
) {
  throw new Error("retry omitted its idempotency key");
}

await handleQualityRejectionControl("end");
if (requests.length !== 2) throw new Error("quality rejection did not expose an end action");
if (requests[1].path !== "/creations/creation%20id/runs/initial/end") {
  throw new Error(`wrong end endpoint: ${requests[1].path}`);
}
if (!requests[1].options.headers["Idempotency-Key"]?.startsWith("web-run-initial-end-")) {
  throw new Error("end omitted its idempotency key");
}

state.creation = {
  persona: { display_name: "测试人格", version: "1" },
  initial: { state: "succeeded" },
  revision: {
    state: "quality_rejected",
    quality_rejection: {
      stage: "accepting_l4",
      evidence: null,
      attempt_count: 3,
      can_retry: false,
    },
  },
};
showQualityRejection(qualityRejectedRun(), { showWorkspace: true });
if (!elements["quality-rejection-stage"].textContent.includes("L4 技法与价值观")) {
  throw new Error("L4 rejection was not identified");
}
if (!elements["quality-rejection-evidence"].textContent.includes("旧版本任务未保存审核证据")) {
  throw new Error("legacy no-evidence state was not explained");
}
if (!elements["retry-final-review"].hidden || !elements["retry-final-review"].disabled) {
  throw new Error("exhausted quality gate still offered retry");
}
if (!elements["failure-guidance"].textContent.includes("三次上限")) {
  throw new Error("exhausted quality gate did not explain the retry limit");
}

state.creation = {
  persona: { display_name: "测试人格", version: "1" },
  initial: { state: "succeeded" },
  revision: { state: "unavailable" },
};
renderQualityRejectionControls();
if (
  !elements["quality-rejection-actions"].hidden ||
  !elements["retry-final-review"].hidden ||
  !elements["retry-final-review"].disabled ||
  !elements["end-quality-rejected-run"].disabled
) {
  throw new Error("quality rejection controls remained interactive without a rejected run");
}
"""
    harness = f"""
const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync({json.dumps(str(script_path))}, "utf8");
const context = {{
  document: {{ addEventListener() {{}} }},
  window: {{ confirm() {{ return true; }} }},
  crypto: {{ randomUUID() {{ return "test-id"; }} }},
  console,
}};
const result = vm.runInNewContext(
  source + "\\n(async () => {{" + {json.dumps(assertions)} + "}})()",
  context,
);
Promise.resolve(result).catch((error) => {{
  console.error(error);
  process.exitCode = 1;
}});
"""

    subprocess.run(
        ["node", "-e", harness],
        check=True,
        capture_output=True,
        text=True,
    )


def test_initial_and_revision_share_authoritative_progress_component() -> None:
    root = Path(__file__).parents[1]
    script_path = root / "src" / "pengine" / "web" / "app.js"
    page = (root / "src" / "pengine" / "web" / "index.html").read_text()
    assert page.count('id="run-progress"') == 1
    assert "确定创作方向" in page
    assert "L0 创作内核" in page
    assert "L4 技法与价值观" in page

    assertions = """
const stageItems = USER_STAGES.map(([stage]) => ({
  dataset: { stage },
  attrs: {},
  setAttribute(name, value) { this.attrs[name] = value; },
  removeAttribute(name) { delete this.attrs[name]; },
}));
Object.assign(elements, {
  "run-progress": { hidden: true },
  "progress-kind": { textContent: "" },
  "progress-title": { textContent: "" },
  "progress-elapsed": { textContent: "" },
  "progress-stages": { querySelectorAll() { return stageItems; } },
  "review-progress": { hidden: true },
  "review-l0": { textContent: "" },
  "review-l4": { textContent: "" },
  "run-controls": { hidden: true },
  "run-control-title": { textContent: "" },
  "run-control-description": { textContent: "" },
  "continue-run": { disabled: false },
  "end-run": { disabled: false },
  "run-control-message": { textContent: "" },
  "episode-progress": { hidden: true },
  "episode-progress-label": { textContent: "" },
  "episode-progress-detail": { textContent: "" },
  "model-call-panel": { hidden: true },
  "model-call-totals": { textContent: "" },
  "model-call-list": { replaceChildren() {} },
});
const progress = {
  current_stage: "generating_story_outline",
  completed_stages: ["determining_direction"],
  elapsed_seconds: 125,
  recovery_state: "none",
  recovery_reason: "none",
  final_review: { l0: "pending", l4: "pending" },
  can_continue: false,
  can_end: false,
};
state.creation = {
  initial: { state: "running", progress },
  revision: { state: "unavailable" },
};
renderProgress();
if (elements["run-progress"].hidden) throw new Error("initial progress stayed hidden");
if (elements["progress-kind"].textContent !== "初稿进度") throw new Error("wrong run kind");
if (elements["progress-title"].textContent !== "生成故事大纲") throw new Error("wrong stage");
if (elements["progress-elapsed"].textContent !== "02:05") throw new Error("wrong elapsed");
if (stageItems[0].dataset.status !== "completed") throw new Error("completed stage lost");
if (stageItems[1].dataset.status !== "current") throw new Error("current stage lost");

state.creation = {
  initial: { state: "succeeded", progress },
  revision: {
    state: "failed",
    progress: { ...progress, current_stage: "generating_episode_outline" },
  },
};
renderProgress();
if (elements["progress-title"].textContent !== "失败于生成分集大纲") {
  throw new Error("terminal revision was still presented as active generation");
}

state.creation = {
  initial: { state: "succeeded", progress },
  revision: {
    state: "paused",
    progress: {
      ...progress,
      current_stage: "final_review",
      completed_stages: USER_STAGES.slice(0, 4).map(([stage]) => stage),
      recovery_state: "paused",
      recovery_reason: "run_timeout",
      final_review: { l0: "passed", l4: "paused" },
      can_continue: true,
      can_end: true,
    },
  },
};
renderProgress();
if (elements["progress-kind"].textContent !== "修订进度") throw new Error("revision not reused");
if (elements["run-controls"].hidden) throw new Error("paused controls stayed hidden");
if (!elements["run-control-title"].textContent.includes("整体运行时限")) {
  throw new Error("timeout pause control did not identify the timeout reason");
}
if (elements["review-progress"].hidden) throw new Error("review substatus stayed hidden");
if (!elements["review-l0"].textContent.includes("已通过")) throw new Error("L0 status lost");
if (!elements["review-l4"].textContent.includes("已暂停")) throw new Error("L4 status lost");
if (shouldPoll()) throw new Error("paused run kept polling");

state.creation.revision.progress.recovery_reason = "relay_interruption";
renderProgress();
if (!elements["run-control-title"].textContent.includes("网络 / Relay")) {
  throw new Error("relay pause control did not identify the relay interruption");
}
if (!elements["run-control-description"].textContent.includes("已提交草稿")) {
  throw new Error("relay pause control did not retain draft guidance");
}

state.creation.revision.progress.recovery_reason = "content_rejected";
state.creation.revision.pause = {
  code: "content_rejected",
  message: "人物知识状态仍与锁定合同冲突。",
  stage: "generating_episode_scripts",
  content_repair_count: 2,
  episode_number: 3,
};
renderProgress();
if (!elements["run-control-title"].textContent.includes("两轮修复")) {
  throw new Error("content rejection did not identify the repair boundary");
}
if (!elements["run-control-description"].textContent.includes("人物知识状态")) {
  throw new Error("content rejection did not expose review evidence");
}

state.creation.revision.progress.recovery_reason = "repair_authorization";
state.creation.revision.authorization = {
  kind: "suffix_rewrite",
  earliest_affected_episode: 2,
  range_episodes: 2,
  estimated_tokens: 1200,
};
state.creation.revision.pause = {
  code: "repair_authorization_required",
  message: "最新前缀仍与硬约束冲突。",
  stage: "generating_episode_scripts",
};
renderProgress();
if (elements["run-control-title"].textContent !== "分集硬约束需修复 · 等待授权") {
  throw new Error("repair authorization title did not identify a hard constraint");
}
if (!elements["run-control-description"].textContent.includes("最新审查证据")) {
  throw new Error("repair authorization did not describe latest review evidence");
}
if (!elements["run-control-description"].textContent.includes("参考上下文量")) {
  throw new Error("repair authorization omitted its reference context amount");
}
if (!elements["run-control-description"].textContent.includes("不是下限、整轮用量或费用预测")) {
  throw new Error("repair authorization presented reference context as a usage forecast");
}
if (elements["run-control-description"].textContent.includes("同一证据")) {
  throw new Error("repair authorization promised stale evidence reuse");
}

state.creation.revision.progress.recovery_reason = "episode_error";
state.creation.revision.progress.current_stage = "generating_episode_scripts";
state.creation.revision.progress.episodes = { total: 10, completed: 7, current: 8 };
state.creation.revision.pause = {
  code: "episode_error",
  message: "算术工具收到非十进制参数。",
  stage: "generating_episode_scripts",
  episode_number: 8,
};
renderProgress();
if (elements["run-control-title"].textContent !== "第 8 集可继续") {
  throw new Error("episode error did not identify the resumable episode");
}
if (!elements["run-control-description"].textContent.includes("非十进制参数")) {
  throw new Error("episode error hid its safe cause");
}
if (!elements["run-control-description"].textContent.includes("已完成的 7 集不会重新生成")) {
  throw new Error("episode error did not preserve completed drafts");
}
if (elements["continue-run"].textContent !== "从第 8 集继续") {
  throw new Error("episode error exposed the wrong continuation action");
}

state.creation.revision.state = "auto_resuming";
state.creation.revision.progress.recovery_reason = "relay_interruption";
state.creation.revision.progress.can_continue = false;
state.creation.revision.progress.can_end = false;
if (!shouldPoll()) throw new Error("auto recovery was not polled");

let postCount = 0;
apiRequest = async () => {
  postCount += 1;
  await Promise.resolve();
  return {};
};
refreshCreation = async () => true;
state.creation.revision.state = "paused";
state.creation.revision.progress.can_continue = true;
state.creation.revision.progress.can_end = true;
state.creationId = "creation-id";
state.progressRunKind = "revision";
await Promise.all([handleRunControl("continue"), handleRunControl("continue")]);
if (postCount !== 1) throw new Error(`expected one control POST, received ${postCount}`);
"""
    harness = f"""
const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync({json.dumps(str(script_path))}, "utf8");
const context = {{
  document: {{ addEventListener() {{}} }},
  window: {{ confirm() {{ return true; }} }},
  crypto: {{ randomUUID() {{ return "test-id"; }} }},
  console,
}};
const result = vm.runInNewContext(
  source + "\\n(async () => {{" + {json.dumps(assertions)} + "}})()",
  context,
);
Promise.resolve(result).catch((error) => {{
  console.error(error);
  process.exitCode = 1;
}});
"""

    subprocess.run(
        ["node", "-e", harness],
        check=True,
        capture_output=True,
        text=True,
    )


def test_completed_stage_controls_open_their_persisted_artifacts() -> None:
    root = Path(__file__).parents[1]
    script_path = root / "src" / "pengine" / "web" / "app.js"

    assertions = """
function button(key, group) {
  return {
    id: `${group}-${key}`,
    dataset: { [group]: key },
    attrs: {},
    hidden: false,
    disabled: false,
    tabIndex: 0,
    number: { textContent: "" },
    setAttribute(name, value) { this.attrs[name] = value; },
    removeAttribute(name) { delete this.attrs[name]; },
    querySelector(selector) { return selector === "span" ? this.number : null; },
  };
}
const stageItems = USER_STAGES.map(([stage]) => button(stage, "stage"));
const artifactButtons = [
  button("direction", "artifact"),
  button("story_outline", "artifact"),
  button("character_relationships", "artifact"),
  button("episode_outline", "artifact"),
  button("episode_scripts", "artifact"),
];
Object.assign(elements, {
  "run-progress": { hidden: true },
  "progress-kind": { textContent: "" },
  "progress-title": { textContent: "" },
  "progress-elapsed": { textContent: "" },
  "progress-stages": { querySelectorAll() { return stageItems; } },
  "review-progress": { hidden: true },
  "review-l0": { textContent: "" },
  "review-l4": { textContent: "" },
  "run-controls": { hidden: true },
  "run-control-title": { textContent: "" },
  "run-control-description": { textContent: "" },
  "continue-run": { hidden: false, disabled: false },
  "end-run": { hidden: false, disabled: false },
  "run-control-message": { textContent: "" },
  "episode-progress": { hidden: true },
  "episode-progress-label": { textContent: "" },
  "episode-progress-detail": { textContent: "" },
  "model-call-panel": { hidden: true },
  "model-call-totals": { textContent: "" },
  "model-call-list": { replaceChildren() {} },
  "result-workspace": { hidden: true, dataset: {} },
  "revision-desk": { hidden: false },
  "version-initial": button("initial", "version"),
  "version-revision": button("revision", "version"),
  "version-tabs": {
    querySelectorAll() { return [elements["version-initial"], elements["version-revision"]]; },
  },
  "version-note": { textContent: "" },
  "export-delivery": { hidden: true, disabled: true },
  "artifact-tabs": {
    querySelectorAll() { return artifactButtons; },
    querySelector(selector) {
      const match = selector.match(/data-artifact="([^"]+)"/);
      return match ? artifactButtons.find((item) => item.dataset.artifact === match[1]) : null;
    },
  },
  "artifact-panel": { setAttribute() {}, focus() {} },
  "artifact-overline": { textContent: "" },
  "artifact-title": { textContent: "" },
  "artifact-version-mark": { textContent: "" },
  "episode-navigator": { hidden: true },
  "episode-progress-summary": { textContent: "" },
  "episode-tabs": { replaceChildren() {} },
  "episode-content": { textContent: "", setAttribute() {}, removeAttribute() {} },
  "artifact-content": { hidden: false, textContent: "" },
});
state.workspaceView = "progress";
state.activeVersion = "initial";
state.activeArtifact = "direction";
state.creation = {
  initial: {
    state: "running",
    progress: {
      current_stage: "generating_story_outline",
      completed_stages: ["determining_direction"],
      elapsed_seconds: 4,
      recovery_state: "none",
      recovery_reason: "none",
      final_review: { l0: "pending", l4: "pending" },
      can_continue: false,
      can_end: false,
    },
    drafts: {
      artifacts: [
        {
          stage: "determining_direction",
          selected_l0_variant: "归返",
          selection_rationale: "匹配故事母题。",
        },
        { stage: "generating_story_outline", content: "已提交故事大纲" },
      ],
    },
  },
  revision: { state: "unavailable" },
};
renderProgress();
if (stageItems[0].disabled || stageItems[1].disabled) {
  throw new Error("persisted stages were not selectable");
}
if (stageItems[2].disabled !== true) {
  throw new Error("unpersisted stages became selectable");
}
handleStageClick({ target: { closest() { return stageItems[1]; } } });
if (state.activeArtifact !== "story_outline") {
  throw new Error("stage selection did not choose its artifact");
}
if (elements["result-workspace"].hidden) {
  throw new Error("stage selection did not open the creation reader");
}
if (elements["artifact-content"].textContent !== "已提交故事大纲") {
  throw new Error("stage selection did not render persisted content");
}
"""
    harness = f"""
const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync({json.dumps(str(script_path))}, "utf8");
const context = {{
  document: {{ addEventListener() {{}} }},
  window: {{}},
  crypto: {{ randomUUID() {{ return "test-id"; }} }},
  console,
}};
const result = vm.runInNewContext(
  source + "\\n(() => {{" + {json.dumps(assertions)} + "}})()",
  context,
);
Promise.resolve(result).catch((error) => {{
  console.error(error);
  process.exitCode = 1;
}});
"""

    subprocess.run(
        ["node", "-e", harness],
        check=True,
        capture_output=True,
        text=True,
    )


def test_live_drafts_render_in_the_creation_scene_but_not_the_formal_reader() -> None:
    root = Path(__file__).parents[1]
    script_path = root / "src" / "pengine" / "web" / "app.js"
    page = (root / "src" / "pengine" / "web" / "index.html").read_text()
    assert 'data-artifact="direction"' in page
    assert 'aria-label="选择文稿类别"' in page

    assertions = """
function button(key, prefix) {
  return {
    id: `${prefix}-${key}`,
    dataset: { [prefix === "version" ? "version" : "artifact"]: key },
    attrs: {},
    hidden: false,
    disabled: false,
    tabIndex: 0,
    number: { textContent: "" },
    setAttribute(name, value) { this.attrs[name] = value; },
    querySelector(selector) { return selector === "span" ? this.number : null; },
  };
}
const versionButtons = [button("initial", "version"), button("revision", "version")];
const artifactButtons = [
  button("direction", "artifact"),
  button("story_outline", "artifact"),
  button("character_relationships", "artifact"),
  button("episode_outline", "artifact"),
  button("episode_scripts", "artifact"),
];
Object.assign(elements, {
  "delivery-section": { hidden: true },
  "task-waiting": { hidden: true },
  "wait-kicker": { textContent: "" },
  "wait-title": { textContent: "" },
  "wait-description": { textContent: "" },
  "failure-panel": { hidden: true },
  "result-workspace": { hidden: true },
  "revision-desk": { hidden: false },
  "version-initial": versionButtons[0],
  "version-revision": versionButtons[1],
  "version-tabs": { querySelectorAll() { return versionButtons; } },
  "version-note": { textContent: "" },
  "export-delivery": { hidden: true, disabled: true },
  "artifact-tabs": {
    querySelectorAll() { return artifactButtons; },
    querySelector(selector) {
      const match = selector.match(/data-artifact="([^"]+)"/);
      return match ? artifactButtons.find((item) => item.dataset.artifact === match[1]) : null;
    },
  },
  "artifact-panel": { setAttribute() {} },
  "artifact-overline": { textContent: "" },
  "artifact-title": { textContent: "" },
  "artifact-version-mark": { textContent: "" },
  "episode-navigator": { hidden: true },
  "episode-progress-summary": { textContent: "" },
  "episode-tabs": { replaceChildren() {} },
  "episode-content": { textContent: "", setAttribute() {}, removeAttribute() {} },
  "artifact-content": { textContent: "" },
});
function episodeTab() {
  return {
    dataset: {},
    disabled: false,
    tabIndex: 0,
    textContent: "",
    setAttribute() {},
  };
}
document.createElement = () => episodeTab();
const renderedEpisodeTabs = [];
elements["episode-tabs"] = {
  replaceChildren(...tabs) {
    renderedEpisodeTabs.splice(0, renderedEpisodeTabs.length, ...tabs);
  },
};
renderProgress = () => {};
renderRevision = () => {};
const progress = {
  current_stage: "generating_story_outline",
  completed_stages: ["determining_direction"],
  elapsed_seconds: 1,
  recovery_state: "none",
  recovery_reason: "none",
  final_review: { l0: "pending", l4: "pending" },
  can_continue: false,
  can_end: false,
};
const drafts = {
  artifacts: [
    {
      stage: "determining_direction",
      selected_l0_variant: "归返",
      selection_rationale: "匹配故事母题。",
    },
    { stage: "generating_story_outline", content: "<b>已提交故事大纲</b>" },
  ],
  review_status: { l0: "pending", l4: "pending" },
};
state.creationId = "creation-id";
state.workspaceView = "progress";
state.activeVersion = "initial";
state.activeArtifact = "story_outline";
state.activeDraftRunKind = "";
state.creation = {
  persona: { display_name: "测试人格", version: "1" },
  initial: { state: "running", progress, drafts },
  revision: { state: "unavailable" },
};
renderCreation();
if (elements["result-workspace"].hidden) {
  throw new Error("running drafts were not visible in the creation scene");
}
state.activeArtifact = "story_outline";
renderArtifact();
if (elements["artifact-content"].textContent !== "<b>已提交故事大纲</b>") {
  throw new Error("running draft content was not rendered");
}
if (elements["artifact-version-mark"].textContent !== "创作中草稿") {
  throw new Error("running content was presented as a formal delivery");
}
state.workspaceView = "reading";
renderWorkspace();
if (!elements["result-workspace"].hidden) {
  throw new Error("drafts appeared in the formal reader");
}
state.workspaceView = "progress";

const pausedProgress = {
  ...progress,
  recovery_state: "paused",
  recovery_reason: "relay_interruption",
  can_continue: true,
  can_end: true,
};
state.creation = {
  ...state.creation,
  initial: { state: "paused", progress: pausedProgress, drafts },
};
renderCreation();
if (elements["result-workspace"].hidden) {
  throw new Error("paused drafts were not retained for inspection");
}
if (!elements["wait-title"].textContent.includes("网络 / Relay")) {
  throw new Error("relay pause did not use safe relay copy");
}

state.creation = {
  ...state.creation,
  initial: { state: "auto_resuming", progress: pausedProgress, drafts },
};
renderCreation();
if (!elements["wait-kicker"].textContent.includes("网络 / Relay")) {
  throw new Error("relay auto recovery did not use safe relay copy");
}
if (elements["result-workspace"].hidden) {
  throw new Error("recovering drafts were not retained for inspection");
}

state.creation = {
  ...state.creation,
  initial: { state: "queued", progress, drafts },
};
renderCreation();
if (elements["result-workspace"].hidden) {
  throw new Error("queued drafts were not retained for inspection");
}

const episodeProgress = {
  ...progress,
  current_stage: "generating_episode_scripts",
  episodes: { total: 3, completed: 1, current: 2 },
};
const episodeDrafts = {
  ...drafts,
  episodes: [{ episode_number: 1, content: "第一集已提交剧本" }],
};
state.activeArtifact = "episode_scripts";
state.creation = {
  ...state.creation,
  initial: { state: "running", progress: episodeProgress, drafts: episodeDrafts },
};
renderCreation();
if (elements["result-workspace"].hidden || elements["episode-navigator"].hidden) {
  throw new Error("episode drafts were not visible during script generation");
}
if (!elements["episode-progress-summary"].textContent.includes("共 3 集")) {
  throw new Error("episode progress summary was not rendered");
}
if (renderedEpisodeTabs[0]?.dataset.status !== "completed") {
  throw new Error("completed episode did not retain its status");
}
if (renderedEpisodeTabs[1]?.dataset.status !== "current" || !renderedEpisodeTabs[1]?.disabled) {
  throw new Error("unsubmitted current episode was presented as readable");
}
if (elements["episode-content"].textContent !== "第一集已提交剧本") {
  throw new Error("persisted episode content was not rendered");
}

const formalInitial = {
  state: "succeeded",
  result: {
    content_package: {
      story_outline: "初稿故事大纲",
      character_biographies: "初稿人物小传",
      relationship_logic: "初稿人物关系",
      episode_outline: "初稿分集大纲",
      episode_scripts: "初稿分集剧本",
    },
  },
};
state.activeDraftRunKind = "";
state.activeArtifact = "story_outline";
state.creation = {
  persona: { display_name: "测试人格", version: "1" },
  initial: formalInitial,
  revision: { state: "available" },
};
renderCreation();
if (state.workspaceView !== "reading") {
  throw new Error("formal initial delivery did not enter reading scene");
}
if (elements["result-workspace"].hidden) {
  throw new Error("formal delivery stayed hidden");
}
if (elements["artifact-content"].textContent !== "初稿故事大纲") {
  throw new Error("formal initial delivery was not rendered");
}
"""
    harness = f"""
const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync({json.dumps(str(script_path))}, "utf8");
const context = {{
  document: {{ addEventListener() {{}} }},
  window: {{}},
  crypto: {{ randomUUID() {{ return "test-id"; }} }},
  console,
}};
const result = vm.runInNewContext(
  source + "\\n(() => {{" + {json.dumps(assertions)} + "}})()",
  context,
);
Promise.resolve(result).catch((error) => {{
  console.error(error);
  process.exitCode = 1;
}});
"""

    subprocess.run(
        ["node", "-e", harness],
        check=True,
        capture_output=True,
        text=True,
    )


def test_episode_navigator_uses_durable_progress_and_drafts() -> None:
    root = Path(__file__).parents[1]
    script_path = root / "src" / "pengine" / "web" / "app.js"
    page = (root / "src" / "pengine" / "web" / "index.html").read_text()
    assert 'id="episode-navigator"' in page
    assert 'id="episode-tabs"' in page
    assert 'id="episode-content"' in page

    assertions = """
function tab() {
  return {
    dataset: {},
    attrs: {},
    disabled: false,
    tabIndex: 0,
    textContent: "",
    setAttribute(name, value) { this.attrs[name] = value; },
    focus() { focusedEpisode = this.dataset.episodeNumber; },
  };
}
document.createElement = () => tab();
let focusedEpisode = null;
const episodeTabs = {
  children: [],
  replaceChildren(...children) { this.children = children; },
  querySelector(selector) {
    const match = selector.match(/data-episode-number="(\\d+)"/);
    return match
      ? this.children.find((child) => child.dataset.episodeNumber === match[1])
      : null;
  },
};
Object.assign(elements, {
  "episode-navigator": { hidden: true },
  "episode-progress-summary": { textContent: "" },
  "episode-tabs": episodeTabs,
  "episode-content": {
    textContent: "",
    attrs: {},
    setAttribute(name, value) { this.attrs[name] = value; },
    removeAttribute(name) { delete this.attrs[name]; },
  },
});

const endedRun = {
  state: "ended",
  progress: {
    episodes: { total: 3, completed: 2, current: null },
  },
  drafts: {
    artifacts: [],
    episodes: [
      { episode_number: 1, content: "第一集已提交剧本" },
      { episode_number: 2, content: "第二集已提交剧本" },
    ],
  },
};
state.activeEpisode = null;
const views = artifactViewsForRun(endedRun);
if (!views.some((view) => view.key === "episode_scripts" && view.isEpisodeNavigator)) {
  throw new Error("episode progress did not expose the script navigator");
}
renderEpisodeNavigator(endedRun);
if (elements["episode-navigator"].hidden) throw new Error("ended navigator was hidden");
if (!elements["episode-progress-summary"].textContent.includes("共 3 集")) {
  throw new Error("total was not rendered");
}
if (!elements["episode-progress-summary"].textContent.includes("已完成 2 集")) {
  throw new Error("completed count was not rendered");
}
if (!elements["episode-progress-summary"].textContent.includes("任务已结束")) {
  throw new Error("ended state was not retained");
}
if (episodeTabs.children.length !== 3) {
  throw new Error("episode count did not create navigator tabs");
}
if (episodeTabs.children[0].disabled || episodeTabs.children[1].disabled) {
  throw new Error("committed episode drafts were not navigable");
}
if (!episodeTabs.children[2].disabled) throw new Error("undrafted episode became readable");
if (elements["episode-content"].textContent !== "第一集已提交剧本") {
  throw new Error("navigator did not use the first durable draft");
}

state.activeEpisode = 2;
renderEpisodeNavigator(endedRun);
if (elements["episode-content"].textContent !== "第二集已提交剧本") {
  throw new Error("navigator did not preserve the selected durable draft");
}

const failedRun = {
  ...endedRun,
  state: "failed",
  progress: { episodes: { total: 3, completed: 2, current: 3 } },
};
state.activeEpisode = 99;
renderEpisodeNavigator(failedRun);
if (!elements["episode-progress-summary"].textContent.includes("当前第 3 集")) {
  throw new Error("current episode was not rendered");
}
if (!elements["episode-progress-summary"].textContent.includes("任务失败")) {
  throw new Error("failed state was not retained");
}
if (episodeTabs.children[2].dataset.status !== "current" || !episodeTabs.children[2].disabled) {
  throw new Error("undrafted current episode was presented as readable");
}
renderArtifact = () => {};
handleEpisodeClick({ target: { closest() { return episodeTabs.children[1]; } } });
if (state.activeEpisode !== 2 || focusedEpisode !== "2") {
  throw new Error("episode selection did not retain keyboard focus");
}
const succeededRun = {
  state: "succeeded",
  progress: { episodes: { total: 3, completed: 3, current: null } },
  result: { content_package: { episode_scripts: "完整交付剧本" } },
};
const formalScript = artifactViewsForRun(succeededRun).find(
  (view) => view.key === "episode_scripts",
);
if (!formalScript || formalScript.isEpisodeNavigator || formalScript.content !== "完整交付剧本") {
  throw new Error("formal script was replaced or parsed as episode drafts");
}
"""
    harness = f"""
const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync({json.dumps(str(script_path))}, "utf8");
const context = {{
  document: {{ addEventListener() {{}}, createElement() {{ return {{}}; }} }},
  window: {{}},
  crypto: {{ randomUUID() {{ return "test-id"; }} }},
  console,
}};
const result = vm.runInNewContext(
  source + "\\n(() => {{" + {json.dumps(assertions)} + "}})()",
  context,
);
Promise.resolve(result).catch((error) => {{
  console.error(error);
  process.exitCode = 1;
}});
"""

    subprocess.run(
        ["node", "-e", harness],
        check=True,
        capture_output=True,
        text=True,
    )


def test_formal_episode_navigation_and_revision_drawer_contracts_are_wired() -> None:
    root = Path(__file__).parents[1]
    page = (root / "src" / "pengine" / "web" / "index.html").read_text()
    script = (root / "src" / "pengine" / "web" / "app.js").read_text()
    styles = (root / "src" / "pengine" / "web" / "styles.css").read_text()

    assert "styles.css?v=20260813-2" in page
    assert "app.js?v=20260813-2" in page
    assert 'id="previous-episode"' in page
    assert 'id="next-episode"' in page
    assert "moveFormalEpisode(-1)" in script
    assert "moveFormalEpisode(1)" in script
    assert 'event.key === "Escape"' in script
    assert ".revision-desk[hidden]" in styles


def test_workbench_presents_one_active_unfinished_series_bible_design() -> None:
    script_path = Path(__file__).parents[1] / "src" / "pengine" / "web" / "app.js"
    assertions = """
const designRun = {
  state: "running",
  progress: {
    episodes: { total: 2, completed: 0, current: null },
    current_stage: "generating_episode_scripts",
    completed_stages: ["generating_story_outline"],
  },
  drafts: {
    artifacts: [
      { stage: "generating_story_outline", content: "故事大纲" },
    ],
    design: {
      candidate_id: "candidate_abcdef1234567890",
      version: 1,
      design_epoch: 1,
      content_hash: "a".repeat(64),
      status: "active",
      is_active: true,
      unfinished: true,
      genre: "general",
      projections: {
        story_outline: "离乡者回到旧屋。",
        character_biographies: "林岚：主角。",
        relationship_logic: "关系逻辑",
        episode_outline: "第 1 集：林岚回到旧屋。",
        story_contract_markdown: "# Story Contract",
      },
    },
  },
};
const designView = seriesBibleDesignView(designRun);
if (!designView) throw new Error("active design was not exposed to the workbench");
if (designView.key !== "series_bible_design") throw new Error("design view key is wrong");
if (designView.overline !== "DESIGN") throw new Error("design view lacks its overline");
if (!designView.content.includes("未完成设计包（不作为正式交付）")) {
  throw new Error("active design was not labeled unfinished");
}
if (!designView.content.includes("candidate_abcdef1234567890")) {
  throw new Error("design view did not expose the candidate id");
}
if (!designView.content.includes("a".repeat(64))) {
  throw new Error("design view did not expose the content hash");
}
if (!designView.content.includes("第 1 集：林岚回到旧屋。")) {
  throw new Error("design view did not expose the episode outline projection");
}
if (seriesBibleDesignView({ state: "running", drafts: {} }) !== null) {
  throw new Error("a run without a design must not expose a design view");
}
const artifactViews = artifactViewsForRun(designRun);
if (!artifactViews.some((view) => view.key === "series_bible_design")) {
  throw new Error("the active design was missing from the run artifact tabs");
}
"""
    harness = f"""
const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync({json.dumps(str(script_path))}, "utf8");
const context = {{
  document: {{ addEventListener() {{}} }},
  window: {{}},
  crypto: {{ randomUUID() {{ return "test-id"; }} }},
  console,
}};
const result = vm.runInNewContext(
  source + "\\n(() => {{" + {json.dumps(assertions)} + "}})()",
  context,
);
Promise.resolve(result).catch((error) => {{
  console.error(error);
  process.exitCode = 1;
}});
"""

    subprocess.run(
        ["node", "-e", harness],
        check=True,
        capture_output=True,
        text=True,
    )


def test_progress_model_call_panel_hidden_by_default_and_renders_when_enabled() -> None:
    script_path = Path(__file__).parents[1] / "src" / "pengine" / "web" / "app.js"
    assertions = """
const stageItems = USER_STAGES.map(([stage]) => ({
  dataset: { stage },
  attrs: {},
  setAttribute(name, value) { this.attrs[name] = value; },
  removeAttribute(name) { delete this.attrs[name]; },
}));
const listItems = [];
Object.assign(elements, {
  "run-progress": { hidden: true },
  "progress-kind": { textContent: "" },
  "progress-title": { textContent: "" },
  "progress-elapsed": { textContent: "" },
  "progress-stages": { querySelectorAll() { return stageItems; } },
  "review-progress": { hidden: true },
  "review-l0": { textContent: "" },
  "review-l4": { textContent: "" },
  "run-controls": { hidden: true },
  "run-control-title": { textContent: "" },
  "run-control-description": { textContent: "" },
  "continue-run": { disabled: false },
  "end-run": { disabled: false },
  "run-control-message": { textContent: "" },
  "episode-progress": { hidden: true },
  "episode-progress-label": { textContent: "" },
  "episode-progress-detail": { textContent: "" },
  "model-call-panel": { hidden: true },
  "model-call-totals": { textContent: "" },
  "model-call-list": {
    replaceChildren() { listItems.length = 0; },
    appendChild(item) { listItems.push(item); },
  },
});
state.workspaceView = "progress";
state.activeVersion = "initial";
state.activeArtifact = "direction";
state.creation = {
  initial: {
    state: "paused",
    pause: {
      code: "context_budget",
      message: "完整请求超出已验证上下文上限。",
      stage: "generating_story_outline",
    },
    progress: {
      current_stage: "generating_story_outline",
      completed_stages: ["determining_direction"],
      elapsed_seconds: 10,
      recovery_state: "paused",
      recovery_reason: "context_budget",
      final_review: { l0: "pending", l4: "pending" },
      can_continue: true,
      can_end: true,
      model_calls: [
        {
          call_id: "call-1",
          role: "generation",
          adapter: "anthropic",
          provider: "anthropic",
          model: "claude-opus-5",
          stage: "generating_story_outline",
          episode_number: null,
          requested_at: "2026-08-03T00:00:00Z",
          estimated_total_tokens: 2128000,
          verified_limit_tokens: 200000,
          preflight: "blocked",
          status: "preflight_blocked",
          usage: { input_tokens: null, output_tokens: null, status: "unavailable" },
          finish_reason: null,
          duration_seconds: 0,
          outcome: "blocked",
        },
        {
          call_id: "call-2",
          role: "review",
          adapter: "deepseek",
          provider: "deepseek",
          model: "deepseek-v4-flash",
          response_model_ids: ["deepseek-v4-flash"],
          stage: "accepting_l0",
          episode_number: null,
          requested_at: "2026-08-03T00:01:00Z",
          estimated_total_tokens: 5000,
          verified_limit_tokens: 64000,
          preflight: "ok",
          status: "succeeded",
          usage: { input_tokens: 1200, output_tokens: 300, status: "reported" },
          finish_reason: "stop",
          duration_seconds: 3,
          outcome: "success",
        },
      ],
    },
    drafts: { artifacts: [], episodes: [] },
  },
  revision: { state: "unavailable" },
};
renderProgress();
if (!elements["model-call-panel"].hidden) {
  throw new Error("model-call panel visible while switch disabled");
}
if (elements["model-call-totals"].textContent !== "") {
  throw new Error("model-call totals rendered while switch disabled");
}
if (listItems.length !== 0) {
  throw new Error("model-call list rendered while switch disabled");
}
state.modelCallPanelEnabled = true;
renderProgress();
if (elements["model-call-panel"].hidden) {
  throw new Error("model-call panel stayed hidden with durable records");
}
if (!elements["model-call-totals"].textContent.includes("共 2 次调用")) {
  throw new Error("model-call totals missing call count");
}
if (!elements["model-call-totals"].textContent.includes("成功 1")) {
  throw new Error("model-call totals missing success classification");
}
if (!elements["model-call-totals"].textContent.includes("预检拦截 1")) {
  throw new Error("model-call totals missing blocked classification");
}
if (!elements["model-call-totals"].textContent.includes("实际用量 输入 1200 / 输出 300")) {
  throw new Error("model-call totals missing reported actual usage");
}
if (listItems.length !== 2) {
  throw new Error("model-call list did not render each durable call");
}
if (listItems[0].dataset.callStatus !== "preflight_blocked") {
  throw new Error("blocked call lost its classification");
}
if (!listItems[1].textContent.includes("deepseek-v4-flash")) {
  throw new Error("review call did not render its model");
}
if (!listItems[1].textContent.includes("响应身份 deepseek-v4-flash")) {
  throw new Error("review call did not render its response model identity");
}
state.creation.initial.progress.recovery_reason = "relay_identity_mismatch";
state.creation.initial.progress.episodes = { total: 30, completed: 26, current: 27 };
state.creation.initial.pause = {
  code: "relay_identity_mismatch",
  message: "reported: relay-fallback",
  stage: "generating_episode_scripts",
  episode_number: 27,
};
renderProgress();
if (!elements["run-control-title"].textContent.includes("第 27 集模型身份待确认")) {
  throw new Error("identity mismatch pause did not render its dedicated state");
}
if (elements["continue-run"].textContent !== "从第 27 集继续") {
  throw new Error("identity mismatch pause did not preserve the episode resume point");
}
"""
    harness = f"""
const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync({json.dumps(str(script_path))}, "utf8");
const context = {{
  document: {{
    addEventListener() {{}},
    createElement() {{
      return {{ dataset: {{}}, textContent: "", appendChild() {{}} }};
    }},
  }},
  window: {{}},
  crypto: {{ randomUUID() {{ return "test-id"; }} }},
  console,
}};
const result = vm.runInNewContext(
  source + "\\n(() => {{" + {json.dumps(assertions)} + "}})()",
  context,
);
Promise.resolve(result).catch((error) => {{
  console.error(error);
  process.exitCode = 1;
}});
"""

    subprocess.run(
        ["node", "-e", harness],
        check=True,
        capture_output=True,
        text=True,
    )


def test_progress_renders_episode_progress_line() -> None:
    script_path = Path(__file__).parents[1] / "src" / "pengine" / "web" / "app.js"
    assertions = """
const stageItems = USER_STAGES.map(([stage]) => ({
  dataset: { stage },
  attrs: {},
  setAttribute(name, value) { this.attrs[name] = value; },
  removeAttribute(name) { delete this.attrs[name]; },
}));
Object.assign(elements, {
  "run-progress": { hidden: true },
  "progress-kind": { textContent: "" },
  "progress-title": { textContent: "" },
  "progress-elapsed": { textContent: "" },
  "progress-stages": { querySelectorAll() { return stageItems; } },
  "episode-progress": { hidden: true },
  "episode-progress-label": { textContent: "" },
  "episode-progress-detail": { textContent: "" },
  "review-progress": { hidden: true },
  "review-l0": { textContent: "" },
  "review-l4": { textContent: "" },
  "run-controls": { hidden: true },
  "run-control-title": { textContent: "" },
  "run-control-description": { textContent: "" },
  "continue-run": { disabled: false },
  "end-run": { disabled: false },
  "run-control-message": { textContent: "" },
  "model-call-panel": { hidden: true },
  "model-call-totals": { textContent: "" },
  "model-call-list": { replaceChildren() {}, appendChild() {} },
});
state.workspaceView = "progress";
state.activeVersion = "initial";
state.activeArtifact = "direction";
state.creation = {
  initial: {
    state: "running",
    progress: {
      current_stage: "generating_episode_scripts",
      completed_stages: ["determining_direction"],
      elapsed_seconds: 10,
      recovery_state: "none",
      recovery_reason: "none",
      final_review: { l0: "pending", l4: "pending" },
      can_continue: false,
      can_end: false,
      episodes: { total: 30, completed: 26, current: 27 },
      model_calls: [],
    },
    drafts: { artifacts: [], episodes: [] },
  },
  revision: { state: "unavailable" },
};
renderProgress();
if (elements["episode-progress"].hidden) {
  throw new Error("episode progress stayed hidden with live episode data");
}
if (!elements["episode-progress-detail"].textContent.includes("第 27/30 集")) {
  throw new Error("episode progress missing current episode");
}
if (!elements["episode-progress-detail"].textContent.includes("已完成 26")) {
  throw new Error("episode progress missing completed count");
}

state.creation.initial.progress.episodes = null;
renderProgress();
if (!elements["episode-progress"].hidden) {
  throw new Error("episode progress visible without episode data");
}

state.creation.initial.progress.episodes = { total: 30, completed: 5, current: null };
renderProgress();
if (elements["episode-progress"].hidden) {
  throw new Error("episode progress hidden with completed-only data");
}
if (elements["episode-progress-detail"].textContent !== "已完成 5/30 集") {
  throw new Error(
    "completed-only text mismatch: " + elements["episode-progress-detail"].textContent,
  );
}

state.creation.initial.progress.episodes = { total: 30, completed: 30, current: null };
renderProgress();
if (!elements["episode-progress-detail"].textContent.includes("已全部完成 30 集")) {
  throw new Error("fully completed text mismatch");
}
"""
    harness = f"""
const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync({json.dumps(str(script_path))}, "utf8");
const context = {{
  document: {{
    addEventListener() {{}},
    createElement() {{
      return {{ dataset: {{}}, textContent: "", appendChild() {{}} }};
    }},
  }},
  window: {{}},
  crypto: {{ randomUUID() {{ return "test-id"; }} }},
  console,
}};
const result = vm.runInNewContext(
  source + "\\n(() => {{" + {json.dumps(assertions)} + "}})()",
  context,
);
Promise.resolve(result).catch((error) => {{
  console.error(error);
  process.exitCode = 1;
}});
"""

    subprocess.run(
        ["node", "-e", harness],
        check=True,
        capture_output=True,
        text=True,
    )


def test_persona_cards_gate_all_but_first() -> None:
    script_path = Path(__file__).parents[1] / "src" / "pengine" / "web" / "app.js"
    assertions = """
function makeElement() {
  const el = {
    className: "",
    textContent: "",
    type: "",
    name: "",
    value: "",
    checked: false,
    disabled: false,
    dataset: {},
    attrs: {},
    children: [],
    listeners: {},
  };
  el.classList = {
    add(...names) {
      el.className = [el.className, ...names].filter(Boolean).join(" ");
    },
  };
  el.setAttribute = (name, value) => {
    el.attrs[name] = value;
  };
  el.removeAttribute = (name) => {
    delete el.attrs[name];
  };
  el.addEventListener = (type, handler) => {
    el.listeners[type] = handler;
  };
  el.append = (...nodes) => {
    el.children.push(...nodes);
  };
  return el;
}
document.createElement = makeElement;
const cards = [];
Object.assign(elements, {
  "persona-grid": {
    replaceChildren() {
      cards.length = 0;
    },
    append(card) {
      cards.push(card);
    },
  },
  "persona-status": { textContent: "" },
  "reload-personas": { disabled: false },
});
state.personas = new Map(
  PERSONA_ORDER.map((id) => [id, { persona_id: id, display_name: id, version: "1" }]),
);
state.selectedPersonaId = "";
renderPersonaCards();
if (cards.length !== PERSONA_ORDER.length) {
  throw new Error(`expected ${PERSONA_ORDER.length} cards, received ${cards.length}`);
}
const findByClass = (el, cls) =>
  el.children.find((child) => child.className === cls) ||
  el.children.map((child) => findByClass(child, cls)).find(Boolean);
const rows = PERSONA_ORDER.map((id, index) => ({
  id,
  card: cards[index],
  input: findByClass(cards[index], "persona-radio"),
  availability: findByClass(cards[index], "persona-availability"),
}));
for (const row of rows) {
  if (!row.input) throw new Error(`${row.id} card missing radio input`);
  if (!row.availability) throw new Error(`${row.id} card missing availability span`);
}
if (rows[0].input.disabled) throw new Error("first persona was disabled");
if (rows[0].card.dataset.supported !== "true") {
  throw new Error("first persona not marked supported");
}
if (rows[0].availability.textContent !== "● 可选择") {
  throw new Error("first persona availability text changed");
}
for (const row of rows.slice(1)) {
  if (!row.input.disabled) throw new Error(`${row.id} stayed selectable`);
  if (row.card.dataset.supported !== "false") {
    throw new Error(`${row.id} not marked unsupported`);
  }
  if (row.availability.textContent !== "◌ 正在支持中") {
    throw new Error(`${row.id} availability text: ${row.availability.textContent}`);
  }
}

apiRequest = async () => ({
  items: PERSONA_ORDER.map((id) => ({ persona_id: id, display_name: id, version: "1" })),
});
setServiceState = () => {};
state.selectedPersonaId = "wuzhen";
await loadPersonas();
if (state.selectedPersonaId !== "") {
  throw new Error("residual unsupported selection survived loadPersonas");
}
if (!elements["persona-status"].textContent.includes("仅开放第一位人格")) {
  throw new Error("status line missing gating note");
}
if (!elements["persona-status"].textContent.includes("正在支持中")) {
  throw new Error("status line missing support-in-progress note");
}
"""
    harness = f"""
const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync({json.dumps(str(script_path))}, "utf8");
const context = {{
  document: {{
    addEventListener() {{}},
    createElement() {{
      return {{ dataset: {{}}, textContent: "", appendChild() {{}} }};
    }},
  }},
  window: {{}},
  crypto: {{ randomUUID() {{ return "test-id"; }} }},
  console,
}};
const result = vm.runInNewContext(
  source + "\\n(async () => {{" + {json.dumps(assertions)} + "}})()",
  context,
);
Promise.resolve(result).catch((error) => {{
  console.error(error);
  process.exitCode = 1;
}});
"""

    subprocess.run(
        ["node", "-e", harness],
        check=True,
        capture_output=True,
        text=True,
    )


def test_story_section_navigation_preserves_hierarchy() -> None:
    root = Path(__file__).parents[1]
    script_path = root / "src" / "pengine" / "web" / "app.js"
    styles = (root / "src" / "pengine" / "web" / "styles.css").read_text(encoding="utf-8")
    assert '#section-items button[data-level="2"]' in styles
    assert '#section-items button[data-level="3"]' in styles

    assertions = """
const sectionItems = {
  children: [],
  replaceChildren(...children) { this.children = children; },
};
Object.assign(elements, {
  "section-nav": { hidden: true },
  "presentation-status": { textContent: "" },
  "section-items": sectionItems,
});
renderSectionNavigation(
  [
    { id: "story-1", ordinal: 1, label: "主因果线", level: 1 },
    { id: "story-2", ordinal: 2, label: "阶段一", level: 2 },
    { id: "story-3", ordinal: 3, label: "阶段细节", level: 3 },
    { id: "story-4", ordinal: 4, label: "历史数据" },
  ],
  { id: "story-2" },
  { mode: "structured" },
);
const levels = sectionItems.children.map((button) => button.dataset.level);
if (JSON.stringify(levels) !== JSON.stringify(["1", "2", "3", "1"])) {
  throw new Error(`section hierarchy was lost: ${JSON.stringify(levels)}`);
}
if (sectionItems.children[1].attributes["aria-selected"] !== "true") {
  throw new Error("hierarchical item selection changed");
}
"""
    harness = f"""
const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync({json.dumps(str(script_path))}, "utf8");
function makeElement(tagName) {{
  return {{
    tagName,
    dataset: {{}},
    attributes: {{}},
    children: [],
    setAttribute(name, value) {{ this.attributes[name] = value; }},
    append(...children) {{ this.children.push(...children); }},
  }};
}}
const context = {{
  document: {{
    addEventListener() {{}},
    createElement: makeElement,
  }},
  window: {{}},
  crypto: {{ randomUUID() {{ return "test-id"; }} }},
  console,
}};
const result = vm.runInNewContext(
  source + "\\n(() => {{" + {json.dumps(assertions)} + "}})()",
  context,
);
Promise.resolve(result).catch((error) => {{
  console.error(error);
  process.exitCode = 1;
}});
"""

    subprocess.run(
        ["node", "-e", harness],
        check=True,
        capture_output=True,
        text=True,
    )


def test_formal_delivery_exports_the_selected_version_as_markdown() -> None:
    root = Path(__file__).parents[1]
    script_path = root / "src" / "pengine" / "web" / "app.js"
    page = (root / "src" / "pengine" / "web" / "index.html").read_text(encoding="utf-8")
    assert 'id="export-delivery"' in page
    assert "正式成品可导出" in page

    assertions = """
const contentPackage = {
  story_outline: "故事大纲原文",
  character_biographies: "人物小传原文",
  relationship_logic: "人物关系原文",
  episode_outline: "分集大纲原文",
  episode_scripts: "分集剧本原文",
};
const initial = { state: "succeeded", result: { content_package: contentPackage } };
const revision = { state: "succeeded", result: { content_package: contentPackage } };
const initialExport = createDeliveryExport(initial, {
  creationId: "abcd1234-creation",
  kind: "initial",
  persona: { display_name: "守拙", version: "0.6.0" },
  exportedAt: new Date("2026-08-15T02:00:00.000Z"),
});
if (initialExport.filename !== "意态短剧_ABCD1234_初稿.md") {
  throw new Error(`unexpected initial filename: ${initialExport.filename}`);
}
const orderedHeadings = [
  "## 01 故事大纲",
  "## 02 人物小传",
  "## 03 人物关系",
  "## 04 分集大纲",
  "## 05 分集剧本",
];
let previous = -1;
for (const heading of orderedHeadings) {
  const position = initialExport.content.indexOf(heading);
  if (position <= previous) throw new Error(`export section order changed at ${heading}`);
  previous = position;
}
for (const value of Object.values(contentPackage)) {
  if (!initialExport.content.includes(value)) throw new Error(`export omitted ${value}`);
}
if (!initialExport.content.includes("稿件版本：初稿")) throw new Error("initial label missing");
if (!initialExport.content.includes("导出时间：2026-08-15T02:00:00.000Z")) {
  throw new Error("export time missing");
}
const revisionExport = createDeliveryExport(revision, {
  creationId: "abcd1234-creation",
  kind: "revision",
  persona: { display_name: "守拙", version: "0.6.0" },
  exportedAt: "2026-08-15T02:00:00.000Z",
});
if (!revisionExport.filename.endsWith("_修订稿.md")) throw new Error("revision filename wrong");
if (!revisionExport.content.includes("稿件版本：修订稿")) throw new Error("revision label missing");

const incomplete = structuredClone(initial);
incomplete.result.content_package.episode_scripts = "";
let incompleteBlocked = false;
try {
  createDeliveryExport(incomplete, {
    creationId: "abcd1234-creation",
    kind: "initial",
    exportedAt: new Date("2026-08-15T02:00:00.000Z"),
  });
} catch (error) {
  incompleteBlocked = error.message === "delivery_artifact_missing:episode_scripts";
}
if (!incompleteBlocked) throw new Error("incomplete formal package was exported");

Object.assign(elements, {
  "export-delivery": { hidden: true, disabled: true },
  toast: { textContent: "", hidden: true },
});
state.workspaceView = "progress";
state.creationId = "abcd1234-creation";
state.creation = {
  persona: { display_name: "守拙", version: "0.6.0" },
  initial,
  revision: { state: "failed" },
};
state.activeVersion = "revision";
renderExportControl();
if (!elements["export-delivery"].hidden || !elements["export-delivery"].disabled) {
  throw new Error("failed revision exposed export");
}
state.activeVersion = "initial";
renderExportControl();
if (elements["export-delivery"].hidden || elements["export-delivery"].disabled) {
  throw new Error("successful initial delivery hid export");
}

handleExportDelivery();
if (downloadState.downloaded !== 1) throw new Error("export did not trigger one download");
if (!downloadState.lastLink.download.endsWith("_初稿.md")) {
  throw new Error("download used wrong version");
}
if (!downloadState.blobParts[0].startsWith("\ufeff# 意态短剧成品包")) {
  throw new Error("download omitted UTF-8 BOM or document title");
}
if (downloadState.revokedUrl !== "blob:test") throw new Error("download URL was not released");
"""
    harness = f"""
const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync({json.dumps(str(script_path))}, "utf8");
const downloadState = {{ blobParts: [], lastLink: null, revokedUrl: "", downloaded: 0 }};
class FakeBlob {{
  constructor(parts, options) {{ downloadState.blobParts = parts; this.options = options; }}
}}
const context = {{
  downloadState,
  Blob: FakeBlob,
  URL: {{
    createObjectURL() {{ return "blob:test"; }},
    revokeObjectURL(value) {{ downloadState.revokedUrl = value; }},
  }},
  structuredClone,
  document: {{
    addEventListener() {{}},
    body: {{ appendChild(link) {{ downloadState.lastLink = link; }} }},
    createElement(tagName) {{
      if (tagName !== "a") throw new Error(`unexpected element: ${{tagName}}`);
      return {{
        hidden: false,
        href: "",
        download: "",
        click() {{ downloadState.downloaded += 1; }},
        remove() {{}},
      }};
    }},
  }},
  window: {{ setTimeout(callback) {{ callback(); }} }},
  crypto: {{ randomUUID() {{ return "test-id"; }} }},
  console,
}};
const result = vm.runInNewContext(
  source + "\\n(() => {{" + {json.dumps(assertions)} + "}})()",
  context,
);
Promise.resolve(result).catch((error) => {{
  console.error(error);
  process.exitCode = 1;
}});
"""

    result = subprocess.run(
        ["node", "-e", harness],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_progress_renders_outline_group_line_during_outline_stage() -> None:
    script_path = Path(__file__).parents[1] / "src" / "pengine" / "web" / "app.js"
    assertions = """
const stageItems = USER_STAGES.map(([stage]) => ({
  dataset: { stage },
  attrs: {},
  setAttribute(name, value) { this.attrs[name] = value; },
  removeAttribute(name) { delete this.attrs[name]; },
}));
Object.assign(elements, {
  "run-progress": { hidden: true },
  "progress-kind": { textContent: "" },
  "progress-title": { textContent: "" },
  "progress-elapsed": { textContent: "" },
  "progress-stages": { querySelectorAll() { return stageItems; } },
  "episode-progress": { hidden: true },
  "episode-progress-label": { textContent: "" },
  "episode-progress-detail": { textContent: "" },
  "review-progress": { hidden: true },
  "review-l0": { textContent: "" },
  "review-l4": { textContent: "" },
  "run-controls": { hidden: true },
  "run-control-title": { textContent: "" },
  "run-control-description": { textContent: "" },
  "continue-run": { disabled: false },
  "end-run": { disabled: false },
  "run-control-message": { textContent: "" },
  "model-call-panel": { hidden: true },
  "model-call-totals": { textContent: "" },
  "model-call-list": { replaceChildren() {}, appendChild() {} },
});
state.workspaceView = "progress";
state.activeVersion = "initial";
state.activeArtifact = "direction";
state.creation = {
  initial: {
    state: "running",
    progress: {
      current_stage: "generating_episode_outline",
      completed_stages: ["determining_direction", "generating_story_outline"],
      elapsed_seconds: 2400,
      recovery_state: "none",
      recovery_reason: "none",
      final_review: { l0: "pending", l4: "pending" },
      can_continue: false,
      can_end: false,
      episodes: null,
      outline_groups: {
        committed_groups: 13,
        committed_through_episode: 36,
        current_group: 14,
        current_start_episode: 37,
        current_end_episode: 38,
      },
      model_calls: [],
    },
    drafts: { artifacts: [], episodes: [] },
  },
  revision: { state: "unavailable" },
};
renderProgress();
if (elements["episode-progress"].hidden) {
  throw new Error("outline group progress stayed hidden during outline stage");
}
if (elements["episode-progress-label"].textContent !== "大纲组进度") {
  throw new Error("outline label mismatch: " + elements["episode-progress-label"].textContent);
}
const outlineDetail = elements["episode-progress-detail"].textContent;
if (!outlineDetail.includes("已提交 13 组")) throw new Error("missing committed groups");
if (!outlineDetail.includes("覆盖第 1–36 集")) throw new Error("missing coverage");
if (!outlineDetail.includes("正在生成第 14 组（第 37–38 集）")) {
  throw new Error("missing current group: " + outlineDetail);
}

state.creation.initial.progress.outline_groups = null;
renderProgress();
if (!elements["episode-progress"].hidden) {
  throw new Error("outline progress visible without group data");
}

state.creation.initial.progress.current_stage = "generating_episode_scripts";
state.creation.initial.progress.episodes = { total: 38, completed: 5, current: 6 };
renderProgress();
if (elements["episode-progress-label"].textContent !== "分集进度") {
  throw new Error("script stage did not restore the episode label");
}
if (!elements["episode-progress-detail"].textContent.includes("第 6/38 集")) {
  throw new Error("script stage detail lost episode position");
}
"""
    harness = f"""
const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync({json.dumps(str(script_path))}, "utf8");
const context = {{
  document: {{
    addEventListener() {{}},
    createElement() {{
      return {{ dataset: {{}}, textContent: "", appendChild() {{}} }};
    }},
  }},
  window: {{}},
  crypto: {{ randomUUID() {{ return "test-id"; }} }},
  console,
}};
const result = vm.runInNewContext(
  source + "\\n(() => {{" + {json.dumps(assertions)} + "}})()",
  context,
);
Promise.resolve(result).catch((error) => {{
  console.error(error);
  process.exitCode = 1;
}});
"""

    result = subprocess.run(
        ["node", "-e", harness],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
