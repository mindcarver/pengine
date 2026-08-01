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
  "delivery-subtitle": { textContent: "" },
  "folio-stamp": { textContent: "" },
  "series-empty": { hidden: true },
  "series-card": { hidden: false },
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
  "delivery-index": { textContent: "" },
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
renderSeries = () => {};
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
    assert "重新审核" in page

    assertions = """
const requests = [];
Object.assign(elements, {
  "delivery-section": { hidden: true },
  "folio-stamp": { textContent: "" },
  "delivery-subtitle": { textContent: "" },
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
    state: "paused",
    progress: {
      ...progress,
      current_stage: "final_review",
      completed_stages: USER_STAGES.slice(0, 6).map(([stage]) => stage),
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

state.creation.revision.state = "auto_resuming";
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
  button("character_biographies", "artifact"),
  button("relationship_logic", "artifact"),
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
  "result-workspace": { hidden: true, dataset: {} },
  "revision-desk": { hidden: false },
  "version-initial": button("initial", "version"),
  "version-revision": button("revision", "version"),
  "version-tabs": {
    querySelectorAll() { return [elements["version-initial"], elements["version-revision"]]; },
  },
  "version-note": { textContent: "" },
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
      current_stage: "generating_character_biographies",
      completed_stages: ["determining_direction", "generating_story_outline"],
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
if (stageItems[2].disabled !== true || stageItems[6].disabled !== true) {
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
  button("character_biographies", "artifact"),
  button("relationship_logic", "artifact"),
  button("episode_outline", "artifact"),
  button("episode_scripts", "artifact"),
];
Object.assign(elements, {
  "delivery-section": { hidden: true },
  "delivery-subtitle": { textContent: "" },
  "folio-stamp": { textContent: "" },
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
