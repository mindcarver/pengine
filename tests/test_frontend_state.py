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
renderWorkspace = () => true;
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
if (elements["result-workspace"].hidden) {
  throw new Error("failed initial run hid durable workspace");
}
state.creation.initial = { state: "ended" };
renderCreation();
if (elements["result-workspace"].hidden) {
  throw new Error("ended initial run hid durable workspace");
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
  throw new Error("quality rejection hid retained workspace");
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
  "continue-run": { disabled: false },
  "end-run": { disabled: false },
  "run-control-message": { textContent: "" },
});
const progress = {
  current_stage: "generating_story_outline",
  completed_stages: ["determining_direction"],
  elapsed_seconds: 125,
  recovery_state: "none",
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
      final_review: { l0: "passed", l4: "paused" },
      can_continue: true,
      can_end: true,
    },
  },
};
renderProgress();
if (elements["progress-kind"].textContent !== "修订进度") throw new Error("revision not reused");
if (elements["run-controls"].hidden) throw new Error("paused controls stayed hidden");
if (elements["review-progress"].hidden) throw new Error("review substatus stayed hidden");
if (!elements["review-l0"].textContent.includes("已通过")) throw new Error("L0 status lost");
if (!elements["review-l4"].textContent.includes("已暂停")) throw new Error("L4 status lost");
if (shouldPoll()) throw new Error("paused run kept polling");

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


def test_active_drafts_render_from_resource_snapshots_without_formal_results() -> None:
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
renderProgress = () => {};
renderRevision = () => {};
const progress = {
  current_stage: "generating_story_outline",
  completed_stages: ["determining_direction"],
  elapsed_seconds: 1,
  recovery_state: "none",
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
state.activeVersion = "initial";
state.activeArtifact = "story_outline";
state.activeDraftRunKind = "";
state.creation = {
  persona: { display_name: "测试人格", version: "1" },
  initial: { state: "running", progress, drafts },
  revision: { state: "unavailable" },
};
renderCreation();
if (elements["result-workspace"].hidden) throw new Error("running drafts stayed hidden");
if (elements["artifact-version-mark"].textContent !== "创作中草稿") {
  throw new Error("initial draft label was not rendered");
}
if (!elements["artifact-content"].textContent.includes("选择理由：匹配故事母题。")) {
  throw new Error("direction draft was not rendered");
}
if (artifactButtons[0].hidden || artifactButtons[1].hidden) {
  throw new Error("committed draft tabs stayed hidden");
}
if (!artifactButtons[2].hidden || !artifactButtons[5].hidden) {
  throw new Error("current or future draft tabs were shown");
}

const pausedProgress = {
  ...progress,
  recovery_state: "paused",
  can_continue: true,
  can_end: true,
};
state.creation = {
  ...state.creation,
  initial: { state: "paused", progress: pausedProgress, drafts },
};
renderCreation();
if (
  elements["result-workspace"].hidden ||
  !elements["artifact-content"].textContent.includes("匹配故事母题")
) {
  throw new Error("paused resource lost durable draft text");
}

state.creation = {
  ...state.creation,
  initial: { state: "queued", progress, drafts },
};
renderCreation();
if (
  elements["result-workspace"].hidden ||
  !elements["artifact-content"].textContent.includes("匹配故事母题")
) {
  throw new Error("continued resource lost durable draft text");
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
const revisionProgress = { ...progress, can_continue: true, can_end: true };
state.creation = {
  persona: { display_name: "测试人格", version: "1" },
  initial: formalInitial,
  revision: { state: "paused", progress: revisionProgress, drafts },
};
renderCreation();
if (state.activeVersion !== "revision") throw new Error("revision draft was not selected");
if (elements["artifact-version-mark"].textContent !== "修订中草稿") {
  throw new Error("revision draft label was not rendered");
}
state.activeVersion = "initial";
state.activeArtifact = "story_outline";
renderVersionControls();
renderArtifact();
if (elements["artifact-content"].textContent !== "初稿故事大纲") {
  throw new Error("revision draft hid formal initial delivery");
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
