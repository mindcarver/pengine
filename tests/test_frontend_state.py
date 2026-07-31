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
