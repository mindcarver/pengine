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
