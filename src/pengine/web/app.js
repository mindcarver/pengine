"use strict";

const PERSONA_ORDER = ["shouzhuo", "wuzhen", "sanfentian", "xinggui"];
const PERSONA_PROFILES = {
  shouzhuo: {
    name: "守拙",
    tag: "正剧",
    monogram: "守",
    bio: "从真实处境与人物选择出发，让冲突落到具体代价。",
    specialty: "现实议题 · 群像 · 人物弧光",
    suitable: "家庭、年代、职场与社会题材",
    accent: "#8e4334",
  },
  wuzhen: {
    name: "雾枕",
    tag: "悬疑",
    monogram: "雾",
    bio: "重视信息差、因果链与逐层揭示，让谜面服务人物。",
    specialty: "悬念铺排 · 线索回收 · 心理张力",
    suitable: "罪案、秘密、身份与封闭空间",
    accent: "#455d63",
  },
  sanfentian: {
    name: "三分甜",
    tag: "言情",
    monogram: "甜",
    bio: "在情感拉扯里寻找行动，用关系变化推动剧情。",
    specialty: "情感节奏 · 关系递进 · 细节互动",
    suitable: "都市、成长、重逢与轻喜情感",
    accent: "#aa5960",
  },
  xinggui: {
    name: "星轨",
    tag: "科幻",
    monogram: "星",
    bio: "先建立世界规则，再用非常处境检验人的选择。",
    specialty: "世界设定 · 规则推演 · 概念冲突",
    suitable: "近未来、太空、人工智能与时间题材",
    accent: "#405d7a",
  },
};

const FORMAL_ARTIFACTS = [
  { key: "story_outline", title: "故事大纲", overline: "DELIVERABLE 01" },
  { key: "character_biographies", title: "人物小传", overline: "DELIVERABLE 02" },
  { key: "relationship_logic", title: "关系逻辑", overline: "DELIVERABLE 03" },
  { key: "episode_outline", title: "分集大纲", overline: "DELIVERABLE 04" },
  { key: "episode_scripts", title: "分集剧本", overline: "DELIVERABLE 05" },
];

const DRAFT_ARTIFACTS = [
  {
    key: "direction",
    stage: "determining_direction",
    title: "创作方向",
    overline: "DRAFT 01",
  },
  {
    key: "story_outline",
    stage: "generating_story_outline",
    title: "故事大纲",
    overline: "DRAFT 02",
  },
  {
    key: "character_biographies",
    stage: "generating_character_biographies",
    title: "人物小传",
    overline: "DRAFT 03",
  },
  {
    key: "relationship_logic",
    stage: "generating_relationships",
    title: "关系逻辑",
    overline: "DRAFT 04",
  },
  {
    key: "episode_outline",
    stage: "generating_episode_outline",
    title: "分集大纲",
    overline: "DRAFT 05",
  },
];

const EPISODE_SCRIPTS_DRAFT = {
  key: "episode_scripts",
  title: "分集剧本",
  overline: "DRAFT 06",
  isEpisodeNavigator: true,
};

const STAGE_ARTIFACT_KEYS = new Map([
  ...DRAFT_ARTIFACTS.map(({ stage, key }) => [stage, key]),
  ["generating_episode_scripts", "episode_scripts"],
]);

const USER_STAGES = [
  ["determining_direction", "确定创作方向"],
  ["generating_story_outline", "生成故事大纲"],
  ["generating_character_biographies", "生成人物小传"],
  ["generating_relationships", "生成人物关系"],
  ["generating_episode_outline", "生成分集大纲"],
  ["generating_episode_scripts", "生成分集剧本"],
  ["final_review", "成品审核"],
];
const USER_STAGE_LABELS = new Map(USER_STAGES);
const REVIEW_STATUS_LABELS = {
  pending: "等待",
  running: "审核中",
  passed: "已通过",
  paused: "已暂停",
  failed: "未通过",
};

const STORAGE_KEY = "pengine.currentCreationId";
const POLL_INTERVAL_MS = 1800;
const DEFAULT_REQUIREMENTS = "按所选人格完成一部完整短剧。";

const state = {
  personas: new Map(),
  selectedPersonaId: "",
  creation: null,
  creationId: "",
  activeVersion: "initial",
  activeArtifact: "story_outline",
  activeDraftRunKind: "",
  activeEpisode: null,
  pollTimer: null,
  loadingCreation: false,
  pendingFeedback: "",
  progressRunKind: "",
  runControlBusy: false,
  workspaceView: "selection",
};

const elements = {};

class ApiError extends Error {
  constructor(message, code, status) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.status = status;
  }
}

document.addEventListener("DOMContentLoaded", start);

function start() {
  cacheElements();
  bindEvents();
  renderPersonaCards();
  void initialize();
}

function cacheElements() {
  const ids = [
    "service-stamp",
    "service-label",
    "reload-personas",
    "persona-grid",
    "persona-status",
    "creation-form",
    "story",
    "requirements",
    "creation-message",
    "create-button",
    "series-empty",
    "series-card",
    "series-status",
    "series-persona",
    "series-id",
    "series-date",
    "flow-select-writer",
    "flow-tell-story",
    "flow-create",
    "flow-read-deliverables",
    "selection-view",
    "brief-view",
    "current-work-view",
    "current-work-placeholder",
    "progress-scene",
    "back-to-selection",
    "brief-persona",
    "delivery-index",
    "delivery-section",
    "delivery-title",
    "delivery-subtitle",
    "folio-stamp",
    "run-progress",
    "progress-kind",
    "progress-title",
    "progress-elapsed",
    "progress-stages",
    "review-progress",
    "review-l0",
    "review-l4",
    "run-controls",
    "run-control-title",
    "run-control-description",
    "continue-run",
    "end-run",
    "run-control-message",
    "task-waiting",
    "wait-kicker",
    "wait-title",
    "wait-description",
    "failure-panel",
    "failure-label",
    "failure-title",
    "failure-message",
    "failure-guidance",
    "quality-rejection-details",
    "quality-rejection-stage",
    "quality-rejection-evidence",
    "quality-rejection-attempt",
    "failure-code",
    "failure-actions",
    "start-new-creation",
    "quality-rejection-actions",
    "retry-final-review",
    "end-quality-rejected-run",
    "quality-rejection-action-message",
    "result-workspace",
    "version-tabs",
    "version-initial",
    "version-revision",
    "version-note",
    "artifact-tabs",
    "artifact-panel",
    "artifact-title",
    "artifact-overline",
    "artifact-version-mark",
    "episode-navigator",
    "episode-progress-summary",
    "episode-tabs",
    "episode-content",
    "artifact-content",
    "revision-desk",
    "revision-description",
    "revision-form",
    "feedback",
    "feedback-state",
    "revision-message",
    "revision-button",
    "toast",
  ];

  for (const id of ids) {
    elements[id] = document.getElementById(id);
  }
}

function bindEvents() {
  elements["reload-personas"].addEventListener("click", () => void loadPersonas());
  elements["creation-form"].addEventListener("submit", handleCreate);
  elements["revision-form"].addEventListener("submit", handleRevision);
  elements["continue-run"].addEventListener("click", () => void handleRunControl("continue"));
  elements["end-run"].addEventListener("click", () => void handleRunControl("end"));
  elements["start-new-creation"].addEventListener("click", startNewCreation);
  elements["retry-final-review"].addEventListener("click", () =>
    void handleQualityRejectionControl("retry-final-review"),
  );
  elements["end-quality-rejected-run"].addEventListener("click", () =>
    void handleQualityRejectionControl("end"),
  );
  elements["series-card"].addEventListener("click", focusDelivery);
  for (const [id, view] of [
    ["flow-select-writer", "selection"],
    ["flow-tell-story", "brief"],
    ["flow-create", "progress"],
    ["flow-read-deliverables", "reading"],
  ]) {
    elements[id].addEventListener("click", () => setWorkspaceView(view));
  }
  elements["back-to-selection"].addEventListener("click", () =>
    setWorkspaceView("selection"),
  );

  elements["version-tabs"].addEventListener("click", handleVersionClick);
  elements["version-tabs"].addEventListener("keydown", handleHorizontalTabs);
  elements["progress-stages"].addEventListener("click", handleStageClick);
  elements["artifact-tabs"].addEventListener("click", handleArtifactClick);
  elements["artifact-tabs"].addEventListener("keydown", handleHorizontalTabs);
  elements["episode-tabs"].addEventListener("click", handleEpisodeClick);
  elements["episode-tabs"].addEventListener("keydown", handleHorizontalTabs);

  window.addEventListener("beforeunload", stopPolling);
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden && state.creationId && shouldPoll()) {
      void refreshCreation();
    }
  });
}

async function initialize() {
  const currentId = readCurrentCreationId();
  state.creationId = currentId;
  state.workspaceView = currentId ? "progress" : "selection";
  renderWorkspaceViews();
  renderSeries();

  await loadPersonas();
  if (currentId) {
    await refreshCreation({ isRestore: true });
  }
}

async function loadPersonas() {
  elements["reload-personas"].disabled = true;
  elements["persona-status"].textContent = "正在读取人格档案……";
  setServiceState("idle", "正在读取人格");

  try {
    const payload = await apiRequest("/personas");
    const items = Array.isArray(payload.items) ? payload.items : [];
    state.personas = new Map(items.map((persona) => [persona.persona_id, persona]));

    if (!state.personas.has(state.selectedPersonaId)) {
      state.selectedPersonaId = "";
    }

    const availableCount = PERSONA_ORDER.filter((id) => state.personas.has(id)).length;
    const missingCount = PERSONA_ORDER.length - availableCount;
    elements["persona-status"].textContent = missingCount
      ? `已读取 ${availableCount} 位原型人格；${missingCount} 位未被当前服务列为可选。`
      : "四位原型人格均已由本地服务确认可用。";
    setServiceState("ready", "本地服务已连接");
  } catch (error) {
    state.personas = new Map();
    state.selectedPersonaId = "";
    elements["persona-status"].textContent = formatError(error);
    setServiceState("error", "本地服务不可达");
  } finally {
    renderPersonaCards();
    elements["reload-personas"].disabled = false;
  }
}

function renderPersonaCards() {
  const grid = elements["persona-grid"];
  grid.replaceChildren();

  for (const personaId of PERSONA_ORDER) {
    const profile = PERSONA_PROFILES[personaId];
    const persona = state.personas.get(personaId);
    const available = Boolean(persona);
    const label = document.createElement("label");
    label.className = "persona-card";
    label.classList.add(`persona-${personaId}`);
    label.dataset.available = String(available);
    label.dataset.selected = String(state.selectedPersonaId === personaId);

    const input = document.createElement("input");
    input.className = "persona-radio";
    input.type = "radio";
    input.name = "persona_id";
    input.value = personaId;
    input.checked = state.selectedPersonaId === personaId;
    input.disabled = !available;
    input.setAttribute("aria-label", `${persona?.display_name || profile.name}，${profile.tag}`);
    input.addEventListener("change", () => {
      state.selectedPersonaId = personaId;
      renderPersonaCards();
      elements["creation-message"].textContent = "";
      setWorkspaceView("brief");
      elements.story.focus({ preventScroll: true });
    });

    const inner = document.createElement("span");
    inner.className = "persona-card-inner";

    const top = document.createElement("span");
    top.className = "persona-card-top";
    const monogram = document.createElement("span");
    monogram.className = "persona-monogram";
    monogram.setAttribute("aria-hidden", "true");
    monogram.textContent = profile.monogram;
    const tag = document.createElement("span");
    tag.className = "persona-tag";
    tag.textContent = profile.tag;
    top.append(monogram, tag);

    const title = document.createElement("h3");
    title.textContent = persona?.display_name || profile.name;
    const version = document.createElement("span");
    version.className = "persona-version";
    version.textContent = persona ? `人格版本 ${persona.version}` : "当前服务未提供";
    const bio = document.createElement("p");
    bio.className = "persona-bio";
    bio.textContent = profile.bio;

    const facts = document.createElement("dl");
    facts.className = "persona-facts";
    facts.append(
      createFact("擅长", profile.specialty),
      createFact("适合", profile.suitable),
    );

    const availability = document.createElement("span");
    availability.className = "persona-availability";
    availability.textContent = available ? "● 可选择" : "○ 暂不可用";

    inner.append(top, title, version, bio, facts, availability);
    label.append(input, inner);
    grid.append(label);
  }
}

function createFact(term, description) {
  const wrapper = document.createElement("div");
  const dt = document.createElement("dt");
  const dd = document.createElement("dd");
  dt.textContent = term;
  dd.textContent = description;
  wrapper.append(dt, dd);
  return wrapper;
}

async function handleCreate(event) {
  event.preventDefault();
  const story = elements.story.value.trim();
  const requirements = elements.requirements.value.trim();

  if (!state.selectedPersonaId) {
    elements["creation-message"].textContent = "请先选择一位当前可用的编剧人格。";
    focusFirstAvailablePersona();
    return;
  }
  if (!story) {
    elements["creation-message"].textContent = "请填写故事。";
    elements.story.focus();
    return;
  }
  setCreationBusy(true);
  elements["creation-message"].textContent = "正在向本地编辑部投递……";
  stopPolling();

  try {
    const accepted = await apiRequest("/creations", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Idempotency-Key": createIdempotencyKey("creation"),
      },
      body: JSON.stringify({
        persona_id: state.selectedPersonaId,
        story,
        requirements: requirements || DEFAULT_REQUIREMENTS,
      }),
    });

    state.creationId = accepted.creation_id;
    state.creation = null;
    state.activeVersion = "initial";
    state.activeArtifact = "story_outline";
    state.activeDraftRunKind = "";
    state.activeEpisode = null;
    state.pendingFeedback = "";
    writeCurrentCreationId(state.creationId);
    setWorkspaceView("progress");
    elements["creation-message"].textContent = "投递成功，正在读取真实任务状态。";
    renderSeries();
    renderCreation();
    await refreshCreation();
    focusDelivery();
  } catch (error) {
    elements["creation-message"].textContent = formatError(error);
    setServiceState("error", "请求未完成");
  } finally {
    setCreationBusy(false);
  }
}

async function handleRevision(event) {
  event.preventDefault();
  const revision = state.creation?.revision;
  const feedback = elements.feedback.value;

  if (!feedback.trim()) {
    elements["revision-message"].textContent = "请填写非空的修改意见。";
    elements.feedback.focus();
    return;
  }
  if (state.pendingFeedback) {
    elements["revision-message"].textContent = "修改意见已经提交，正在读取真实修订状态。";
    return;
  }
  if (!revision || revision.state !== "available") {
    elements["revision-message"].textContent = "当前作品不能提交修订。";
    return;
  }

  state.pendingFeedback = feedback;
  setRevisionBusy(true);
  elements["revision-message"].textContent = "正在提交一次全量重写……";

  try {
    await apiRequest(`/creations/${encodeURIComponent(state.creationId)}/revision`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Idempotency-Key": createIdempotencyKey("revision"),
      },
      body: JSON.stringify({ feedback }),
    });
    setWorkspaceView("progress");
    elements["revision-message"].textContent = "修改意见已冻结，正在读取真实修订状态。";
    await refreshCreation();
  } catch (error) {
    const refreshed = await refreshCreation();
    const definitivelyRejected =
      error instanceof ApiError &&
      error.status >= 400 &&
      refreshed &&
      state.creation?.revision.state === "available";

    if (definitivelyRejected) {
      state.pendingFeedback = "";
      renderRevision();
      elements["revision-message"].textContent = formatError(error);
    } else {
      elements["revision-message"].textContent =
        `${formatError(error)} 提交结果尚未确认；请刷新页面读取服务端状态，勿重复提交。`;
    }
  } finally {
    setRevisionBusy(false);
  }
}

async function handleRunControl(action, options = {}) {
  const runKind = options.runKind || state.progressRunKind;
  const messageElement = options.messageElement || elements["run-control-message"];
  if (!runKind || state.runControlBusy) {
    return;
  }
  if (
    action === "end" &&
    !window.confirm("结束后，本次任务将不能继续。确定结束吗？")
  ) {
    return;
  }

  state.runControlBusy = true;
  renderProgress();
  renderQualityRejectionControls();
  if (messageElement) {
    let actionMessage = "正在结束本次任务……";
    if (action === "continue") {
      actionMessage = "正在恢复当前阶段……";
    } else if (action === "retry-final-review") {
      actionMessage = "正在重新提交成品审核……";
    }
    messageElement.textContent = actionMessage;
  }
  try {
    await apiRequest(
      `/creations/${encodeURIComponent(state.creationId)}/runs/${runKind}/${action}`,
      {
        method: "POST",
        headers: {
          "Idempotency-Key": createIdempotencyKey(`run-${runKind}-${action}`),
        },
      },
    );
    await refreshCreation();
  } catch (error) {
    if (messageElement) {
      messageElement.textContent = formatError(error);
    }
  } finally {
    state.runControlBusy = false;
    renderProgress();
    renderQualityRejectionControls();
  }
}

async function handleQualityRejectionControl(action) {
  const rejected = qualityRejectedRun();
  if (!rejected) {
    return;
  }
  await handleRunControl(action, {
    runKind: rejected.kind,
    messageElement: elements["quality-rejection-action-message"],
  });
}

async function refreshCreation(options = {}) {
  if (!state.creationId || state.loadingCreation) {
    return false;
  }

  state.loadingCreation = true;
  stopPolling();
  let refreshed = false;

  try {
    const resource = await apiRequest(
      `/creations/${encodeURIComponent(state.creationId)}`,
    );
    state.creation = resource;
    refreshed = true;
    if (options.isRestore) {
      state.workspaceView = hasReadableDelivery() ? "reading" : "progress";
    }
    setServiceState("ready", "本地服务已连接");
    renderSeries();
    renderCreation();

    if (shouldPoll()) {
      schedulePoll();
    }
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) {
      clearCurrentCreationId();
      state.creationId = "";
      state.creation = null;
      renderSeries();
      renderCreation();
      showToast("保存的作品编号在当前本地数据中不存在。");
    } else {
      setServiceState("error", "状态读取失败");
      showToast(formatError(error));
      if (!options.isRestore) {
        schedulePoll();
      }
    }
  } finally {
    state.loadingCreation = false;
  }
  return refreshed;
}

function renderCreation() {
  renderWorkspaceViews();
  if (!state.creationId) {
    elements["delivery-section"].hidden = true;
    elements["run-progress"].hidden = true;
    return;
  }

  elements["delivery-section"].hidden = false;
  elements["folio-stamp"].textContent = `卷宗 ${shortId(state.creationId)}`;

  if (!state.creation) {
    renderProgress();
    showWaiting(
      "任务已接收",
      "正在读取编辑部状态",
      "本页正在查询本地服务；尚未取得真实状态或成品。",
    );
    return;
  }

  const initial = state.creation.initial;
  elements["delivery-subtitle"].textContent =
    `${state.creation.persona.display_name} · 人格版本 ${state.creation.persona.version}`;
  if (requiresAttentionScene()) {
    setWorkspaceView("progress");
  } else if (shouldOpenReader()) {
    setWorkspaceView("reading");
  }
  renderProgress();

  const activeDraft = activeDraftRun();
  if (activeDraft && state.activeDraftRunKind !== activeDraft.kind) {
    state.activeVersion = activeDraft.kind;
    state.activeArtifact = artifactViewsForRun(activeDraft.run)[0]?.key || "story_outline";
    state.activeEpisode = null;
  }
  state.activeDraftRunKind = activeDraft?.kind || "";

  const rejected = qualityRejectedRun();
  if (rejected) {
    const showWorkspace = renderWorkspace();
    showQualityRejection(rejected, { showWorkspace });
    return;
  }

  if (initial.state === "queued") {
    const showWorkspace = renderWorkspace();
    showWaiting(
      "任务已排队",
      "编辑部已收到你的故事",
      "本页会持续查询本地服务；已提交草稿会在下方保持可读，成品通过审核前不会开启成品阅览。",
      { showWorkspace },
    );
    return;
  }

  if (initial.state === "running") {
    const showWorkspace = renderWorkspace();
    showWaiting(
      "任务创作中",
      "编辑部正在处理当前阶段",
      "上方阶段与已运行时长均来自本地服务；已完成阶段和已提交分集草稿可在下方查看。",
      { showWorkspace },
    );
    return;
  }

  if (initial.state === "auto_resuming") {
    const showWorkspace = renderWorkspace();
    const relayInterrupted = initial.progress.recovery_reason === "relay_interruption";
    showWaiting(
      relayInterrupted ? "网络 / Relay 暂时中断 · 自动恢复" : "首次超时 · 自动恢复",
      relayInterrupted ? "正在等待后从已批准检查点继续" : "正在从已批准检查点继续",
      relayInterrupted
        ? "当前阶段、已运行时长和已提交草稿均已保留；未批准阶段将在短暂等待后重新执行。"
        : "已完成阶段和已提交草稿不会重新生成；当前未批准阶段将重新执行。",
      { showWorkspace },
    );
    return;
  }

  if (initial.state === "paused") {
    const showWorkspace = renderWorkspace();
    const relayInterrupted = initial.progress.recovery_reason === "relay_interruption";
    showWaiting(
      "任务已暂停",
      relayInterrupted ? "当前阶段再次发生网络 / Relay 中断" : "当前阶段再次超过整体运行时限",
      "请在上方选择继续当前阶段，或结束本次任务；已完成阶段、分集草稿与已运行时长均已保留。",
      { showWorkspace },
    );
    return;
  }

  if (initial.state === "ended") {
    const showWorkspace = renderWorkspace();
    showEnded("初稿任务已结束", { showWorkspace });
    return;
  }

  if (initial.state === "failed") {
    const showWorkspace = renderWorkspace();
    showFailure(initial.failure, "初稿生成失败", {
      canStartNewCreation: true,
      showWorkspace,
    });
    return;
  }

  elements["task-waiting"].hidden = true;
  elements["failure-panel"].hidden = true;
  renderWorkspace();
}

function activeProgressRun() {
  if (!state.creation) {
    return null;
  }
  if (state.creation.initial.state !== "succeeded") {
    return state.creation.initial.progress
      ? { kind: "initial", run: state.creation.initial }
      : null;
  }
  const revision = state.creation.revision;
  if (revision.progress && revision.state !== "succeeded") {
    return { kind: "revision", run: revision };
  }
  return null;
}

function qualityRejectedRun() {
  if (!state.creation) {
    return null;
  }
  if (state.creation.initial.state === "quality_rejected") {
    return { kind: "initial", run: state.creation.initial };
  }
  if (state.creation.revision?.state === "quality_rejected") {
    return { kind: "revision", run: state.creation.revision };
  }
  return null;
}

function renderProgress() {
  const active = activeProgressRun();
  if (!active) {
    state.progressRunKind = "";
    elements["run-progress"].hidden = true;
    return;
  }

  const { kind, run } = active;
  const progress = run.progress;
  const artifactKeys = new Set(artifactViewsForRun(run).map(({ key }) => key));
  state.progressRunKind = kind;
  elements["run-progress"].hidden = false;
  elements["progress-kind"].textContent = kind === "initial" ? "初稿进度" : "修订进度";
  elements["progress-title"].textContent =
    USER_STAGE_LABELS.get(progress.current_stage) || "正在读取阶段";
  elements["progress-elapsed"].textContent = formatElapsed(progress.elapsed_seconds);

  const completed = new Set(progress.completed_stages);
  for (const item of elements["progress-stages"].querySelectorAll("[data-stage]")) {
    const stage = item.dataset.stage;
    const artifactKey = STAGE_ARTIFACT_KEYS.get(stage);
    const status = completed.has(stage)
      ? "completed"
      : stage === progress.current_stage
        ? "current"
        : "pending";
    item.dataset.status = status;
    if (artifactKey) {
      item.dataset.artifact = artifactKey;
    } else {
      delete item.dataset.artifact;
    }
    item.dataset.reading = String(artifactKey === state.activeArtifact);
    item.disabled = !artifactKey || !artifactKeys.has(artifactKey);
    if (status === "current") {
      item.setAttribute("aria-current", "step");
    } else {
      item.removeAttribute("aria-current");
    }
  }

  const showReview =
    progress.current_stage === "final_review" ||
    progress.final_review.l0 !== "pending" ||
    progress.final_review.l4 !== "pending";
  elements["review-progress"].hidden = !showReview;
  elements["review-l0"].textContent =
    `L0 创作内核 · ${REVIEW_STATUS_LABELS[progress.final_review.l0]}`;
  elements["review-l4"].textContent =
    `L4 技法与价值观 · ${REVIEW_STATUS_LABELS[progress.final_review.l4]}`;

  const controllable = progress.can_continue || progress.can_end;
  elements["run-controls"].hidden = !controllable;
  const relayInterrupted = progress.recovery_reason === "relay_interruption";
  elements["run-control-title"].textContent = relayInterrupted
    ? "本阶段已两次发生网络 / Relay 中断"
    : "本阶段已两次超过整体运行时限";
  elements["run-control-description"].textContent = relayInterrupted
    ? "可从当前未批准阶段继续；已完成阶段、时长和已提交草稿不会重新生成或丢失，也可以结束本次任务。"
    : "可从当前未批准阶段继续，已完成阶段不会重新生成；也可以结束本次任务。";
  elements["continue-run"].hidden = !progress.can_continue;
  elements["continue-run"].disabled = state.runControlBusy || !progress.can_continue;
  elements["end-run"].hidden = !progress.can_end;
  elements["end-run"].disabled = state.runControlBusy || !progress.can_end;
  if (!controllable) {
    elements["run-control-message"].textContent = "";
  }
}

function showWaiting(kicker, title, description, options = {}) {
  resetQualityRejectionPresentation();
  elements["task-waiting"].hidden = false;
  elements["failure-panel"].hidden = true;
  elements["result-workspace"].hidden = options.showWorkspace !== true;
  elements["wait-kicker"].textContent = kicker;
  elements["wait-title"].textContent = title;
  elements["wait-description"].textContent = description;
}

function showFailure(failure, title, options = {}) {
  resetQualityRejectionPresentation();
  elements["task-waiting"].hidden = true;
  elements["failure-panel"].hidden = false;
  elements["result-workspace"].hidden = options.showWorkspace !== true;
  elements["failure-label"].textContent = "任务未完成";
  elements["failure-title"].textContent = title;
  elements["failure-message"].textContent = failure?.message || "本地服务未提供失败说明。";
  const canStartNewCreation = options.canStartNewCreation === true;
  elements["failure-guidance"].hidden = !canStartNewCreation;
  elements["failure-guidance"].textContent = canStartNewCreation
    ? "本次任务已停止，刷新页面不会自动重试。请重新填写故事并投递新的任务。"
    : "";
  elements["failure-code"].textContent = failure?.code
    ? `错误代码：${failure.code}`
    : "错误代码：未提供";
  elements["failure-actions"].hidden = !canStartNewCreation;
}

function showQualityRejection(rejected, options = {}) {
  resetQualityRejectionPresentation();
  const rejection = rejected.run.quality_rejection || {};
  const stage = rejection.stage;
  const stageLabel = qualityReviewStageLabel(stage);
  const runLabel = rejected.kind === "revision" ? "修订稿" : "初稿";
  const evidence =
    typeof rejection.evidence === "string" && rejection.evidence.trim()
      ? rejection.evidence.trim()
      : "旧版本任务未保存审核证据；请查看保留工作区后再决定是否重新审核。";
  const attempt = Number.isInteger(rejection.attempt_count)
    ? `审核尝试：第 ${rejection.attempt_count} 次`
    : "审核尝试：服务端未提供次数。";
  const canRetry = canRetryQualityReview(rejected);

  elements["task-waiting"].hidden = true;
  elements["failure-panel"].hidden = false;
  elements["result-workspace"].hidden = options.showWorkspace !== true;
  elements["failure-label"].textContent = "成品审核未通过";
  elements["failure-title"].textContent = `${runLabel} ${stageLabel}未通过`;
  elements["failure-message"].textContent =
    `${runLabel}在${stageLabel}未通过。审核证据已保留；成品阅览会继续保持关闭。`;
  elements["failure-guidance"].hidden = false;
  elements["failure-guidance"].textContent =
    canRetry
      ? "请根据审核证据选择重新审核，或明确结束本次任务。"
      : "该审核关已达到三次上限；工作区仍保留，请结束本次任务并据此处理。";
  elements["quality-rejection-details"].hidden = false;
  elements["quality-rejection-stage"].textContent = `审核关卡：${stageLabel}`;
  elements["quality-rejection-evidence"].textContent = `审核证据：${evidence}`;
  elements["quality-rejection-attempt"].textContent = attempt;
  elements["failure-code"].textContent = "状态：quality_rejected";
  elements["failure-actions"].hidden = true;
  renderQualityRejectionControls();
}

function qualityReviewStageLabel(stage) {
  if (stage === "accepting_l0") {
    return "L0 创作内核审核";
  }
  if (stage === "accepting_l4") {
    return "L4 技法与价值观审核";
  }
  return "成品审核";
}

function renderQualityRejectionControls() {
  const actions = elements["quality-rejection-actions"];
  const retry = elements["retry-final-review"];
  const end = elements["end-quality-rejected-run"];
  const message = elements["quality-rejection-action-message"];
  if (!actions || !retry || !end) {
    return;
  }

  const rejected = qualityRejectedRun();
  const canRetry = rejected !== null && canRetryQualityReview(rejected);
  actions.hidden = !rejected;
  retry.hidden = !rejected || !canRetry;
  retry.disabled = !rejected || state.runControlBusy || !canRetry;
  end.disabled = !rejected || state.runControlBusy;
  if (!rejected && message) {
    message.textContent = "";
  }
}

function canRetryQualityReview(rejected) {
  return rejected?.run?.quality_rejection?.can_retry === true;
}

function resetQualityRejectionPresentation() {
  if (elements["quality-rejection-details"]) {
    elements["quality-rejection-details"].hidden = true;
  }
  if (elements["quality-rejection-actions"]) {
    elements["quality-rejection-actions"].hidden = true;
  }
  if (elements["quality-rejection-action-message"]) {
    elements["quality-rejection-action-message"].textContent = "";
  }
}

function showEnded(title, options = {}) {
  resetQualityRejectionPresentation();
  elements["task-waiting"].hidden = true;
  elements["failure-panel"].hidden = false;
  elements["result-workspace"].hidden = options.showWorkspace !== true;
  elements["failure-label"].textContent = "任务已结束";
  elements["failure-title"].textContent = title;
  elements["failure-message"].textContent =
    "你已结束暂停中的任务；已批准检查点保留，但本次任务不能再次继续。";
  elements["failure-guidance"].hidden = true;
  elements["failure-guidance"].textContent = "";
  elements["failure-code"].textContent = "状态：ended";
  elements["failure-actions"].hidden = true;
}

function startNewCreation() {
  stopPolling();
  clearCurrentCreationId();
  state.creationId = "";
  state.creation = null;
  state.activeVersion = "initial";
  state.activeArtifact = "story_outline";
  state.activeDraftRunKind = "";
  state.activeEpisode = null;
  state.pendingFeedback = "";
  setWorkspaceView("selection");
  renderSeries();
  renderCreation();
  elements["creation-message"].textContent =
    "前一次任务未自动恢复。请重新填写故事并投递新的任务。";
  elements["creation-form"].scrollIntoView({ behavior: "smooth", block: "start" });
  elements.story.focus({ preventScroll: true });
}

function activeDraftRun() {
  if (!state.creation) {
    return null;
  }
  const initial = state.creation.initial;
  if (initial.state !== "succeeded" && artifactViewsForRun(initial).length) {
    return { kind: "initial", run: initial };
  }
  const revision = state.creation.revision;
  if (revision.state !== "succeeded" && artifactViewsForRun(revision).length) {
    return { kind: "revision", run: revision };
  }
  return null;
}

function isFormalRun(run) {
  return Boolean(run && run.state === "succeeded" && run.result && run.result.content_package);
}

function artifactViewsForRun(run) {
  if (isFormalRun(run)) {
    return FORMAL_ARTIFACTS.flatMap((artifact) => {
      const content = run.result.content_package[artifact.key];
      return typeof content === "string" && content.trim()
        ? [{ ...artifact, content, isDraft: false }]
        : [];
    });
  }

  const drafts = run?.drafts?.artifacts;
  if (!Array.isArray(drafts)) {
    return episodeScriptDraftView(run);
  }
  const artifacts = drafts.flatMap((draft) => {
    const artifact = DRAFT_ARTIFACTS.find((item) => item.stage === draft?.stage);
    const content = draftArtifactContent(draft);
    return artifact && content
      ? [{ ...artifact, content, isDraft: true }]
      : [];
  });
  return [...artifacts, ...episodeScriptDraftView(run)];
}

function visibleArtifactViews(run) {
  if (state.workspaceView === "progress") {
    return artifactViewsForRun(run);
  }
  return state.workspaceView === "reading" && isFormalRun(run)
    ? artifactViewsForRun(run)
    : [];
}

function episodeScriptDraftView(run) {
  const episodes = run?.progress?.episodes;
  if (!episodes || !Number.isInteger(episodes.total) || episodes.total < 1) {
    return [];
  }
  return [{ ...EPISODE_SCRIPTS_DRAFT, content: "", isDraft: true }];
}

function episodeDraftsForRun(run) {
  return Array.isArray(run?.drafts?.episodes) ? run.drafts.episodes : [];
}

function draftArtifactContent(draft) {
  if (draft?.stage === "determining_direction") {
    const variant = draft.selected_l0_variant;
    const rationale = draft.selection_rationale;
    if (typeof variant !== "string" || !variant.trim() || typeof rationale !== "string" || !rationale.trim()) {
      return "";
    }
    return `选择的 L0 变体：${variant}\n\n选择理由：${rationale}`;
  }
  return typeof draft?.content === "string" && draft.content.trim() ? draft.content : "";
}

function renderWorkspace() {
  const initialArtifacts = visibleArtifactViews(state.creation.initial);
  const revisionArtifacts = visibleArtifactViews(state.creation.revision);
  if (!initialArtifacts.length && !revisionArtifacts.length) {
    elements["result-workspace"].hidden = true;
    return false;
  }

  elements["result-workspace"].hidden = false;
  if (elements["result-workspace"].dataset) {
    elements["result-workspace"].dataset.mode = state.workspaceView;
  }
  elements["revision-desk"].hidden =
    state.workspaceView !== "reading" || state.creation.initial.state !== "succeeded";
  renderVersionControls();
  renderArtifact();
  if (!elements["revision-desk"].hidden) {
    renderRevision();
  }
  return true;
}

function renderVersionControls() {
  const initialAvailable = visibleArtifactViews(state.creation.initial).length > 0;
  const revisionAvailable = visibleArtifactViews(state.creation.revision).length > 0;
  if (state.activeVersion === "initial" && !initialAvailable) {
    state.activeVersion = "revision";
  }
  if (state.activeVersion === "revision" && !revisionAvailable) {
    state.activeVersion = "initial";
  }

  elements["version-initial"].disabled = !initialAvailable;
  elements["version-revision"].disabled = !revisionAvailable;
  for (const button of elements["version-tabs"].querySelectorAll("[data-version]")) {
    const active = button.dataset.version === state.activeVersion;
    button.setAttribute("aria-selected", String(active));
    button.tabIndex = active ? 0 : -1;
  }

  const run = state.activeVersion === "revision"
    ? state.creation.revision
    : state.creation.initial;
  if (!isFormalRun(run)) {
    elements["version-note"].textContent =
      `${draftLabel(state.activeVersion)}，尚未通过成品审核。`;
    return;
  }
  elements["version-note"].textContent =
    state.activeVersion === "revision" ? "正在查看修订后的完整交付" : "正在查看首次交付";
}

function renderArtifact() {
  const run = state.activeVersion === "revision"
    ? state.creation.revision
    : state.creation.initial;
  const artifacts = visibleArtifactViews(run);
  if (!artifacts.length) {
    return;
  }
  let artifact = artifacts.find((item) => item.key === state.activeArtifact);
  if (!artifact) {
    [artifact] = artifacts;
    state.activeArtifact = artifact.key;
  }

  for (const button of elements["artifact-tabs"].querySelectorAll("[data-artifact]")) {
    const index = artifacts.findIndex((item) => item.key === button.dataset.artifact);
    const active = index >= 0 && button.dataset.artifact === artifact.key;
    button.hidden = index < 0;
    button.disabled = index < 0;
    button.setAttribute("aria-selected", String(active));
    button.tabIndex = active ? 0 : -1;
    const number = button.querySelector?.("span");
    if (number && index >= 0) {
      number.textContent = String(index + 1).padStart(2, "0");
    }
  }

  const activeTab = elements["artifact-tabs"].querySelector(
    `[data-artifact="${artifact.key}"]`,
  );
  if (activeTab) {
    elements["artifact-panel"].setAttribute("aria-labelledby", activeTab.id);
  }
  elements["artifact-overline"].textContent = artifact.overline;
  elements["artifact-title"].textContent = artifact.title;
  elements["artifact-version-mark"].textContent = artifact.isDraft
    ? draftLabel(state.activeVersion)
    : state.activeVersion === "revision"
      ? "修订稿"
      : "初稿";
  const showEpisodeNavigator = artifact.isEpisodeNavigator === true;
  elements["episode-navigator"].hidden = !showEpisodeNavigator;
  elements["artifact-content"].hidden = showEpisodeNavigator;
  if (showEpisodeNavigator) {
    renderEpisodeNavigator(run);
    return;
  }
  elements["artifact-content"].textContent = artifact.content;
}

function renderEpisodeNavigator(run) {
  const progress = run?.progress?.episodes;
  if (!progress) {
    elements["episode-navigator"].hidden = true;
    return;
  }
  elements["episode-navigator"].hidden = false;

  const drafts = episodeDraftsForRun(run);
  const draftByEpisode = new Map(
    drafts.map((draft) => [draft.episode_number, draft]),
  );
  const availableEpisodes = drafts
    .map((draft) => draft.episode_number)
    .filter((episodeNumber) => episodeNumber >= 1 && episodeNumber <= progress.total);
  if (!availableEpisodes.includes(state.activeEpisode)) {
    state.activeEpisode = availableEpisodes.includes(progress.current)
      ? progress.current
      : availableEpisodes[0] || null;
  }

  const summary = [
    `共 ${progress.total} 集`,
    `已完成 ${progress.completed} 集`,
  ];
  if (progress.current !== null && progress.current !== undefined) {
    summary.push(`当前第 ${progress.current} 集`);
  }
  if (run.state === "ended") {
    summary.push("任务已结束");
  } else if (run.state === "failed") {
    summary.push("任务失败");
  }
  elements["episode-progress-summary"].textContent = summary.join(" · ");

  const buttons = [];
  for (let episodeNumber = 1; episodeNumber <= progress.total; episodeNumber += 1) {
    const draft = draftByEpisode.get(episodeNumber);
    const active = draft !== undefined && episodeNumber === state.activeEpisode;
    const button = document.createElement("button");
    button.id = `episode-tab-${episodeNumber}`;
    button.type = "button";
    button.setAttribute("role", "tab");
    button.setAttribute("aria-controls", "episode-content");
    button.setAttribute("aria-selected", String(active));
    button.dataset.episodeNumber = String(episodeNumber);
    button.dataset.status = draft
      ? "completed"
      : episodeNumber === progress.current
        ? "current"
        : "pending";
    button.disabled = !draft;
    button.tabIndex = active ? 0 : -1;
    button.textContent = `第 ${episodeNumber} 集`;
    buttons.push(button);
  }
  elements["episode-tabs"].replaceChildren(...buttons);

  const activeDraft = draftByEpisode.get(state.activeEpisode);
  elements["episode-content"].textContent = activeDraft
    ? activeDraft.content
    : "本地服务尚未提交可阅览的分集草稿。";
  if (activeDraft) {
    elements["episode-content"].setAttribute(
      "aria-labelledby",
      `episode-tab-${state.activeEpisode}`,
    );
  } else {
    elements["episode-content"].removeAttribute("aria-labelledby");
  }
}

function draftLabel(kind) {
  return kind === "revision" ? "修订中草稿" : "创作中草稿";
}

function renderRevision() {
  const revision = state.creation.revision;
  const feedback = elements.feedback;
  const submit = elements["revision-button"];
  let disabled = true;
  let feedbackState = "尚未提交";
  let description = "修改意见会冻结，并使用同一人格快照完整重跑五类文稿；初稿不会被覆盖。";
  let buttonLabel = "提交全量重写";
  let message = "";

  if (revision.state === "available" && !state.pendingFeedback) {
    disabled = false;
  } else if (revision.state === "available") {
    feedbackState = "意见已冻结";
    buttonLabel = "等待状态同步";
    description = "修改意见已被服务端接收；本页正在重新读取修订状态。";
    message = "修改意见已提交，等待状态同步。";
  } else if (revision.state === "queued") {
    feedbackState = "意见已冻结";
    buttonLabel = "修订已排队";
    description = "修订任务已排队。初稿仍可浏览；本页会继续查询真实状态。";
    message = "修订任务已排队。";
  } else if (revision.state === "running") {
    feedbackState = "意见已冻结";
    buttonLabel = "修订创作中";
    description = "修订正在真实运行。初稿和已提交的修订草稿仍可浏览；未提交内容不会显示。";
    message = "修订创作中。";
  } else if (revision.state === "auto_resuming") {
    feedbackState = "自动恢复中";
    buttonLabel = "自动恢复中";
    description =
      revision.progress.recovery_reason === "relay_interruption"
        ? "网络 / Relay 暂时中断后，修订将在短暂等待后从已批准检查点继续；初稿仍可浏览。"
        : "首次整体超时后，修订正在从已批准检查点自动继续；初稿仍可浏览。";
    message = "修订正在自动恢复。";
  } else if (revision.state === "paused") {
    feedbackState = "修订已暂停";
    buttonLabel = "修订已暂停";
    description =
      revision.progress.recovery_reason === "relay_interruption"
        ? "当前阶段再次发生网络 / Relay 中断。请使用上方进度卡继续或结束；初稿仍可浏览。"
        : "当前阶段再次超时。请使用上方进度卡继续或结束；初稿仍可浏览。";
    message = "修订等待你的决定。";
  } else if (revision.state === "ended") {
    feedbackState = "修订已结束";
    buttonLabel = "修订已结束";
    description = "暂停中的修订已由你结束，不能再次继续；初稿仍可浏览。";
    message = "本次修订已结束。";
  } else if (revision.state === "quality_rejected") {
    feedbackState = "成品审核未通过";
    buttonLabel = "审核未通过";
    description = "修订工作区已保留；请在上方查看审核证据，并选择重新审核或结束任务。";
    message = "修订稿正在等待你的审核决定。";
  } else if (revision.state === "failed") {
    feedbackState = "修订失败";
    buttonLabel = "修订失败";
    description =
      "原意见已被服务端锁定；本地原型不保存意见，因此不提供可能改变原文的重试。初稿仍可浏览。";
    message = `${revision.failure.message}（${revision.failure.code}）`;
  } else if (revision.state === "succeeded") {
    feedbackState = "修订已完成";
    buttonLabel = "修订已完成";
    description = "一次修订额度已使用；可用上方版本按钮在初稿与修订稿之间切换。";
    message = "修订稿已交付。";
  } else {
    feedbackState = "初稿尚未完成";
    buttonLabel = "等待初稿";
    description = "初稿成功交付后，才可提交一次全量重写。";
  }

  feedback.disabled = disabled;
  submit.disabled = disabled;
  submit.textContent = buttonLabel;
  elements["feedback-state"].textContent = feedbackState;
  elements["revision-description"].textContent = description;
  elements["revision-message"].textContent = message;

  if (state.pendingFeedback && !feedback.value) {
    feedback.value = state.pendingFeedback;
  }
}

function renderSeries() {
  const hasId = Boolean(state.creationId);
  elements["series-empty"].hidden = hasId;
  elements["series-card"].hidden = !hasId;
  if (!hasId) {
    return;
  }

  const status = currentSeriesStatus();
  elements["series-status"].textContent = status.label;
  elements["series-card"].dataset.tone = status.tone;
  elements["series-persona"].textContent =
    state.creation?.persona.display_name || "正在读取人格";
  elements["series-id"].textContent = `卷宗编号 ${shortId(state.creationId)}`;

  const createdAt = state.creation?.created_at;
  elements["series-date"].textContent = createdAt
    ? formatDate(createdAt)
    : "等待服务端记录";
  elements["series-date"].dateTime = createdAt || "";
}

function currentSeriesStatus() {
  if (!state.creation) {
    return { label: "读取中", tone: "waiting" };
  }
  if (state.creation.initial.state === "failed") {
    return { label: "初稿失败", tone: "failed" };
  }
  if (state.creation.initial.state === "quality_rejected") {
    return { label: "初稿审核未通过", tone: "failed" };
  }
  if (state.creation.initial.state === "ended") {
    return { label: "初稿已结束", tone: "failed" };
  }
  if (state.creation.initial.state === "paused") {
    return { label: "初稿待决定", tone: "waiting" };
  }
  if (["queued", "running", "auto_resuming"].includes(state.creation.initial.state)) {
    return {
      label:
        state.creation.initial.state === "queued"
          ? "初稿排队"
          : state.creation.initial.state === "auto_resuming"
            ? "初稿恢复中"
            : "初稿创作中",
      tone: "waiting",
    };
  }
  if (state.creation.revision.state === "failed") {
    return { label: "修订失败", tone: "failed" };
  }
  if (state.creation.revision.state === "quality_rejected") {
    return { label: "修订审核未通过", tone: "failed" };
  }
  if (state.creation.revision.state === "ended") {
    return { label: "修订已结束", tone: "failed" };
  }
  if (state.creation.revision.state === "paused") {
    return { label: "修订待决定", tone: "waiting" };
  }
  if (["queued", "running", "auto_resuming"].includes(state.creation.revision.state)) {
    return {
      label:
        state.creation.revision.state === "queued"
          ? "修订排队"
          : state.creation.revision.state === "auto_resuming"
            ? "修订恢复中"
            : "修订创作中",
      tone: "waiting",
    };
  }
  if (state.creation.revision.state === "succeeded") {
    return { label: "修订稿已交付", tone: "ready" };
  }
  return { label: "初稿已交付", tone: "ready" };
}

function handleVersionClick(event) {
  const button = event.target.closest("[data-version]");
  if (!button || button.disabled) {
    return;
  }
  state.activeVersion = button.dataset.version;
  renderVersionControls();
  renderArtifact();
}

function handleArtifactClick(event) {
  const button = event.target.closest("[data-artifact]");
  if (!button) {
    return;
  }
  state.activeArtifact = button.dataset.artifact;
  renderArtifact();
}

function handleStageClick(event) {
  const button = event.target.closest("[data-stage]");
  const artifactKey = button?.dataset.artifact;
  if (!button || button.disabled || !artifactKey) {
    return;
  }
  state.activeVersion = state.progressRunKind || state.activeVersion;
  state.activeArtifact = artifactKey;
  renderProgress();
  renderWorkspace();
  elements["artifact-panel"].focus?.({ preventScroll: true });
}

function handleEpisodeClick(event) {
  const button = event.target.closest("[data-episode-number]");
  if (!button || button.disabled) {
    return;
  }
  const episodeNumber = Number(button.dataset.episodeNumber);
  state.activeEpisode = episodeNumber;
  renderArtifact();
  elements["episode-tabs"]
    .querySelector(`[data-episode-number="${episodeNumber}"]`)
    ?.focus();
}

function handleHorizontalTabs(event) {
  if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
    return;
  }

  const buttons = [...event.currentTarget.querySelectorAll('[role="tab"]:not(:disabled)')];
  const currentIndex = buttons.indexOf(document.activeElement);
  if (currentIndex < 0) {
    return;
  }

  event.preventDefault();
  let nextIndex;
  if (event.key === "Home") {
    nextIndex = 0;
  } else if (event.key === "End") {
    nextIndex = buttons.length - 1;
  } else {
    const direction = event.key === "ArrowRight" ? 1 : -1;
    nextIndex = (currentIndex + direction + buttons.length) % buttons.length;
  }
  buttons[nextIndex].focus();
  buttons[nextIndex].click();
}

function focusFirstAvailablePersona() {
  const input = elements["persona-grid"].querySelector(".persona-radio:not(:disabled)");
  input?.focus();
}

function focusDelivery() {
  const targetView = hasReadableDelivery() ? "reading" : "progress";
  if (!setWorkspaceView(targetView)) {
    return;
  }
  if (elements["delivery-section"].hidden) {
    return;
  }
  elements["delivery-section"].scrollIntoView({ behavior: "smooth", block: "start" });
  elements["delivery-section"].focus?.({ preventScroll: true });
}

function setWorkspaceView(view) {
  if (view === "brief" && !state.selectedPersonaId) {
    return false;
  }
  if (view === "progress" && !state.creationId) {
    return false;
  }
  if (view === "reading" && !hasReadableDelivery()) {
    return false;
  }
  state.workspaceView = view;
  renderWorkspaceViews();
  return true;
}

function renderWorkspaceViews() {
  const selectionView = elements["selection-view"];
  const briefView = elements["brief-view"];
  const currentView = elements["current-work-view"];
  const progressScene = elements["progress-scene"];
  const flowButtons = [
    ["flow-select-writer", "selection", true],
    ["flow-tell-story", "brief", Boolean(state.selectedPersonaId)],
    ["flow-create", "progress", Boolean(state.creationId)],
    ["flow-read-deliverables", "reading", hasReadableDelivery()],
  ];
  if (!selectionView || !briefView || !currentView || !progressScene) {
    return;
  }

  if (state.workspaceView === "brief" && !state.selectedPersonaId) {
    state.workspaceView = "selection";
  }
  if (state.workspaceView === "progress" && !state.creationId) {
    state.workspaceView = "selection";
  }
  if (state.workspaceView === "reading" && !hasReadableDelivery()) {
    state.workspaceView = state.creationId ? "progress" : "selection";
  }
  const showingCurrent = ["progress", "reading"].includes(state.workspaceView);
  selectionView.hidden = state.workspaceView !== "selection";
  briefView.hidden = state.workspaceView !== "brief";
  currentView.hidden = !showingCurrent;
  currentView.dataset.scene = state.workspaceView;
  progressScene.hidden = state.workspaceView === "reading";
  if (elements["delivery-title"]) {
    elements["delivery-title"].textContent =
      state.workspaceView === "reading" ? "成品阅览室" : "创作进行中";
  }
  if (elements["delivery-index"]) {
    elements["delivery-index"].textContent = state.workspaceView === "reading" ? "04" : "03";
  }
  for (const [id, view, available] of flowButtons) {
    const button = elements[id];
    if (!button) {
      continue;
    }
    button.disabled = !available;
    button.setAttribute("aria-current", state.workspaceView === view ? "page" : "false");
  }

  const placeholder = elements["current-work-placeholder"];
  if (placeholder) {
    placeholder.hidden = Boolean(state.creation);
  }
  const briefPersona = elements["brief-persona"];
  if (briefPersona) {
    const profile = PERSONA_PROFILES[state.selectedPersonaId];
    const persona = state.personas.get(state.selectedPersonaId);
    briefPersona.textContent = profile
      ? `你选择了 —— ${persona?.display_name || profile.name} · ${profile.tag}`
      : "先选择一位当前可用的编剧人格。";
  }
}

function hasReadableDelivery() {
  return isFormalRun(state.creation?.initial) || isFormalRun(state.creation?.revision);
}

function shouldOpenReader() {
  if (state.workspaceView !== "progress" || !isFormalRun(state.creation?.initial)) {
    return false;
  }
  const revisionState = state.creation?.revision?.state;
  return ["available", "unavailable", "succeeded"].includes(revisionState);
}

function requiresAttentionScene() {
  const initialState = state.creation?.initial?.state;
  if (initialState && initialState !== "succeeded") {
    return true;
  }
  return ["paused", "quality_rejected", "ended", "failed"].includes(
    state.creation?.revision?.state,
  );
}

function shouldPoll() {
  if (!state.creation) {
    return Boolean(state.creationId);
  }
  return (
    ["queued", "running", "auto_resuming"].includes(state.creation.initial.state) ||
    ["queued", "running", "auto_resuming"].includes(state.creation.revision.state)
  );
}

function schedulePoll() {
  stopPolling();
  state.pollTimer = window.setTimeout(() => {
    state.pollTimer = null;
    void refreshCreation();
  }, POLL_INTERVAL_MS);
}

function stopPolling() {
  if (state.pollTimer !== null) {
    window.clearTimeout(state.pollTimer);
    state.pollTimer = null;
  }
}

async function apiRequest(path, options = {}) {
  let response;
  try {
    response = await fetch(path, {
      ...options,
      headers: {
        Accept: "application/json",
        ...(options.headers || {}),
      },
    });
  } catch {
    throw new ApiError("无法连接本地服务，请确认 Pengine 已启动。", "network_error", 0);
  }

  const contentType = response.headers.get("content-type") || "";
  let payload = null;
  if (contentType.includes("application/json")) {
    try {
      payload = await response.json();
    } catch {
      payload = null;
    }
  }

  if (!response.ok) {
    throw new ApiError(
      typeof payload?.message === "string" ? payload.message : `请求失败（HTTP ${response.status}）。`,
      typeof payload?.code === "string" ? payload.code : `http_${response.status}`,
      response.status,
    );
  }

  if (payload === null) {
    throw new ApiError("本地服务返回了无法读取的响应。", "invalid_response", response.status);
  }
  return payload;
}

function formatError(error) {
  if (error instanceof ApiError) {
    return `${error.message}（${error.code}）`;
  }
  return "发生未预期错误，请重新读取本地服务状态。";
}

function setCreationBusy(busy) {
  elements["create-button"].disabled = busy;
  elements["create-button"].querySelector("span").textContent = busy
    ? "正在投递"
    : "送入编辑部";
  elements["creation-form"].setAttribute("aria-busy", String(busy));
}

function setRevisionBusy(busy) {
  if (busy) {
    elements.feedback.disabled = true;
    elements["revision-button"].disabled = true;
    elements["revision-button"].textContent = "正在提交";
  } else {
    const revisionState = state.creation?.revision.state;
    const canSubmit =
      revisionState === "available" && !state.pendingFeedback;
    const lockedLabels = {
      unavailable: "等待初稿",
      queued: "修订已排队",
      running: "修订创作中",
      auto_resuming: "自动恢复中",
      paused: "修订已暂停",
      ended: "修订已结束",
      quality_rejected: "审核未通过",
      failed: "修订失败",
      succeeded: "修订已完成",
    };
    elements.feedback.disabled = !canSubmit;
    elements["revision-button"].disabled = !canSubmit;
    elements["revision-button"].textContent =
      state.pendingFeedback
        ? "意见已冻结"
        : lockedLabels[revisionState] || "提交全量重写";
  }
  elements["revision-form"].setAttribute("aria-busy", String(busy));
}

function setServiceState(tone, label) {
  elements["service-stamp"].dataset.tone = tone;
  elements["service-label"].textContent = label;
}

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.hidden = false;
  window.setTimeout(() => {
    elements.toast.hidden = true;
  }, 5000);
}

function createIdempotencyKey(scope) {
  if (typeof crypto.randomUUID === "function") {
    return `web-${scope}-${crypto.randomUUID()}`;
  }
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  const token = [...bytes].map((byte) => byte.toString(16).padStart(2, "0")).join("");
  return `web-${scope}-${token}`;
}

function readCurrentCreationId() {
  try {
    return window.localStorage.getItem(STORAGE_KEY) || "";
  } catch {
    return "";
  }
}

function writeCurrentCreationId(creationId) {
  try {
    window.localStorage.setItem(STORAGE_KEY, creationId);
  } catch {
    showToast("浏览器未允许保存当前作品编号；本次页面内仍可继续查看。");
  }
}

function clearCurrentCreationId() {
  try {
    window.localStorage.removeItem(STORAGE_KEY);
  } catch {
    // Storage is optional; the current in-memory state is still cleared.
  }
}

function shortId(value) {
  return value ? value.slice(0, 8).toUpperCase() : "—";
}

function formatElapsed(value) {
  const total = Number.isFinite(value) ? Math.max(0, Math.floor(value)) : 0;
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const seconds = total % 60;
  const parts = [
    String(minutes).padStart(2, "0"),
    String(seconds).padStart(2, "0"),
  ];
  if (hours > 0) {
    parts.unshift(String(hours).padStart(2, "0"));
  }
  return parts.join(":");
}

function formatDate(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "服务端时间不可读";
  }
  return new Intl.DateTimeFormat("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}
