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

const ARTIFACTS = [
  { key: "story_outline", title: "故事大纲", overline: "DELIVERABLE 01" },
  { key: "character_biographies", title: "人物小传", overline: "DELIVERABLE 02" },
  { key: "relationship_logic", title: "关系逻辑", overline: "DELIVERABLE 03" },
  { key: "episode_outline", title: "分集大纲", overline: "DELIVERABLE 04" },
  { key: "episode_scripts", title: "分集剧本", overline: "DELIVERABLE 05" },
];

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
  pollTimer: null,
  loadingCreation: false,
  pendingFeedback: "",
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
    "delivery-section",
    "delivery-subtitle",
    "folio-stamp",
    "task-waiting",
    "wait-kicker",
    "wait-title",
    "wait-description",
    "failure-panel",
    "failure-title",
    "failure-message",
    "failure-code",
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
  elements["series-card"].addEventListener("click", focusDelivery);

  elements["version-tabs"].addEventListener("click", handleVersionClick);
  elements["version-tabs"].addEventListener("keydown", handleHorizontalTabs);
  elements["artifact-tabs"].addEventListener("click", handleArtifactClick);
  elements["artifact-tabs"].addEventListener("keydown", handleHorizontalTabs);

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
    state.pendingFeedback = "";
    writeCurrentCreationId(state.creationId);
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
  if (!state.creationId) {
    elements["delivery-section"].hidden = true;
    return;
  }

  elements["delivery-section"].hidden = false;
  elements["folio-stamp"].textContent = `卷宗 ${shortId(state.creationId)}`;

  if (!state.creation) {
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

  if (initial.state === "queued") {
    showWaiting(
      "任务已排队",
      "编辑部已收到你的故事",
      "本页会持续查询本地服务；后端返回成功或失败前，不会展示假成品。",
    );
    return;
  }

  if (initial.state === "running") {
    showWaiting(
      "任务创作中",
      "真实创作仍在进行",
      "本页只知道任务正在运行，不推测内部阶段。完成后会读取服务端交付的五类文稿。",
    );
    return;
  }

  if (initial.state === "failed") {
    showFailure(initial.failure, "初稿生成失败");
    return;
  }

  elements["task-waiting"].hidden = true;
  elements["failure-panel"].hidden = true;
  elements["result-workspace"].hidden = false;
  renderVersionControls();
  renderArtifact();
  renderRevision();
}

function showWaiting(kicker, title, description) {
  elements["task-waiting"].hidden = false;
  elements["failure-panel"].hidden = true;
  elements["result-workspace"].hidden = true;
  elements["wait-kicker"].textContent = kicker;
  elements["wait-title"].textContent = title;
  elements["wait-description"].textContent = description;
}

function showFailure(failure, title) {
  elements["task-waiting"].hidden = true;
  elements["failure-panel"].hidden = false;
  elements["result-workspace"].hidden = true;
  elements["failure-title"].textContent = title;
  elements["failure-message"].textContent = failure?.message || "本地服务未提供失败说明。";
  elements["failure-code"].textContent = failure?.code
    ? `错误代码：${failure.code}`
    : "错误代码：未提供";
}

function renderVersionControls() {
  const revisionSucceeded = state.creation.revision.state === "succeeded";
  if (!revisionSucceeded && state.activeVersion === "revision") {
    state.activeVersion = "initial";
  }

  elements["version-revision"].disabled = !revisionSucceeded;
  for (const button of elements["version-tabs"].querySelectorAll("[data-version]")) {
    const active = button.dataset.version === state.activeVersion;
    button.setAttribute("aria-selected", String(active));
    button.tabIndex = active ? 0 : -1;
  }

  elements["version-note"].textContent =
    state.activeVersion === "revision" ? "正在查看修订后的完整交付" : "正在查看首次交付";
}

function renderArtifact() {
  const artifact = ARTIFACTS.find((item) => item.key === state.activeArtifact) || ARTIFACTS[0];
  const run = state.activeVersion === "revision"
    ? state.creation.revision
    : state.creation.initial;
  const content = run.result.content_package[artifact.key];

  for (const button of elements["artifact-tabs"].querySelectorAll("[data-artifact]")) {
    const active = button.dataset.artifact === artifact.key;
    button.setAttribute("aria-selected", String(active));
    button.tabIndex = active ? 0 : -1;
  }

  const activeTab = elements["artifact-tabs"].querySelector(
    `[data-artifact="${artifact.key}"]`,
  );
  elements["artifact-panel"].setAttribute("aria-labelledby", activeTab.id);
  elements["artifact-overline"].textContent = artifact.overline;
  elements["artifact-title"].textContent = artifact.title;
  elements["artifact-version-mark"].textContent =
    state.activeVersion === "revision" ? "修订稿" : "初稿";
  elements["artifact-content"].textContent = content;
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
    description = "修订任务已排队。初稿仍可浏览；本页会继续查询真实状态。";
    message = "修订任务已排队。";
  } else if (revision.state === "running") {
    feedbackState = "意见已冻结";
    description = "修订正在真实运行。初稿仍可浏览；完成前不会展示假修订稿。";
    message = "修订创作中。";
  } else if (revision.state === "failed") {
    feedbackState = "修订失败";
    buttonLabel = "修订失败";
    description =
      "原意见已被服务端锁定；本地原型不保存意见，因此不提供可能改变原文的重试。初稿仍可浏览。";
    message = `${revision.failure.message}（${revision.failure.code}）`;
  } else if (revision.state === "succeeded") {
    feedbackState = "修订已完成";
    description = "一次修订额度已使用；可用上方版本按钮在初稿与修订稿之间切换。";
    message = "修订稿已交付。";
  } else {
    feedbackState = "初稿尚未完成";
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
  if (["queued", "running"].includes(state.creation.initial.state)) {
    return {
      label: state.creation.initial.state === "queued" ? "初稿排队" : "初稿创作中",
      tone: "waiting",
    };
  }
  if (state.creation.revision.state === "failed") {
    return { label: "修订失败", tone: "failed" };
  }
  if (["queued", "running"].includes(state.creation.revision.state)) {
    return {
      label: state.creation.revision.state === "queued" ? "修订排队" : "修订创作中",
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
  if (elements["delivery-section"].hidden) {
    return;
  }
  elements["delivery-section"].scrollIntoView({ behavior: "smooth", block: "start" });
  elements["delivery-section"].focus?.({ preventScroll: true });
}

function shouldPoll() {
  if (!state.creation) {
    return Boolean(state.creationId);
  }
  return (
    ["queued", "running"].includes(state.creation.initial.state) ||
    ["queued", "running"].includes(state.creation.revision.state)
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
    elements.feedback.disabled = !canSubmit;
    elements["revision-button"].disabled = !canSubmit;
    elements["revision-button"].textContent =
      state.pendingFeedback
        ? "意见已冻结"
        : revisionState === "failed"
          ? "修订失败"
          : "提交全量重写";
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
