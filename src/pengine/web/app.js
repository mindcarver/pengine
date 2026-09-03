"use strict";

const PERSONA_ORDER = ["shouzhuo", "wuzhen", "sanfentian", "xinggui"];
// #230：当前仅开放第一位人格，其余置灰显示“正在支持中”；恢复支持时向集合追加 id。
const SUPPORTED_PERSONA_IDS = new Set([PERSONA_ORDER[0]]);
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
  { key: "relationship_logic", title: "人物关系", overline: "DELIVERABLE 03" },
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
    key: "character_relationships",
    stage: "generating_character_relationships",
    title: "人物与关系",
    overline: "DRAFT 03",
  },
  {
    key: "episode_outline",
    stage: "generating_episode_outline",
    title: "分集大纲",
    overline: "DRAFT 04",
  },
];

const EPISODE_SCRIPTS_DRAFT = {
  key: "episode_scripts",
  title: "分集剧本",
  overline: "DRAFT 05",
  isEpisodeNavigator: true,
};

const SERIES_BIBLE_DESIGN = {
  key: "series_bible_design",
  title: "设计包",
  overline: "DESIGN",
  isSeriesBibleDesign: true,
};

const STAGE_ARTIFACT_KEYS = new Map([
  ...DRAFT_ARTIFACTS.map(({ stage, key }) => [stage, key]),
  ["generating_episode_scripts", "episode_scripts"],
]);

const USER_STAGES = [
  ["determining_direction", "确定创作方向"],
  ["generating_story_outline", "生成故事大纲"],
  ["generating_character_relationships", "生成人物与关系"],
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
const READING_STORAGE_KEY = "pengine.deliveryReadingPosition";
const POLL_INTERVAL_MS = 1800;
const DEFAULT_REQUIREMENTS = "按所选人格完成一部完整短剧。";

const state = {
  user: null,
  authMode: "login",
  personas: new Map(),
  selectedPersonaId: "",
  creation: null,
  creationId: "",
  activeVersion: "initial",
  activeArtifact: "story_outline",
  activeDraftRunKind: "",
  activeEpisode: null,
  presentations: { initial: null, revision: null },
  presentationLoading: { initial: false, revision: false },
  readingPositions: readReadingPositions(),
  pollTimer: null,
  libraryNotice: "",
  loadingCreation: false,
  pendingFeedback: "",
  progressRunKind: "",
  runControlBusy: false,
  workspaceView: "selection",
  // 暂时隐藏“模型调用用量”面板（#226）；恢复展示时改为 true。
  modelCallPanelEnabled: false,
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
    "auth-view",
    "auth-form",
    "auth-title",
    "auth-intro",
    "auth-username",
    "auth-password",
    "auth-username-error",
    "auth-password-error",
    "auth-message",
    "auth-submit",
    "auth-switch-copy",
    "auth-switch",
    "main-content",
    "page-title",
    "creations-title",
    "account-tools",
    "account-trigger",
    "account-username",
    "account-menu",
    "open-creations",
    "logout",
    "creations-library",
    "creations-status",
    "creation-list",
    "back-to-workbench",
    "workbench-hero",
    "workbench",
    "reload-personas",
    "persona-grid",
    "persona-status",
    "creation-form",
    "story",
    "requirements",
    "creation-message",
    "create-button",
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
    "delivery-section",
    "delivery-title",
    "run-progress",
    "progress-kind",
    "progress-title",
    "progress-elapsed",
    "progress-stages",
    "episode-progress",
    "episode-progress-label",
    "episode-progress-detail",
    "review-progress",
    "review-l0",
    "review-l4",
    "model-call-panel",
    "model-call-totals",
    "model-call-list",
    "run-controls",
    "run-control-title",
    "run-control-description",
    "continue-run",
    "authorize-repair-run",
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
    "quality-rejection-repair",
    "quality-rejection-attempt",
    "failure-code",
    "failure-actions",
    "retry-run",
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
    "export-delivery",
    "open-revision",
    "close-revision",
    "artifact-tabs",
    "artifact-panel",
    "artifact-title",
    "artifact-overline",
    "artifact-version-mark",
    "section-nav",
    "section-items",
    "presentation-status",
    "episode-navigator",
    "episode-progress-summary",
    "episode-tabs",
    "episode-content",
    "artifact-content",
    "episode-stepper",
    "previous-episode",
    "episode-position",
    "next-episode",
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
  elements["auth-form"].addEventListener("submit", handleAuthentication);
  elements["auth-switch"].addEventListener("click", toggleAuthMode);
  elements["account-trigger"].addEventListener("click", toggleAccountMenu);
  elements["open-creations"].addEventListener("click", () => void showCreationsLibrary());
  elements.logout.addEventListener("click", () => void handleLogout());
  elements["back-to-workbench"].addEventListener("click", showWorkbench);
  elements["reload-personas"].addEventListener("click", () => void loadPersonas());
  elements["creation-form"].addEventListener("submit", handleCreate);
  elements["revision-form"].addEventListener("submit", handleRevision);
  elements["continue-run"].addEventListener("click", () => void handleRunControl("continue"));
  if (elements["authorize-repair-run"]) {
    elements["authorize-repair-run"].addEventListener("click", () =>
      void handleRunControl("authorize-repair"),
    );
  }
  elements["end-run"].addEventListener("click", () => void handleRunControl("end"));
  elements["retry-run"].addEventListener("click", () => {
    elements["retry-run"].disabled = true;
    void handleRunControl("retry", {
      runKind: "initial",
      messageElement: elements["failure-guidance"],
    });
  });
  elements["start-new-creation"].addEventListener("click", startNewCreation);
  elements["retry-final-review"].addEventListener("click", () =>
    void handleQualityRejectionControl("retry-final-review"),
  );
  elements["end-quality-rejected-run"].addEventListener("click", () =>
    void handleQualityRejectionControl("end"),
  );
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
  elements["section-items"].addEventListener("click", handleSectionClick);
  elements["section-items"].addEventListener("keydown", handleHorizontalTabs);
  elements["episode-tabs"].addEventListener("click", handleEpisodeClick);
  elements["episode-tabs"].addEventListener("keydown", handleHorizontalTabs);
  elements["export-delivery"].addEventListener("click", handleExportDelivery);
  elements["open-revision"].addEventListener("click", openRevisionDrawer);
  elements["close-revision"].addEventListener("click", closeRevisionDrawer);
  elements["previous-episode"]?.addEventListener("click", () => moveFormalEpisode(-1));
  elements["next-episode"]?.addEventListener("click", () => moveFormalEpisode(1));
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !elements["revision-desk"].hidden) {
      closeRevisionDrawer();
    }
    if (event.key === "Escape" && !elements["account-menu"].hidden) {
      elements["account-menu"].hidden = true;
      elements["account-trigger"].setAttribute("aria-expanded", "false");
      elements["account-trigger"].focus();
    }
  });

  window.addEventListener("beforeunload", stopPolling);
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden && state.creationId && shouldPoll()) {
      void refreshCreation();
    }
  });
}

async function initialize() {
  try {
    state.user = await apiRequest("/me");
  } catch (error) {
    if (error instanceof ApiError && error.status === 401) {
      showAuthentication();
      return;
    }
    elements["auth-message"].textContent = formatError(error);
    showAuthentication();
    return;
  }
  showAuthenticatedWorkspace();
  const currentId = readCurrentCreationId();
  state.creationId = currentId;
  const hashPosition = readReadingHash(currentId);
  state.activeVersion =
    hashPosition?.version || state.readingPositions[`${currentId}:activeVersion`] || "initial";
  state.activeArtifact =
    hashPosition?.artifact ||
    state.readingPositions[`${currentId}:${state.activeVersion}:activeArtifact`] ||
    "story_outline";
  if (hashPosition?.itemId) {
    state.readingPositions[`${currentId}:${state.activeVersion}:${state.activeArtifact}`] =
      hashPosition.itemId;
  }
  state.workspaceView = currentId ? "progress" : "selection";
  renderWorkspaceViews();

  await loadPersonas();
  if (currentId) {
    await refreshCreation({ isRestore: true });
  }
}

function showAuthentication(message = "") {
  stopPolling();
  state.user = null;
  state.creation = null;
  state.creationId = "";
  elements["auth-view"].hidden = false;
  elements["main-content"].hidden = true;
  elements["account-tools"].hidden = true;
  elements["account-menu"].hidden = true;
  elements["account-trigger"].setAttribute("aria-expanded", "false");
  if (message) {
    elements["auth-message"].textContent = message;
  }
  window.setTimeout(() => elements["auth-username"].focus(), 0);
}

function showAuthenticatedWorkspace() {
  elements["auth-view"].hidden = true;
  elements["main-content"].hidden = false;
  elements["account-tools"].hidden = false;
  elements["account-username"].textContent = state.user.username;
  showWorkbench();
}

function toggleAuthMode() {
  state.authMode = state.authMode === "login" ? "register" : "login";
  const registering = state.authMode === "register";
  elements["auth-title"].textContent = registering ? "登记新账户" : "登录创作台";
  elements["auth-intro"].textContent = registering
    ? "用一个用户名和至少八位密码建立账户。"
    : "登录后继续你的创作与修订。";
  elements["auth-submit"].textContent = registering ? "注册并进入" : "登录";
  elements["auth-switch-copy"].textContent = registering ? "已经有账户？" : "还没有账户？";
  elements["auth-switch"].textContent = registering ? "登录" : "注册";
  elements["auth-password"].autocomplete = registering ? "new-password" : "current-password";
  elements["auth-message"].textContent = "";
  elements["auth-username-error"].textContent = "";
  elements["auth-password-error"].textContent = "";
  elements["auth-username"].focus();
}

async function handleAuthentication(event) {
  event.preventDefault();
  const username = elements["auth-username"].value.trim();
  const password = elements["auth-password"].value;
  elements["auth-username-error"].textContent = username ? "" : "请输入用户名。";
  elements["auth-password-error"].textContent =
    password.length >= 8 ? "" : "密码至少需要八位。";
  if (!username || password.length < 8) {
    (username ? elements["auth-password"] : elements["auth-username"]).focus();
    return;
  }

  elements["auth-submit"].disabled = true;
  elements["auth-submit"].textContent = state.authMode === "register" ? "正在登记…" : "正在登录…";
  elements["auth-message"].textContent = "";
  try {
    state.user = await apiRequest(`/auth/${state.authMode}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });
    elements["auth-password"].value = "";
    showAuthenticatedWorkspace();
    await loadPersonas();
    const currentId = readCurrentCreationId();
    state.creationId = currentId;
    state.workspaceView = currentId ? "progress" : "selection";
    renderCreation();
    if (currentId) {
      await refreshCreation({ isRestore: true });
    }
    elements["page-title"]?.focus?.({ preventScroll: true });
  } catch (error) {
    if (error instanceof ApiError && error.code === "username_taken") {
      elements["auth-username-error"].textContent = "这个用户名已被使用。";
      elements["auth-username"].focus();
    } else {
      elements["auth-message"].textContent = formatError(error);
    }
  } finally {
    elements["auth-submit"].disabled = false;
    elements["auth-submit"].textContent = state.authMode === "register" ? "注册并进入" : "登录";
  }
}

function toggleAccountMenu() {
  const opening = elements["account-menu"].hidden;
  elements["account-menu"].hidden = !opening;
  elements["account-trigger"].setAttribute("aria-expanded", String(opening));
}

async function handleLogout() {
  try {
    await apiRequest("/auth/logout", { method: "POST" });
  } finally {
    showAuthentication();
  }
}

function showWorkbench() {
  elements["creations-library"].hidden = true;
  elements["workbench-hero"].hidden = false;
  elements.workbench.hidden = false;
  elements["account-menu"].hidden = true;
  elements["account-trigger"].setAttribute("aria-expanded", "false");
}

async function showCreationsLibrary(options = {}) {
  const { focus = true, background = false } = options;
  elements["account-menu"].hidden = true;
  elements["account-trigger"].setAttribute("aria-expanded", "false");
  elements["workbench-hero"].hidden = true;
  elements.workbench.hidden = true;
  elements["creations-library"].hidden = false;
  if (!background) {
    // 后台轮询只刷新数据不覆盖提示；用户主动打开时才清掉上一条通知。
    state.libraryNotice = "";
    elements["creations-status"].textContent = "正在读取你的创作卷宗……";
    elements["creation-list"].replaceChildren();
  }
  if (focus) {
    elements["creations-title"].focus?.({ preventScroll: true });
  }
  try {
    const payload = await apiRequest("/creations");
    const items = Array.isArray(payload.items) ? payload.items : [];
    renderCreationList(items);
    if (
      items.some(
        (item) => item.initial_state === "queued" || item.revision_state === "queued",
      )
    ) {
      scheduleLibraryPoll();
    }
  } catch (error) {
    elements["creations-status"].textContent = formatError(error);
  }
}

function renderCreationList(items) {
  elements["creation-list"].replaceChildren();
  if (state.libraryNotice) {
    elements["creations-status"].textContent = state.libraryNotice;
  } else if (!items.length) {
    elements["creations-status"].textContent = "还没有创作。返回创作台，开始第一份卷宗。";
    return;
  } else {
    elements["creations-status"].textContent = `共 ${items.length} 份创作。`;
  }
  for (const item of items) {
    const row = document.createElement("div");
    row.className = "creation-file-row";
    const button = document.createElement("button");
    button.type = "button";
    button.className = "creation-file";
    const updated = new Intl.DateTimeFormat("zh-CN", {
      dateStyle: "medium",
      timeStyle: "short",
    }).format(new Date(item.updated_at));
    const queueLabel = Number.isInteger(item.queue_position)
      ? ` · 排队第 ${item.queue_position} 位`
      : "";
    button.innerHTML = `
      <span class="creation-file-meta">${escapeHtml(item.persona_display_name)} · 初稿 ${escapeHtml(item.initial_state)} · 修订 ${escapeHtml(item.revision_state)}${escapeHtml(queueLabel)}</span>
      <strong>故事：${escapeHtml(item.story_excerpt)}</strong>
      <time datetime="${escapeHtml(item.updated_at)}">更新于 ${escapeHtml(updated)}</time>
    `;
    button.addEventListener("click", () => {
      state.creationId = item.creation_id;
      state.creation = null;
      writeCurrentCreationId(item.creation_id);
      state.workspaceView = "progress";
      showWorkbench();
      renderCreation();
      void refreshCreation({ isRestore: true });
    });
    row.append(button, createCreationDeleteButton(item));
    elements["creation-list"].append(row);
  }
}

function createCreationDeleteButton(item) {
  const deleteButton = document.createElement("button");
  deleteButton.type = "button";
  deleteButton.className = "creation-delete";
  const restingLabel = `删除故事：${item.story_excerpt}`;
  const confirmLabel = `确认删除故事：${item.story_excerpt}`;
  deleteButton.textContent = "删除";
  deleteButton.setAttribute("aria-label", restingLabel);
  let confirming = false;
  let confirmTimer = 0;
  const restConfirm = () => {
    if (deleteButton.disabled) {
      return;
    }
    confirming = false;
    window.clearTimeout(confirmTimer);
    deleteButton.classList.remove("confirming");
    deleteButton.setAttribute("aria-label", restingLabel);
    deleteButton.textContent = "删除";
  };
  deleteButton.addEventListener("blur", restConfirm);
  deleteButton.addEventListener("click", () => {
    if (deleteButton.disabled) {
      return;
    }
    if (!confirming) {
      confirming = true;
      deleteButton.classList.add("confirming");
      deleteButton.setAttribute("aria-label", confirmLabel);
      deleteButton.textContent = "确认删除？";
      confirmTimer = window.setTimeout(restConfirm, 5000);
      return;
    }
    window.clearTimeout(confirmTimer);
    confirming = false;
    void deleteCreation(item.creation_id, deleteButton);
  });
  return deleteButton;
}

async function deleteCreation(creationId, deleteButton) {
  deleteButton.disabled = true;
  deleteButton.textContent = "正在删除……";
  try {
    await apiRequest(`/creations/${encodeURIComponent(creationId)}`, {
      method: "DELETE",
    });
    if (state.creationId === creationId) {
      resetDeletedCreation();
    }
    await showCreationsLibrary({ focus: false, background: true });
    state.libraryNotice = "该创作卷宗已删除。";
    elements["creations-status"].textContent = state.libraryNotice;
    elements["creations-title"].focus?.({ preventScroll: true });
  } catch (error) {
    deleteButton.disabled = false;
    deleteButton.classList.remove("confirming");
    deleteButton.textContent = "删除";
    state.libraryNotice =
      error instanceof ApiError && error.code === "creation_not_deletable"
        ? "该创作还有排队或进行中的初稿/修订，请先结束运行再删除。"
        : `删除失败：${formatError(error)}`;
    elements["creations-status"].textContent = state.libraryNotice;
    deleteButton.focus?.({ preventScroll: true });
  }
}

function resetDeletedCreation() {
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
  renderCreation();
}

async function loadPersonas() {
  elements["reload-personas"].disabled = true;
  elements["persona-status"].textContent = "正在读取人格档案……";

  try {
    const payload = await apiRequest("/personas");
    const items = Array.isArray(payload.items) ? payload.items : [];
    state.personas = new Map(items.map((persona) => [persona.persona_id, persona]));

    if (
      !state.personas.has(state.selectedPersonaId) ||
      !SUPPORTED_PERSONA_IDS.has(state.selectedPersonaId)
    ) {
      state.selectedPersonaId = "";
    }

    const availableCount = PERSONA_ORDER.filter((id) => state.personas.has(id)).length;
    const missingCount = PERSONA_ORDER.length - availableCount;
    const gatingNote = `当前仅开放第一位人格（${
      PERSONA_PROFILES[PERSONA_ORDER[0]].name
    }），其余正在支持中。`;
    elements["persona-status"].textContent = missingCount
      ? `已读取 ${availableCount} 位原型人格；${missingCount} 位未被当前服务列为可选。${gatingNote}`
      : `四位原型人格均已由本地服务确认可用。${gatingNote}`;
  } catch (error) {
    state.personas = new Map();
    state.selectedPersonaId = "";
    elements["persona-status"].textContent = formatError(error);
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
    const supported = SUPPORTED_PERSONA_IDS.has(personaId);
    const label = document.createElement("label");
    label.className = "persona-card";
    label.classList.add(`persona-${personaId}`);
    label.dataset.available = String(available);
    label.dataset.supported = String(supported);
    label.dataset.selected = String(state.selectedPersonaId === personaId);

    const input = document.createElement("input");
    input.className = "persona-radio";
    input.type = "radio";
    input.name = "persona_id";
    input.value = personaId;
    input.checked = state.selectedPersonaId === personaId;
    input.disabled = !available || !supported;
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
    availability.textContent = !supported
      ? "◌ 正在支持中"
      : available
        ? "● 可选择"
        : "○ 暂不可用";

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
    state.presentations = { initial: null, revision: null };
    state.presentationLoading = { initial: false, revision: false };
    state.pendingFeedback = "";
    writeCurrentCreationId(state.creationId);
    setWorkspaceView("progress");
    elements["creation-message"].textContent = "投递成功，正在读取真实任务状态。";
    renderCreation();
    await refreshCreation();
    focusDelivery();
  } catch (error) {
    elements["creation-message"].textContent = formatError(error);
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
      actionMessage = "正在按审核证据执行受限修复并重新审核……";
    } else if (action === "authorize-repair") {
      actionMessage = "正在授权一次修复循环……";
    } else if (action === "retry") {
      actionMessage = "正在从已批准进度重试当前任务……";
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
    renderCreation();

    if (shouldPoll()) {
      schedulePoll();
    }
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) {
      clearCurrentCreationId();
      state.creationId = "";
      state.creation = null;
      renderCreation();
      showToast("保存的作品编号在当前本地数据中不存在。");
    } else {
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
  if (requiresAttentionScene()) {
    setWorkspaceView("progress");
  } else if (shouldOpenReader()) {
    setWorkspaceView("reading");
  }
  if (state.workspaceView === "reading" && elements["presentation-status"]) {
    void loadPresentation(state.activeVersion);
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
      `任务已排队 · 第 ${initial.queue_position} 位`,
      "编辑部已收到你的故事；同一账户的任务会依次执行",
      "排队序号按全局提交顺序实时更新；本页会持续查询本地服务。已提交草稿会在下方保持可读，成品通过审核前不会开启成品阅览。",
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
    const contentRejected = initial.progress.recovery_reason === "content_rejected";
    const episodeError = initial.progress.recovery_reason === "episode_error";
    const contextBudget = initial.progress.recovery_reason === "context_budget";
    const identityMismatch =
      initial.progress.recovery_reason === "relay_identity_mismatch";
    const episodeNumber = initial.pause?.episode_number || initial.progress.episodes?.current;
    const retainedEpisodes = initial.progress.episodes?.completed || 0;
    const repairAuthorization = initial.progress.recovery_reason === "repair_authorization";
    const authorization = initial.authorization || null;
    showWaiting(
      repairAuthorization
        ? "任务已暂停 · 等待一次修复授权"
        : contextBudget
        ? "任务已暂停 · 上下文预算不足"
        : identityMismatch
        ? "任务已暂停 · 模型身份待确认"
        : "任务已暂停",
      repairAuthorization
        ? authorization && authorization.kind === "design_rebuild"
          ? "设计需整体重建 · 自动预算已用尽"
          : "分集硬约束需修复 · 自动预算已用尽"
        : contextBudget
        ? "模型请求未发出，已完成的批准内容保持不变"
        : identityMismatch
        ? episodeNumber
          ? `第 ${episodeNumber} 集响应的模型身份验证未通过`
          : "当前响应的模型身份验证未通过"
        : episodeError
        ? `第 ${episodeNumber} 集生成遇到可恢复错误`
        : contentRejected
        ? "内容一致性审查连续修复后仍未通过"
        : relayInterrupted
          ? "当前阶段再次发生网络 / Relay 中断"
          : "当前阶段再次超过整体运行时限",
      repairAuthorization
        ? `${initial.pause?.message || "内容审查未通过。"} 自动修复预算已用尽，需在上方授权一次修复循环。`
        : contextBudget
        ? `${initial.pause?.message || "完整请求超出已验证上下文上限。"} 请配置更高的已验证上限后，在上方继续当前阶段；未发出的请求不消耗额度。`
        : identityMismatch
        ? `${initial.pause?.message || "Relay 未能证明本次响应来自配置模型。"} 本次响应已丢弃，已完成的 ${retainedEpisodes} 集保持不变；确认 Relay 后再继续。`
        : episodeError
        ? `${initial.pause?.message || "当前集生成遇到可恢复错误。"} 已完成的 ${retainedEpisodes} 集已保留，请在上方从第 ${episodeNumber} 集继续或结束任务。`
        : contentRejected
        ? "审查证据已保留。请在上方选择继续重新生成当前未锁内容，或结束本次任务；锁定内容不会改变。"
        : "请在上方选择继续当前阶段，或结束本次任务；已完成阶段、分集草稿与已运行时长均已保留。",
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
      canRetry: initial.progress?.can_retry === true,
      showWorkspace,
      stageLabel: USER_STAGE_LABELS.get(initial.failure?.failed_stage),
    });
    return;
  }

  const revision = state.creation.revision;
  if (revision.state === "failed") {
    const showWorkspace = renderWorkspace();
    showFailure(revision.failure, "修订生成失败", {
      showWorkspace,
      stageLabel: USER_STAGE_LABELS.get(revision.failure?.failed_stage),
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
  const stageLabel = USER_STAGE_LABELS.get(progress.current_stage) || "正在读取阶段";
  elements["progress-title"].textContent =
    run.state === "failed" ? `失败于${stageLabel}` : stageLabel;
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

  renderModelCalls(progress);
  renderEpisodeProgress(progress);

  const controllable = progress.can_continue || progress.can_end;
  elements["run-controls"].hidden = !controllable;
  const relayInterrupted = progress.recovery_reason === "relay_interruption";
  const contentRejected = progress.recovery_reason === "content_rejected";
  const episodeError = progress.recovery_reason === "episode_error";
  const contextBudget = progress.recovery_reason === "context_budget";
  const identityMismatch = progress.recovery_reason === "relay_identity_mismatch";
  const repairAuthorization = progress.recovery_reason === "repair_authorization";
  const authorization = run.authorization || null;
  const episodeNumber = run.pause?.episode_number || progress.episodes?.current;
  const retainedEpisodes = progress.episodes?.completed || 0;
  elements["run-control-title"].textContent = repairAuthorization
    ? authorization && authorization.kind === "design_rebuild"
      ? "设计需整体重建 · 等待授权"
      : "分集硬约束需修复 · 等待授权"
    : contextBudget
    ? "模型上下文预算不足 · 已安全暂停"
    : identityMismatch
    ? episodeNumber
      ? `第 ${episodeNumber} 集模型身份待确认`
      : "模型身份待确认"
    : episodeError
    ? `第 ${episodeNumber} 集可继续`
    : contentRejected
    ? "内容一致性审查已完成两轮修复"
    : relayInterrupted
      ? "本阶段已两次发生网络 / Relay 中断"
      : "本阶段已两次超过整体运行时限";
  elements["run-control-description"].textContent = repairAuthorization
    ? `${run.pause?.message || "内容审查未通过。"} 自动修复预算已用尽，需授权一次修复循环：`
        .concat(
          authorization && authorization.earliest_affected_episode != null
            ? `影响范围：从第 ${authorization.earliest_affected_episode} 集到第 ${
                authorization.earliest_affected_episode +
                (authorization.range_episodes || 1) -
                1
              } 集。`
            : "影响范围：完整设计。",
        )
        .concat(
          authorization && authorization.estimated_tokens != null
            ? ` 参考上下文量：${authorization.estimated_tokens.toLocaleString()} tokens（仅统计暂停时的活动设计投影与保留前缀；不是下限、整轮用量或费用预测）。`
            : "",
        )
        .concat(" 授权将执行一次生成加审查循环；若仍有硬约束冲突，将按最新审查证据再次暂停。")
    : contextBudget
    ? `${run.pause?.message || "请求未发出，未消耗任何额度。"} 请为对应路由配置更高的已验证上下文上限后继续当前阶段。`
    : identityMismatch
    ? `${run.pause?.message || "Relay 未能证明本次响应来自配置模型。"} 本次响应已丢弃，已完成的 ${retainedEpisodes} 集不会重新生成。确认 Relay 后再继续。`
    : episodeError
    ? `${run.pause?.message || "当前集生成遇到可恢复错误。"} 已完成的 ${retainedEpisodes} 集不会重新生成。`
    : contentRejected
    ? `${run.pause?.message || "当前内容仍与锁定合同冲突。"} 可重新生成当前未锁内容，或结束本次任务。`
    : relayInterrupted
      ? "可从当前未批准阶段继续；已完成阶段、时长和已提交草稿不会重新生成或丢失，也可以结束本次任务。"
      : "可从当前未批准阶段继续，已完成阶段不会重新生成；也可以结束本次任务。";
  elements["continue-run"].textContent = repairAuthorization
    ? "授权一次修复"
    : contextBudget
    ? "继续创作"
    : identityMismatch && episodeNumber
    ? `从第 ${episodeNumber} 集继续`
    : episodeError
    ? `从第 ${episodeNumber} 集继续`
    : "继续创作";
  elements["continue-run"].hidden = !progress.can_continue || repairAuthorization;
  elements["continue-run"].disabled = state.runControlBusy || !progress.can_continue;
  const authorizeRepairRun = elements["authorize-repair-run"];
  if (authorizeRepairRun) {
    authorizeRepairRun.hidden = !repairAuthorization || !progress.can_continue;
    authorizeRepairRun.disabled =
      state.runControlBusy || !repairAuthorization || !progress.can_continue;
  }
  elements["end-run"].hidden = !progress.can_end;
  elements["end-run"].disabled = state.runControlBusy || !progress.can_end;
  if (!controllable) {
    elements["run-control-message"].textContent = "";
  }
}

function renderEpisodeProgress(progress) {
  const panel = elements["episode-progress"];
  const label = elements["episode-progress-label"];
  const detail = elements["episode-progress-detail"];
  const groups = progress?.outline_groups;
  if (groups && (groups.committed_groups > 0 || Number.isInteger(groups.current_group))) {
    panel.hidden = false;
    label.textContent = "大纲组进度";
    const parts = [`已提交 ${groups.committed_groups} 组`];
    if (groups.committed_through_episode > 0) {
      parts.push(`覆盖第 1–${groups.committed_through_episode} 集`);
    }
    if (Number.isInteger(groups.current_group)) {
      parts.push(
        `正在生成第 ${groups.current_group} 组（第 ${groups.current_start_episode}–${groups.current_end_episode} 集）`,
      );
    }
    detail.textContent = parts.join(" · ");
    return;
  }
  label.textContent = "分集进度";
  const episodes = progress?.episodes;
  if (!episodes || !Number.isInteger(episodes.total) || episodes.total < 1) {
    panel.hidden = true;
    detail.textContent = "";
    return;
  }
  panel.hidden = false;
  const total = episodes.total;
  const completed = Number.isInteger(episodes.completed) ? episodes.completed : 0;
  if (completed >= total) {
    detail.textContent = `已全部完成 ${total} 集`;
  } else if (Number.isInteger(episodes.current)) {
    detail.textContent = `第 ${episodes.current}/${total} 集 · 已完成 ${completed}`;
  } else {
    detail.textContent = `已完成 ${completed}/${total} 集`;
  }
}

function renderModelCalls(progress) {
  const panel = elements["model-call-panel"];
  if (!state.modelCallPanelEnabled) {
    panel.hidden = true;
    return;
  }
  const calls = progress?.model_calls;
  if (!calls || calls.length === 0) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;

  const totals = {
    succeeded: 0,
    failed: 0,
    timed_out: 0,
    superseded: 0,
    stale: 0,
    preflight_blocked: 0,
  };
  let inputTokens = 0;
  let outputTokens = 0;
  for (const call of calls) {
    const status = call.status || "started";
    if (status in totals) totals[status] += 1;
    if (call.usage?.status === "reported" || call.usage?.status === "partial") {
      if (typeof call.usage.input_tokens === "number") inputTokens += call.usage.input_tokens;
      if (typeof call.usage.output_tokens === "number") outputTokens += call.usage.output_tokens;
    }
  }
  const total = calls.length;
  const success = totals.succeeded;
  const failed = totals.failed + totals.timed_out + totals.superseded + totals.stale;
  const blocked = totals.preflight_blocked;
  elements["model-call-totals"].textContent =
    `共 ${total} 次调用 · 成功 ${success} · 失败 ${failed} · 预检拦截 ${blocked}` +
    (inputTokens + outputTokens > 0 ? ` · 实际用量 输入 ${inputTokens} / 输出 ${outputTokens}` : "");

  const list = elements["model-call-list"];
  list.replaceChildren();
  for (const call of calls) {
    const item = document.createElement("li");
    const usageText = formatCallUsage(call);
    const durationText =
      typeof call.duration_seconds === "number" ? ` · ${formatElapsed(Math.round(call.duration_seconds))}` : "";
    const identityText = Array.isArray(call.response_model_ids)
      ? call.response_model_ids.length > 0
        ? ` · 响应身份 ${call.response_model_ids.join(", ")}`
        : " · 响应身份未携带"
      : "";
    item.textContent =
      `${MODEL_CALL_ROLE_LABELS[call.role] || call.role} ${call.model}` +
      (call.stage ? ` · ${USER_STAGE_LABELS.get(call.stage) || call.stage}` : "") +
      (call.episode_number ? ` · 第 ${call.episode_number} 集` : "") +
      ` · ${MODEL_CALL_STATUS_LABELS[call.status] || call.status}` +
      (call.estimated_total_tokens ? ` · 预估 ${call.estimated_total_tokens}` : "") +
      (call.verified_limit_tokens ? ` / 上限 ${call.verified_limit_tokens}` : "") +
      (usageText ? ` · ${usageText}` : "") +
      (call.finish_reason ? ` · ${call.finish_reason}` : "") +
      identityText +
      durationText;
    item.dataset.callStatus = call.status;
    list.appendChild(item);
  }
}

function formatCallUsage(call) {
  const usage = call.usage || {};
  if (usage.status === "unavailable" || usage.status === undefined) {
    return "用量不可用";
  }
  const parts = [];
  if (typeof usage.input_tokens === "number") parts.push(`输入 ${usage.input_tokens}`);
  if (typeof usage.output_tokens === "number") parts.push(`输出 ${usage.output_tokens}`);
  if (typeof usage.cache_read_tokens === "number" && usage.cache_read_tokens > 0) {
    parts.push(`缓存读 ${usage.cache_read_tokens}`);
  }
  return parts.length > 0 ? `实际 ${parts.join(" · ")}` : "用量不可用";
}

const MODEL_CALL_ROLE_LABELS = { generation: "生成", review: "审核" };
const MODEL_CALL_STATUS_LABELS = {
  started: "进行中",
  succeeded: "成功",
  failed: "失败",
  timed_out: "超时",
  stale: "过期",
  superseded: "被取代",
  preflight_blocked: "预检拦截",
};

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
  const failureMessage = failure?.message || "本地服务未提供失败说明。";
  elements["failure-message"].textContent = options.stageLabel
    ? `${failureMessage} 失败阶段：${options.stageLabel}。`
    : failureMessage;
  const canStartNewCreation = options.canStartNewCreation === true;
  const canRetry = options.canRetry === true;
  const retryButton = elements["retry-run"];
  if (retryButton) {
    retryButton.hidden = !canRetry;
    retryButton.disabled = state.runControlBusy || !canRetry;
  }
  elements["failure-guidance"].hidden = !canStartNewCreation && !canRetry;
  elements["failure-guidance"].textContent = canRetry
    ? "本次任务因外部服务错误停止。修复 relay 后可从这里原样重试：已批准内容不会重新生成；也可以重新开始新任务。"
    : canStartNewCreation
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
      : "旧版本任务未保存审核证据；系统会先尝试把证据绑定到具体剧集与原文。";
  const attempt = Number.isInteger(rejection.attempt_count)
    ? `审核尝试：第 ${rejection.attempt_count} 次`
    : "审核尝试：服务端未提供次数。";
  const canRetry = canRetryQualityReview(rejected);
  const repairPlan = rejection.repair_plan;
  const repairScope = repairPlan?.scope;
  const repairText =
    repairScope === "episode_content"
      ? `修复范围：仅修改审核证据绑定的 ${repairPlan.issues
          .map((issue) => `第 ${issue.episode_number} 集`)
          .filter((value, index, values) => values.indexOf(value) === index)
          .join("、")}原文。`
      : repairScope === "design_rebuild"
        ? "修复范围：证据涉及设计重建，不能执行局部剧本修复。"
        : repairScope === "unresolved"
          ? "修复范围：证据尚不能安全绑定到具体剧集原文。"
          : "修复范围：提交后先将旧审核证据绑定到具体剧集原文。";

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
      ? "可选择执行一次受限修复；系统会验证只改了证据相关内容，再重新审核。"
      : "该审核关已达到三次上限；工作区仍保留，请结束本次任务并据此处理。";
  elements["quality-rejection-details"].hidden = false;
  elements["quality-rejection-stage"].textContent = `审核关卡：${stageLabel}`;
  elements["quality-rejection-evidence"].textContent = `审核证据：${evidence}`;
  elements["quality-rejection-repair"].textContent = repairText;
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
  const rejection = rejected?.run?.quality_rejection;
  return (
    rejection?.can_retry === true &&
    rejection?.repair_state !== "blocked" &&
    !["design_rebuild", "unresolved"].includes(rejection?.repair_plan?.scope)
  );
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

function selectedRun() {
  return state.activeVersion === "revision"
    ? state.creation?.revision
    : state.creation?.initial;
}

async function loadPresentation(kind) {
  const run = kind === "revision" ? state.creation?.revision : state.creation?.initial;
  if (
    !state.creationId ||
    !isFormalRun(run) ||
    state.presentations[kind] ||
    state.presentationLoading[kind]
  ) {
    return;
  }
  state.presentationLoading[kind] = true;
  if (!state.presentations[kind] && elements["presentation-status"]) {
    elements["presentation-status"].textContent = "正在整理";
  }
  try {
    const presentation = await apiRequest(
      `/creations/${encodeURIComponent(state.creationId)}/runs/${kind}/presentation`,
    );
    state.presentations[kind] = presentation;
  } catch (error) {
    if (!state.presentations[kind] && elements["presentation-status"]) {
      elements["presentation-status"].textContent = "完整原文";
    }
    if (elements.toast) {
      showToast(`成品目录读取失败：${formatError(error)}`);
    }
  } finally {
    state.presentationLoading[kind] = false;
    if (state.workspaceView === "reading" && state.activeVersion === kind) {
      renderArtifact();
    }
  }
}

function presentationArtifact(key) {
  return state.presentations[state.activeVersion]?.[key] || null;
}

function presentationItems(artifact) {
  if (!artifact || artifact.mode !== "structured") {
    return [];
  }
  if (Array.isArray(artifact.sections)) return artifact.sections;
  if (Array.isArray(artifact.characters)) return artifact.characters;
  if (Array.isArray(artifact.relationships)) return artifact.relationships;
  if (Array.isArray(artifact.episodes)) return artifact.episodes;
  return [];
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
  const design = seriesBibleDesignView(run);
  if (design) {
    artifacts.push(design);
  }
  return [...artifacts, ...episodeScriptDraftView(run)];
}

function seriesBibleDesignView(run) {
  const design = run?.drafts?.design;
  if (!design || typeof design !== "object") {
    return null;
  }
  const sections = [
    "未完成设计包（不作为正式交付）",
    "",
    `候选 ID: ${design.candidate_id || ""}`,
    `版本: ${design.version ?? ""}`,
    `内容哈希: ${design.content_hash || ""}`,
    `类型: ${design.genre || ""}`,
    design.is_active ? "状态: 当前激活" : `状态: ${design.status || ""}`,
    "",
    "故事大纲:",
    design.projections?.story_outline || "",
    "",
    "人物小传:",
    design.projections?.character_biographies || "",
    "",
    "关系逻辑:",
    design.projections?.relationship_logic || "",
    "",
    "分集大纲:",
    design.projections?.episode_outline || "",
  ];
  return { ...SERIES_BIBLE_DESIGN, content: sections.join("\n"), isDraft: true };
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
  const canRevise = state.workspaceView === "reading" && state.creation.initial.state === "succeeded";
  if (elements["open-revision"]) {
    elements["open-revision"].hidden = !canRevise;
  }
  if (!canRevise) {
    closeRevisionDrawer();
  }
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

  const run = selectedRun();
  renderExportControl(run);
  if (!isFormalRun(run)) {
    elements["version-note"].textContent =
      `${draftLabel(state.activeVersion)}，尚未通过成品审核。`;
    return;
  }
  elements["version-note"].textContent =
    state.activeVersion === "revision" ? "正在查看修订后的完整交付" : "正在查看首次交付";
}

function renderExportControl(run = selectedRun()) {
  const available = isFormalRun(run);
  elements["export-delivery"].hidden = !available;
  elements["export-delivery"].disabled = !available;
}

function createDeliveryExport(run, { creationId, kind, persona, exportedAt = new Date() }) {
  if (!isFormalRun(run)) {
    throw new Error("delivery_not_formal");
  }
  const sections = FORMAL_ARTIFACTS.map((artifact, index) => {
    const content = run.result.content_package[artifact.key];
    if (typeof content !== "string" || !content.trim()) {
      throw new Error(`delivery_artifact_missing:${artifact.key}`);
    }
    return `## ${String(index + 1).padStart(2, "0")} ${artifact.title}\n\n${content.trim()}`;
  });
  const versionLabel = kind === "revision" ? "修订稿" : "初稿";
  const exportedTime = exportedAt instanceof Date ? exportedAt : new Date(exportedAt);
  if (Number.isNaN(exportedTime.getTime())) {
    throw new Error("delivery_export_time_invalid");
  }
  const metadata = [
    "# 意态短剧成品包",
    "",
    `- 卷宗编号：${shortId(creationId)}`,
    `- 稿件版本：${versionLabel}`,
    `- 编剧人格：${persona?.display_name || "未标注"}`,
    `- 人格版本：${persona?.version || "未标注"}`,
    `- 导出时间：${exportedTime.toISOString()}`,
    "",
    "---",
  ];
  const safeId = shortId(creationId).replace(/[^0-9A-Z_-]/g, "-");
  return {
    content: [...metadata, "", ...sections].join("\n") + "\n",
    filename: `意态短剧_${safeId}_${versionLabel}.md`,
  };
}

function handleExportDelivery() {
  try {
    const run = selectedRun();
    const exported = createDeliveryExport(run, {
      creationId: state.creationId,
      kind: state.activeVersion,
      persona: state.creation?.persona,
    });
    const url = URL.createObjectURL(
      new Blob([`\uFEFF${exported.content}`], { type: "text/markdown;charset=utf-8" }),
    );
    const link = document.createElement("a");
    link.href = url;
    link.download = exported.filename;
    link.hidden = true;
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
    showToast(`已导出${state.activeVersion === "revision" ? "修订稿" : "初稿"}成品。`);
  } catch {
    showToast("当前版本没有完整正式成品，无法导出。");
  }
}

function openRevisionDrawer() {
  if (!elements["revision-desk"] || !elements["close-revision"]) {
    return;
  }
  elements["revision-desk"].hidden = false;
  if (document.body) {
    document.body.dataset.revisionOpen = "true";
  }
  renderRevision();
  elements["close-revision"].focus();
}

function closeRevisionDrawer() {
  if (!elements["revision-desk"]) {
    return;
  }
  const wasOpen = !elements["revision-desk"].hidden;
  elements["revision-desk"].hidden = true;
  if (document.body) {
    delete document.body.dataset.revisionOpen;
  }
  if (wasOpen && state.workspaceView === "reading" && elements["open-revision"]) {
    elements["open-revision"].focus();
  }
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
  if (elements["episode-stepper"]) {
    elements["episode-stepper"].hidden = true;
  }
  elements["episode-navigator"].hidden = !showEpisodeNavigator;
  elements["artifact-content"].hidden = showEpisodeNavigator;
  if (showEpisodeNavigator) {
    if (elements["section-nav"]) {
      elements["section-nav"].hidden = true;
    }
    renderEpisodeNavigator(run);
    return;
  }
  if (artifact.isDraft || state.workspaceView !== "reading" || !elements["section-nav"]) {
    elements["artifact-content"].textContent = artifact.content;
    return;
  }
  const projected = presentationArtifact(artifact.key);
  const items = presentationItems(projected);
  const source = projected?.source_text || artifact.content;
  const positionKey = `${state.creationId}:${state.activeVersion}:${artifact.key}`;
  let activeItem = items.find((item) => item.id === state.readingPositions[positionKey]);
  if (!activeItem && items.length) {
    [activeItem] = items;
    state.readingPositions[positionKey] = activeItem.id;
    writeReadingPositions();
  }
  renderSectionNavigation(items, activeItem, projected);
  renderReadableText(elements["artifact-content"], activeItem?.content || source);
  renderEpisodeStepper(items, activeItem, artifact.key);
}

function renderEpisodeStepper(items, activeItem, artifactKey) {
  const isEpisodeArtifact = artifactKey === "episode_outline" || artifactKey === "episode_scripts";
  if (!elements["episode-stepper"] || !isEpisodeArtifact || items.length < 1 || !activeItem) {
    return;
  }
  const index = items.findIndex((item) => item.id === activeItem.id);
  elements["episode-stepper"].hidden = false;
  elements["previous-episode"].disabled = index <= 0;
  elements["next-episode"].disabled = index >= items.length - 1;
  elements["episode-position"].textContent = `第 ${index + 1} / ${items.length} 集`;
}

function moveFormalEpisode(offset) {
  const projected = presentationArtifact(state.activeArtifact);
  const items = presentationItems(projected);
  const positionKey = `${state.creationId}:${state.activeVersion}:${state.activeArtifact}`;
  const currentIndex = items.findIndex(
    (item) => item.id === state.readingPositions[positionKey],
  );
  const target = items[currentIndex + offset];
  if (!target) {
    return;
  }
  state.readingPositions[positionKey] = target.id;
  writeReadingPositions();
  writeReadingHash(target.id);
  renderArtifact();
  elements["artifact-content"].focus({ preventScroll: true });
}

function renderSectionNavigation(items, activeItem, projected) {
  const nav = elements["section-nav"];
  nav.hidden = items.length === 0;
  elements["presentation-status"].textContent = projected?.mode === "structured"
    ? "已按结构整理"
    : "完整原文";
  if (!items.length) {
    elements["section-items"].replaceChildren();
    return;
  }
  const buttons = items.map((item) => {
    const button = document.createElement("button");
    const itemLevel = Number(item.level);
    const level = Number.isInteger(itemLevel) && itemLevel >= 1 && itemLevel <= 3 ? itemLevel : 1;
    button.type = "button";
    button.role = "tab";
    button.dataset.sectionId = item.id;
    button.dataset.level = String(level);
    button.setAttribute("aria-controls", "artifact-content");
    button.setAttribute("aria-selected", String(item.id === activeItem?.id));
    button.tabIndex = item.id === activeItem?.id ? 0 : -1;
    const number = document.createElement("span");
    number.textContent = String(item.ordinal).padStart(2, "0");
    const label = document.createElement("strong");
    label.textContent = item.label;
    button.append(number, label);
    return button;
  });
  elements["section-items"].replaceChildren(...buttons);
}

function renderReadableText(container, content) {
  const blocks = String(content || "")
    .split(/\n\s*\n/)
    .map((block) => block.trim())
    .filter(Boolean)
    .map((block) => {
      const paragraph = document.createElement("p");
      paragraph.textContent = block;
      return paragraph;
    });
  container.replaceChildren(...blocks);
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
    description = `修订任务当前排队第 ${revision.queue_position} 位。同一账户的任务会依次执行；初稿仍可浏览，本页会继续查询真实状态。`;
    message = `修订任务已排队 · 第 ${revision.queue_position} 位。`;
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
      revision.progress.recovery_reason === "content_rejected"
        ? "内容一致性审查连续修复后仍未通过。请使用上方进度卡重新生成当前未锁内容或结束；初稿仍可浏览。"
        : revision.progress.recovery_reason === "relay_identity_mismatch"
        ? "Relay 响应的模型身份验证未通过。响应已丢弃；请确认 Relay 后使用上方进度卡继续或结束。"
        : revision.progress.recovery_reason === "relay_interruption"
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

function handleVersionClick(event) {
  const button = event.target.closest("[data-version]");
  if (!button || button.disabled) {
    return;
  }
  state.activeVersion = button.dataset.version;
  state.readingPositions[`${state.creationId}:activeVersion`] = state.activeVersion;
  state.activeArtifact =
    state.readingPositions[`${state.creationId}:${state.activeVersion}:activeArtifact`] ||
    "story_outline";
  writeReadingPositions();
  writeReadingHash();
  renderVersionControls();
  renderArtifact();
  void loadPresentation(state.activeVersion);
}

function handleArtifactClick(event) {
  const button = event.target.closest("[data-artifact]");
  if (!button) {
    return;
  }
  state.activeArtifact = button.dataset.artifact;
  state.readingPositions[`${state.creationId}:${state.activeVersion}:activeArtifact`] =
    state.activeArtifact;
  writeReadingPositions();
  writeReadingHash();
  renderArtifact();
  elements["artifact-content"].focus({ preventScroll: true });
}

function handleSectionClick(event) {
  const button = event.target.closest("[data-section-id]");
  if (!button) {
    return;
  }
  const positionKey = `${state.creationId}:${state.activeVersion}:${state.activeArtifact}`;
  state.readingPositions[positionKey] = button.dataset.sectionId;
  writeReadingPositions();
  writeReadingHash(button.dataset.sectionId);
  renderArtifact();
  elements["artifact-content"].focus({ preventScroll: true });
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
  if (view === "reading" && elements["presentation-status"]) {
    void loadPresentation(state.activeVersion);
  }
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
  if (document.body) {
    document.body.dataset.workspaceView = state.workspaceView;
  }
  selectionView.hidden = state.workspaceView !== "selection";
  briefView.hidden = state.workspaceView !== "brief";
  currentView.hidden = !showingCurrent;
  currentView.dataset.scene = state.workspaceView;
  progressScene.hidden = state.workspaceView === "reading";
  if (elements["delivery-title"]) {
    elements["delivery-title"].textContent =
      state.workspaceView === "reading" ? "成品阅览室" : "创作进行中";
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

function scheduleLibraryPoll() {
  stopPolling();
  state.pollTimer = window.setTimeout(() => {
    state.pollTimer = null;
    if (!elements["creations-library"].hidden) {
      void showCreationsLibrary({ focus: false, background: true });
    }
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
    throw new ApiError("无法连接本地服务，请确认意态短剧已启动。", "network_error", 0);
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
    if (response.status === 401 && path !== "/me" && !path.startsWith("/auth/")) {
      showAuthentication("登录已失效，请重新登录。未提交的内容仍保留在本页。 ");
    }
    throw new ApiError(
      typeof payload?.message === "string" ? payload.message : `请求失败（HTTP ${response.status}）。`,
      typeof payload?.code === "string" ? payload.code : `http_${response.status}`,
      response.status,
    );
  }

  if (response.status === 204) {
    return null;
  }
  if (payload === null) {
    throw new ApiError("本地服务返回了无法读取的响应。", "invalid_response", response.status);
  }
  return payload;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
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

function readReadingPositions() {
  try {
    const value = JSON.parse(window.localStorage.getItem(READING_STORAGE_KEY) || "{}");
    return value && typeof value === "object" && !Array.isArray(value) ? value : {};
  } catch {
    return {};
  }
}

function writeReadingPositions() {
  try {
    window.localStorage.setItem(READING_STORAGE_KEY, JSON.stringify(state.readingPositions));
  } catch {
    // Reading position is optional and never affects the delivered content.
  }
}

function readReadingHash(creationId) {
  try {
    const params = new URLSearchParams(window.location.hash.slice(1));
    const value = JSON.parse(params.get("read") || "null");
    if (
      value?.creationId !== creationId ||
      !["initial", "revision"].includes(value?.version) ||
      !FORMAL_ARTIFACTS.some((artifact) => artifact.key === value?.artifact)
    ) {
      return null;
    }
    return value;
  } catch {
    return null;
  }
}

function writeReadingHash(itemId = null) {
  try {
    const positionKey = `${state.creationId}:${state.activeVersion}:${state.activeArtifact}`;
    const params = new URLSearchParams(window.location.hash.slice(1));
    params.set(
      "read",
      JSON.stringify({
        creationId: state.creationId,
        version: state.activeVersion,
        artifact: state.activeArtifact,
        itemId: itemId || state.readingPositions[positionKey] || null,
      }),
    );
    window.history.replaceState(
      null,
      "",
      `${window.location.pathname}${window.location.search}#${params.toString()}`,
    );
  } catch {
    // URL state is optional; localStorage remains the primary restoration path.
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
