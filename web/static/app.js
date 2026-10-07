"use strict";

const $ = (sel) => document.querySelector(sel);
const FINISHED = new Set(["done", "failed", "cancelled", "interrupted"]);
const MODEL_NAMES = {
  "nvidia/nemotron-3-super-120b-a12b": "Nemotron 3 Super 120B",
  "nvidia/nemotron-3-ultra-550b-a55b": "Nemotron 3 Ultra 550B",
  "deepseek-ai/deepseek-v4.1-flash": "DeepSeek v4.1-flash",
  "qwen/qwen2.5-coder-32b-instruct": "Qwen2.5-Coder 32B",
};
const TOOL_LABEL = { read_file: "讀檔", search_code: "搜尋", run_tests: "跑測試", propose_patch: "提修正" };
const STATUS_TEXT = { failed: "失敗", cancelled: "已停止", interrupted: "已中斷(伺服器曾重開)" };
const state = { workspaces: [], current: null, running: null, streams: new Map(), views: [] };

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value == null) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value);
  }
  for (const child of children) if (child != null) node.append(child);
  return node;
}

async function api(path, options = {}) {
  const headers = options.body && !(options.body instanceof FormData) ? { "Content-Type": "application/json" } : {};
  const res = await fetch(path, { ...options, headers });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : `伺服器回應 ${res.status}`);
  return data;
}

const fmt = (n) => Number(n || 0).toLocaleString("zh-TW");
const modelName = (id) => MODEL_NAMES[id] || (id || "").split("/").pop();
const renderMarkdown = (text) => DOMPurify.sanitize(marked.parse(text || ""));
const currentMode = () => document.querySelector('input[name="mode"]:checked').value;
const thinkingValue = () => ({ true: true, false: false })[$("#thinking").value] ?? null;
const workspaceById = (id) => state.workspaces.find((w) => w.id === id);

function hint(text, isError = false) {
  const box = $("#hint");
  box.textContent = text || "";
  box.classList.toggle("error", isError);
}

function setSending(running) {
  const send = $("#send");
  send.textContent = running ? "停止" : "送出";
  send.classList.toggle("stop", running);
  if (running) send.disabled = false;
}

// ---------- 資料夾與紀錄 ----------

async function loadWorkspaces(selectId) {
  state.workspaces = await api("/api/workspaces");
  const select = $("#workspace");
  const keep = selectId || select.value;
  select.replaceChildren(...state.workspaces.map((w) => el("option", { value: w.id, text: w.name })));
  if (keep && !workspaceById(keep)) select.append(el("option", { value: keep, text: "(資料夾已刪除)" }));
  if (keep) select.value = keep;
}

async function loadHistory() {
  const items = await api("/api/conversations");
  const nav = $("#history");
  nav.replaceChildren();
  const today = new Date().toDateString();
  const yesterday = new Date(Date.now() - 86400000).toDateString();
  let lastGroup = null;
  for (const conv of items) {
    const day = new Date(conv.created_at).toDateString();
    const group = day === today ? "今天" : day === yesterday ? "昨天" : "更早";
    if (group !== lastGroup) {
      nav.append(el("div", { class: "group", text: group }));
      lastGroup = group;
    }
    const dot = FINISHED.has(conv.status) ? (conv.status === "done" ? "ok" : "bad") : "run";
    nav.append(el("button", {
      type: "button", title: conv.title,
      class: `conv${state.current?.id === conv.id ? " active" : ""}`,
      onclick: () => openConversation(conv.id),
    }, el("span", { class: `dot ${dot}` }), el("span", { class: "title", text: conv.title })));
  }
}

// ---------- 對話 ----------

function emptyState() {
  return el("div", { class: "empty" },
    el("h1", { text: "研發助理" }),
    el("p", { text: "選一個資料夾,用中文問程式碼的問題。agent 會讀檔、在沙箱跑測試、提出修正等你審核,不會改動任何檔案。" }));
}

function resetThread() {
  for (const source of state.streams.values()) source.close();
  state.streams.clear();
  for (const view of state.views) stopTimer(view);
  state.views = [];
  $("#thread").replaceChildren();
}

function newChat() {
  resetThread();
  state.current = null;
  state.running = null;
  $("#thread").append(emptyState());
  $("#workspace").disabled = false;
  setSending(false);
  history.replaceState(null, "", location.pathname);
  loadHistory().catch((e) => hint(e.message, true));
  updateEstimate();
}

async function openConversation(cid, liveTurnId = null) {
  const conv = await api(`/api/conversations/${cid}`);
  resetThread();
  state.current = conv;
  state.running = null;
  setSending(false);
  await loadWorkspaces(conv.workspace);
  $("#workspace").disabled = true;
  for (const turn of conv.turns) addTurn(conv, turn, turn.id === liveTurnId);
  history.replaceState(null, "", `#${cid}`);
  await loadHistory();
  updateEstimate();
}

function addTurn(conv, turn, live) {
  const view = buildTurnView(turn);
  state.views.push(view);
  $("#thread").append(view.root);
  view.root.scrollIntoView({ block: "end" });
  if (!FINISHED.has(turn.status)) {
    state.running = { cid: conv.id, tid: turn.id };
    setSending(true);
  }
  streamTurn(conv, turn, view, live);
}

function buildTurnView(turn) {
  const steps = el("ol", { class: "steps" });
  const meta = el("span", { class: "meta", text: "排隊中…" });
  const work = el("details", { class: "work", open: "" }, el("summary", {}, "工作過程 · ", meta), steps);
  const answer = el("div", { class: "answer" });
  const extras = el("div", { class: "extras" });
  const footer = el("div", { class: "turn-footer" });
  const root = el("section", { class: "turn" },
    el("div", { class: "user-msg", text: turn.question }),
    el("div", { class: "assistant" }, work, answer, extras, footer));
  return { root, steps, meta, work, answer, extras, footer, turn, rows: [], pending: null,
           timer: null, startedAt: null, metaText: "", model: "" };
}

// ---------- 事件 ----------

function streamTurn(conv, turn, view, live) {
  const source = new EventSource(`/api/conversations/${conv.id}/turns/${turn.id}/events`);
  state.streams.set(turn.id, source);
  // 連線失敗時瀏覽器也會發一個沒有 data 的 "error" 事件,和我們自己的 error 事件同名,要先過濾
  const on = (type, handler) => source.addEventListener(type, (e) => {
    if (typeof e.data === "string") handler(JSON.parse(e.data));
  });
  on("start", (d) => {
    view.model = modelName(d.model);
    view.startedAt = Date.parse(d.t);
    setMeta(view, `${view.model} · 預估 ${fmt(d.est_tokens)} tokens`);
    startTimer(view);
  });
  on("step", (d) => {
    view.model = modelName(d.model);
    setPending(view, `第 ${d.step} 步:等 ${view.model} 回應…`);
  });
  on("tool_call", (d) => addToolRow(view, d));
  on("tool_result", (d) => fillToolRow(view, d));
  on("model_switch", (d) => {
    clearPending(view);
    view.steps.append(el("li", { class: "notice", text: `${d.reason},改用 ${d.to_role}` }));
  });
  on("patch", (d) => addPatchCard(conv, view, d));
  on("answer", (d) => showAnswer(view, d.text, live));
  on("error", (d) => showError(view, d));
  on("end", (d) => {
    source.close();
    state.streams.delete(turn.id);
    finishTurn(view, d);
  });
  source.onerror = () => {
    if (source.readyState === EventSource.CLOSED) {
      stopTimer(view);
      view.meta.textContent = "連線中斷,重新整理頁面可以接續";
    }
  };
}

function setMeta(view, text) {
  view.metaText = text;
  renderMeta(view);
}

function renderMeta(view) {
  const secs = view.startedAt ? Math.max(0, Math.round((Date.now() - view.startedAt) / 1000)) : null;
  view.meta.textContent = secs == null ? view.metaText : `${view.metaText} · ${secs} 秒`;
}

function startTimer(view) {
  stopTimer(view);
  view.timer = setInterval(() => renderMeta(view), 1000);
}

function stopTimer(view) {
  if (view.timer) clearInterval(view.timer);
  view.timer = null;
}

function setPending(view, text) {
  clearPending(view);
  view.pending = el("li", { class: "pending", text });
  view.steps.append(view.pending);
}

function clearPending(view) {
  view.pending?.remove();
  view.pending = null;
}

function describeArgs(name, a = {}) {
  if (name === "read_file") {
    const range = a.start_line || a.end_line ? ` ${a.start_line || 1}–${a.end_line || "結尾"}` : "";
    return `${a.path}${range}`;
  }
  if (name === "search_code") return `"${a.pattern}"${a.glob ? ` · ${a.glob}` : ""}`;
  if (name === "run_tests") return a.target || "tests";
  if (name === "propose_patch") return a.path;
  return JSON.stringify(a).slice(0, 80);
}

function testChip(text, prefix = "") {
  const failed = /(\d+) failed/.exec(text);
  const passed = /(\d+) passed/.exec(text);
  if (failed) return ["bad", `${prefix}${failed[1]} failed`];
  if (passed) return ["good", `${prefix}${passed[1]} passed`];
  return null;
}

function addToolRow(view, d) {
  clearPending(view);
  const chip = el("span", { class: "chip muted", text: "執行中" });
  const detail = el("pre", { class: "tool-output", hidden: "" });
  const head = el("button", { type: "button", class: "tool-head", onclick: () => { detail.hidden = !detail.hidden; } },
    el("span", { class: "tool-name", text: TOOL_LABEL[d.name] || d.name }),
    el("code", { text: describeArgs(d.name, d.args) }),
    chip);
  view.steps.append(el("li", { class: "tool" }, head, detail));
  view.rows.push({ step: d.step, name: d.name, chip, detail, done: false });
}

function fillToolRow(view, d) {
  const row = view.rows.find((r) => !r.done && r.step === d.step && r.name === d.name);
  if (!row) return;
  row.done = true;
  row.detail.textContent = d.preview;
  const isTest = d.name === "run_tests" || d.name === "propose_patch";
  const test = isTest ? testChip(d.preview, d.name === "propose_patch" ? "套用後 " : "") : null;
  const [cls, text] = !d.ok ? ["bad", "錯誤"] : test || ["muted", "完成"];
  row.chip.className = `chip ${cls}`;
  row.chip.textContent = text;
}

function addPatchCard(conv, view, d) {
  const diff = el("pre", { class: "diff" });
  for (const line of d.diff.split("\n")) {
    const cls = line.startsWith("+") && !line.startsWith("+++") ? "add"
      : line.startsWith("-") && !line.startsWith("---") ? "del" : null;
    diff.append(el("span", { class: cls }, `${line}\n`));
  }
  const [cls, text] = testChip(d.test_output, "套用後 ")
    || (d.passed ? ["good", "套用後測試通過"] : ["bad", "套用後測試失敗"]);
  const actions = el("div", { class: "patch-actions" }, el("span", { class: `chip ${cls}`, text }));
  const ws = workspaceById(conv.workspace);
  if (ws && ws.kind === "config" && d.patch_file) {
    const cmd = `git -C "${ws.path}" apply "${d.patch_file}"`;
    actions.append(el("button", { type: "button", text: "複製 git apply 指令",
      onclick: (e) => copyText(cmd, e.currentTarget) }));
  } else {
    actions.append(el("span", { class: "note", text: "這是上傳的複本,patch 不能直接套回你的專案" }));
  }
  view.extras.append(el("div", { class: "patch" }, el("code", { text: d.path }), diff, actions));
}

async function copyText(text, button) {
  try {
    await navigator.clipboard.writeText(text);
    button.textContent = "已複製";
  } catch {
    window.prompt("複製這行指令:", text);
  }
}

function showAnswer(view, text, live) {
  clearPending(view);
  const long = view.turn.mode === "whole" || text.length > 3000;
  if (!long) {
    view.answer.innerHTML = renderMarkdown(text);
    return;
  }
  const title = ((text.split("\n").find((l) => l.trim()) || "文件").replace(/^#+\s*/, "")).slice(0, 40);
  view.answer.replaceChildren(el("div", { class: "doc-card" },
    el("div", {}, el("strong", { text: title }), el("div", { class: "note", text: `共 ${fmt(text.length)} 字` })),
    el("button", { type: "button", text: "在側欄開啟", onclick: () => openPanel(title, text) })));
  if (live) openPanel(title, text);
}

function showError(view, d) {
  clearPending(view);
  view.answer.append(el("div", { class: "error-box" },
    el("span", { text: d.message }),
    el("button", { type: "button", text: "重試",
      onclick: () => sendTurn({ question: view.turn.question, mode: view.turn.mode, ...(view.turn.options || {}) })
        .catch((e) => hint(e.message, true)) })));
}

function finishTurn(view, d) {
  clearPending(view);
  stopTimer(view);
  view.work.open = false;
  view.meta.textContent = `${d.steps} 步 · ${d.seconds} 秒 · ${view.model || "未呼叫模型"}`;
  view.footer.textContent = [STATUS_TEXT[d.status], `${fmt(d.tokens)} tokens`].filter(Boolean).join(" · ");
  if (state.running?.tid === view.turn.id) {
    state.running = null;
    setSending(false);
  }
  loadHistory().catch(() => {});
}

// ---------- 送出、停止、上傳、估算 ----------

async function sendTurn(body) {
  if (state.current) {
    const r = await api(`/api/conversations/${state.current.id}/turns`, { method: "POST", body: JSON.stringify(body) });
    addTurn(state.current, { id: r.turn_id, question: body.question, mode: body.mode, options: body, status: "queued" }, true);
  } else {
    const r = await api("/api/conversations", {
      method: "POST", body: JSON.stringify({ ...body, workspace: $("#workspace").value }) });
    await openConversation(r.conversation_id, r.turn_id);
  }
}

async function onSubmit(event) {
  event.preventDefault();
  if (state.running) {
    const { cid, tid } = state.running;
    try {
      await api(`/api/conversations/${cid}/turns/${tid}/cancel`, { method: "POST" });
      hint("已要求停止:正在等模型回應的那一步回來後就會停。");
    } catch (e) { hint(e.message, true); }
    return;
  }
  const question = $("#question").value.trim();
  if (!question) { hint("先輸入問題", true); return; }
  if (!state.current && !$("#workspace").value) { hint("先選一個資料夾", true); return; }
  const body = { question, mode: currentMode(), role: $("#role").value || null,
                 thinking: thinkingValue(), files: $("#files").value.trim() || null };
  try {
    await sendTurn(body);
    $("#question").value = "";
    hint("");
  } catch (e) { hint(e.message, true); }
}

async function onUpload(event) {
  const files = [...event.target.files];
  event.target.value = "";
  if (!files.length) return;
  const form = new FormData();
  for (const f of files) form.append("files", f, f.name);
  hint("上傳中…");
  try {
    const r = await api("/api/uploads", { method: "POST", body: form });
    if (state.current) newChat();
    await loadWorkspaces(r.workspace.id);
    const skipped = r.skipped.length ? `;略過 ${r.skipped.length} 個機密或系統檔:${r.skipped.slice(0, 5).join("、")}` : "";
    hint(`已上傳「${r.workspace.name}」${skipped}。這是複本,24 小時後刪除,patch 不能直接套回你的專案。`);
    updateEstimate();
  } catch (e) { hint(e.message, true); }
}

async function updateEstimate() {
  const send = $("#send");
  if (!state.running) send.disabled = false;
  if (currentMode() !== "whole") return;
  const ws = state.current ? state.current.workspace : $("#workspace").value;
  if (!ws) return;
  hint("估算整份文件大小中…");
  try {
    const e = await api(`/api/estimate?workspace=${encodeURIComponent(ws)}`);
    if (e.ok) {
      const wait = e.tokens > 100000 ? "可能要好幾分鐘,可以先去做別的事" : "大約一兩分鐘";
      hint(`整份文件:${fmt(e.files)} 個檔案,約 ${fmt(e.tokens)} tokens → 會用 ${modelName(e.model)},${wait}。`);
    } else {
      hint(`約 ${fmt(e.tokens)} tokens,超過上限 ${fmt(e.limit)},請改用較小的資料夾。`, true);
      if (!state.running) send.disabled = true;
    }
  } catch (err) { hint(err.message, true); }
}

// ---------- 側欄 ----------

function openPanel(title, markdownText) {
  $("#panel-title").textContent = title;
  $("#panel-body").innerHTML = renderMarkdown(markdownText);
  $("#panel").hidden = false;
  document.body.classList.add("panel-open");
}

function closePanel() {
  $("#panel").hidden = true;
  document.body.classList.remove("panel-open");
}

// ---------- 啟動 ----------

async function init() {
  $("#new-chat").addEventListener("click", newChat);
  $("#composer").addEventListener("submit", onSubmit);
  $("#question").addEventListener("keydown", (e) => {
    // isComposing:注音 / 倉頡選字時按 Enter 是確認選字,不是送出
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      $("#composer").requestSubmit();
    }
  });
  $("#upload").addEventListener("change", onUpload);
  for (const radio of document.querySelectorAll('input[name="mode"]')) {
    radio.addEventListener("change", () => { hint(""); updateEstimate(); });
  }
  $("#workspace").addEventListener("change", updateEstimate);
  $("#panel-close").addEventListener("click", closePanel);
  try {
    await loadWorkspaces();
    const cid = location.hash.slice(1);
    if (cid) await openConversation(cid);
    else newChat();
  } catch (e) {
    hint(e.message, true);
  }
}

init();
