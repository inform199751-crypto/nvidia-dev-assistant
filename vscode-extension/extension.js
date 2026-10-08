// 小助理 VS Code 外掛:在編輯器旁邊打開小助理的網頁介面(webui.py),也能把選取的程式碼直接拿去問。
// 外掛自己在背景啟動 webui.py(token 由外掛產生,才能直接送問題),分析的資料夾是 VS Code 目前開著的那個。
// 純 JavaScript、不用編譯;打包用 build_vsix.py。
"use strict";
const vscode = require("vscode");
const cp = require("child_process");
const crypto = require("crypto");
const fs = require("fs");
const path = require("path");

let server = null; // { proc, port, token, workspace }
let panel = null;
let panelSrc = "";
let output = null;

const config = () => vscode.workspace.getConfiguration("devAssistant");
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function projectPath() {
  let p = config().get("projectPath");
  if (p && fs.existsSync(path.join(p, "webui.py"))) return p;
  const pick = await vscode.window.showOpenDialog({
    canSelectFolders: true, canSelectFiles: false, openLabel: "選這個資料夾",
    title: "選擇小助理的資料夾(裡面有 webui.py 和 .venv)",
  });
  if (!pick) return null;
  p = pick[0].fsPath;
  if (!fs.existsSync(path.join(p, "webui.py"))) {
    vscode.window.showErrorMessage(`${p} 裡沒有 webui.py,請選小助理的資料夾`);
    return null;
  }
  await config().update("projectPath", p, vscode.ConfigurationTarget.Global);
  return p;
}

function pythonPath(project) {
  for (const p of [path.join(project, ".venv", "Scripts", "python.exe"), path.join(project, ".venv", "bin", "python")]) {
    if (fs.existsSync(p)) return p;
  }
  return null;
}

function workspaceFolder() {
  const editor = vscode.window.activeTextEditor;
  const folder = (editor && vscode.workspace.getWorkspaceFolder(editor.document.uri))
    || (vscode.workspace.workspaceFolders || [])[0];
  return folder ? folder.uri.fsPath : null;
}

function stopServer() {
  if (!server) return;
  const { proc } = server;
  server = null;
  if (proc.exitCode !== null) return;
  // Windows 的 .venv\Scripts\python.exe 只是轉介,真正的 Python 是它的子行程:要連子行程一起關
  if (process.platform === "win32") cp.spawnSync("taskkill", ["/pid", String(proc.pid), "/T", "/F"]);
  else proc.kill();
}

async function waitUntilReady(s, ms) {
  // 用 token 打 /api/state:回 200 才確定是我們自己啟動的那一個(port 被別的程式占用時不會誤認)
  const until = Date.now() + ms;
  while (Date.now() < until) {
    if (s.proc.exitCode !== null) return false;
    try {
      const r = await fetch(`http://127.0.0.1:${s.port}/api/state`, { headers: { "X-Token": s.token } });
      if (r.ok) return true;
    } catch (e) { /* 還沒起來 */ }
    await sleep(500);
  }
  return false;
}

async function ensureServer() {
  // 已經在跑就沿用(切換到別的資料夾的檔案不自動重啟,不然對話會不見;要換工作區用「重新啟動」)
  if (server && server.proc.exitCode === null) return server;
  stopServer();
  const workspace = config().get("useWorkspaceFolder") ? workspaceFolder() : null;
  const project = await projectPath();
  if (!project) return null;
  const python = pythonPath(project);
  if (!python) {
    vscode.window.showErrorMessage("找不到小助理的 .venv,請先在小助理資料夾建立虛擬環境並安裝 requirements.txt");
    return null;
  }
  const port = config().get("port");
  const token = crypto.randomBytes(24).toString("base64url");
  const env = { ...process.env, PYTHONUTF8: "1", PYTHONWARNINGS: "ignore" };
  if (workspace) env.AGENT_WORKSPACE = workspace; // 環境變數優先於小助理 .env 裡的設定
  output.appendLine(`啟動小助理:port ${port},工作區 ${workspace || "(照 .env)"}`);
  const proc = cp.spawn(python, ["webui.py", "--no-browser", "--port", String(port), "--token", token],
                        { cwd: project, env });
  proc.stdout.on("data", (d) => output.append(d.toString()));
  proc.stderr.on("data", (d) => output.append(d.toString()));
  proc.on("exit", (code) => {
    output.appendLine(`小助理已結束(exit ${code})`);
    if (server && server.proc === proc) server = null;
  });
  const s = { proc, port, token, workspace };
  server = s;
  const ready = await vscode.window.withProgress(
    { location: vscode.ProgressLocation.Notification, title: "小助理啟動中…" },
    () => waitUntilReady(s, 40000));
  if (!ready) {
    vscode.window.showErrorMessage(`小助理沒有啟動成功(port ${port} 可能被占用),詳情請看「輸出」面板的「小助理」`);
    output.show(true);
    stopServer();
    return null;
  }
  return s;
}

async function openPanel() {
  const s = await ensureServer();
  if (!s) return null;
  const uri = await vscode.env.asExternalUri(vscode.Uri.parse(`http://127.0.0.1:${s.port}/`));
  const src = uri.toString(true);
  if (!panel) {
    panel = vscode.window.createWebviewPanel("devAssistant", "小助理", { viewColumn: vscode.ViewColumn.Beside, preserveFocus: true },
                                             { enableScripts: true, retainContextWhenHidden: true });
    panel.onDidDispose(() => { panel = null; panelSrc = ""; });
  }
  if (panelSrc !== src) { // 換了伺服器才重設,不然每次都會重新載入、對話畫面閃一下
    const origin = `${uri.scheme}://${uri.authority}`;
    panel.webview.html = `<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; frame-src ${origin}; style-src 'unsafe-inline';">
<style>html,body,iframe{margin:0;padding:0;width:100%;height:100%;border:0;overflow:hidden}</style></head>
<body><iframe src="${src}" allow="clipboard-read; clipboard-write"></iframe></body></html>`;
    panelSrc = src;
  }
  panel.reveal(vscode.ViewColumn.Beside, true);
  return s;
}

async function askSelection() {
  const editor = vscode.window.activeTextEditor;
  if (!editor || editor.selection.isEmpty) {
    vscode.window.showInformationMessage("先在編輯器裡選取一段程式碼");
    return;
  }
  const question = await vscode.window.showInputBox({ prompt: "要問小助理什麼?", value: "這段程式在做什麼?有沒有問題?" });
  if (question === undefined) return;
  const s = await openPanel();
  if (!s) return;
  const doc = editor.document;
  const sel = editor.selection;
  const rel = s.workspace ? path.relative(s.workspace, doc.uri.fsPath).split(path.sep).join("/") : "";
  const name = rel && !rel.startsWith("..") ? rel : path.basename(doc.uri.fsPath);
  // 只附選取的部分和位置,不整檔附上(大檔很耗 token);需要的話小助理會自己 read_file
  const text = `${question}\n\n(${name} 第 ${sel.start.line + 1}-${sel.end.line + 1} 行)\n` +
               "```" + doc.languageId + "\n" + doc.getText(sel) + "\n```";
  try {
    const r = await fetch(`http://127.0.0.1:${s.port}/api/ask`, {
      method: "POST", headers: { "Content-Type": "application/json", "X-Token": s.token }, body: JSON.stringify({ text }),
    });
    if (r.status === 409) vscode.window.showWarningMessage("小助理還在回答上一個問題,等它答完再問");
    else if (!r.ok) vscode.window.showErrorMessage(`送出失敗:HTTP ${r.status}`);
  } catch (e) {
    vscode.window.showErrorMessage(`送出失敗:${e.message}`);
  }
}

function activate(context) {
  output = vscode.window.createOutputChannel("小助理");
  context.subscriptions.push(
    output,
    vscode.commands.registerCommand("devAssistant.open", openPanel),
    vscode.commands.registerCommand("devAssistant.askSelection", askSelection),
    vscode.commands.registerCommand("devAssistant.restart", async () => { stopServer(); await openPanel(); }),
    vscode.commands.registerCommand("devAssistant.stop", () => { stopServer(); if (panel) panel.dispose(); }),
    { dispose: stopServer },
  );
}

function deactivate() {
  stopServer();
}

module.exports = { activate, deactivate };
