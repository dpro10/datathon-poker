"use strict";

const $ = (s) => document.querySelector(s);
const fmt = new Intl.NumberFormat("en-US");

function el(tag, props = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (v == null || v === false) continue;
    if (k === "class") n.className = v;
    else if (k === "text") n.textContent = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) {
    if (kid == null || kid === false) continue;
    n.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return n;
}

const tone = (n) => (n > 0 ? "pos" : n < 0 ? "neg" : "");
const bbText = (v) => (v == null ? "-" : `${v > 0 ? "+" : v < 0 ? "-" : ""}${Math.abs(v).toFixed(2)}`);
const scoreText = (v) => (v == null ? "-" : v.toFixed(2));
const record = (b) => `${b.wins}-${b.chops}-${b.losses}`;
function ago(ts) {
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 60) return `${Math.round(s)}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  return `${Math.round(s / 3600)}h ago`;
}
function health(b) {
  const parts = [];
  if (b.forfeits) parts.push(`${b.forfeits} forfeit${b.forfeits === 1 ? "" : "s"}`);
  if (b.errors) parts.push(`${b.errors} error${b.errors === 1 ? "" : "s"}`);
  if (b.bank_outs) parts.push(`out of time ${b.bank_outs}x`);
  return parts;
}

const store = {
  get(k, d) { try { return JSON.parse(localStorage.getItem(k)) ?? d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode */ } },
};
const keys = store.get("ladderKeys", {});

let state = null;
let selected = null;
let showRetired = false;
let retireArmed = false;

async function refresh() {
  try {
    const r = await fetch("/api/state", { cache: "no-store" });
    if (!r.ok) throw new Error(String(r.status));
    state = await r.json();
    $("#offline").hidden = true;
    render();
    if (selected != null) loadDetail(selected, false);
  } catch {
    $("#offline").hidden = false;
  }
}

function render() {
  const active = state.bots.filter((b) => b.active).length;
  $("#stats").replaceChildren(
    el("span", { class: "chip", text: `${active} bots` }),
    el("span", { class: "chip", text: `${fmt.format(state.total_matches)} matches` }),
    el("span", { class: "chip", text: `${state.running.length} playing now` }),
  );
  renderBoard();
  renderFeed();
  const s = state.settings;
  $("#limits").replaceChildren(
    el("li", { text: `${fmt.format(s.deals)} deals per half, played twice with seats swapped (${fmt.format(2 * s.deals)} hands). Each half runs in brand-new bot processes.` }),
    el("li", { text: `${s.stack} chip stacks, blinds 1/2. ${s.time_bank}s of thinking per half, ${s.timeout}s max per decision.` }),
    el("li", { text: `${s.error_limit} errors in a half forfeit it, and a forfeit loses the match.` }),
    el("li", { text: "Bots get Python, numpy and the official game.py and evaluator.py. No internet, no files outside the bot's folder." }),
  );
}

function renderBoard() {
  const body = $("#board tbody");
  body.replaceChildren();
  let rank = 0;
  for (const b of state.bots) {
    if (!b.active && !showRetired) continue;
    if (b.active) rank += 1;
    const open = () => select(b.id);
    const problems = health(b);
    body.append(el("tr", {
      class: [b.active ? "" : "retired", selected === b.id ? "sel" : ""].join(" ").trim() || null,
      tabindex: "0", onclick: open, onkeydown: (e) => { if (e.key === "Enter") open(); },
    },
      el("td", { class: "num sub", text: b.active ? rank : "-" }),
      el("td", {},
        el("span", { class: "name", text: b.name }),
        b.house ? el("span", { class: "tag", text: "baseline" }) : null,
        keys[b.id] ? el("span", { class: "tag mine", text: "yours" }) : null,
        el("div", { class: "sub", text: b.house ? "official baseline" : `by ${b.author}` })),
      el("td", { class: "num strong", text: b.matches ? fmt.format(b.rating) : "new" }),
      el("td", { class: "num", text: record(b) }),
      el("td", { class: "num", text: scoreText(b.score) }),
      el("td", { class: `num ${tone(b.bb_deal || 0)}`, text: bbText(b.bb_deal) }),
      el("td", { class: "num", text: fmt.format(b.matches) }),
      el("td", {}, problems.length
        ? el("span", { class: "warn", title: b.last_problem || "", text: problems.join(", ") })
        : el("span", { class: "ok", text: "ok" })),
    ));
  }
  if (!body.children.length) body.append(el("tr", {}, el("td", { colspan: "8", class: "sub", text: "No bots yet." })));
}

function renderFeed() {
  const list = $("#matches");
  list.replaceChildren();
  for (const r of state.running) {
    list.append(el("li", { class: "live" }, el("span", { class: "dot" }),
      el("span", {}, el("b", { text: r.a }), " vs ", el("b", { text: r.b })), el("span", { class: "sub", text: "playing now" })));
  }
  for (const m of state.matches) list.append(matchItem(m, null));
  if (!list.children.length) list.append(el("li", { class: "sub", text: "Matches start as soon as two bots are entered." }));
}

function matchMeta(m) {
  const ci = m.ci && m.ci[0] != null ? ` · 95% [${bbText(m.ci[0])}, ${bbText(m.ci[1])}]` : "";
  return ` · ${ago(m.created_at)}${ci}`;
}

function matchItem(m, me) {
  const forfeit = m.verdict === "BOTH_FORFEIT" ? "both forfeited" : null;
  if (me != null) {
    const mine = m.a === me ? m.points_a : m.points_b;
    const bb = m.a === me ? m.mean_bb : -m.mean_bb;
    const opp = m.a === me ? m.b_name : m.a_name;
    const word = forfeit ? "Both forfeited vs " : mine === 1 ? "Beat " : mine === 0.5 ? "Chopped with " : "Lost to ";
    return el("li", {},
      el("span", { class: mine === 1 ? "pos" : mine === 0 ? "neg" : "", text: word }),
      el("b", { text: opp }), " ", el("span", { class: tone(bb), text: `${bbText(bb)} bb/deal` }),
      el("span", { class: "sub", text: matchMeta(m) }));
  }
  if (forfeit) return el("li", {}, el("b", { text: m.a_name }), " and ", el("b", { text: m.b_name }), " both forfeited", el("span", { class: "sub", text: matchMeta(m) }));
  if (m.points_a === 0.5) {
    return el("li", {}, el("b", { text: m.a_name }), " chopped with ", el("b", { text: m.b_name }),
      ` (${bbText(m.mean_bb)} bb/deal, too close to call)`, el("span", { class: "sub", text: matchMeta(m) }));
  }
  const aWon = m.points_a === 1;
  const [w, l] = aWon ? [m.a_name, m.b_name] : [m.b_name, m.a_name];
  return el("li", {}, el("b", { text: w }), " beat ", el("b", { text: l }),
    ` by ${Math.abs(m.mean_bb).toFixed(2)} bb/deal`, el("span", { class: "sub", text: matchMeta(m) }));
}

async function select(id) {
  selected = id;
  retireArmed = false;
  if (state) renderBoard();
  await loadDetail(id, true);
}

async function loadDetail(id, scroll) {
  try {
    const r = await fetch(`/api/bots/${id}`, { cache: "no-store" });
    if (!r.ok) return;
    const d = await r.json();
    if (selected !== id) return;
    renderDetail(d);
    if (scroll) $("#detail").scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch { /* next refresh retries */ }
}

const kpi = (label, value, cls) => el("div", { class: "kpi" }, el("b", { class: cls || null, text: value }), el("span", { text: label }));

function renderDetail(d) {
  const b = d.bot;
  const box = $("#detail");
  const logBox = box.querySelector("pre.log");
  const problems = health(b);
  box.hidden = false;
  box.replaceChildren(...[
    el("div", { class: "detail-head" },
      el("div", {}, el("h2", { text: b.name }),
        el("div", { class: "sub", text: b.house ? "Official baseline bot" : `by ${b.author} · entered ${ago(b.created_at)}` })),
      el("button", { class: "ghost", onclick: () => { selected = null; box.hidden = true; renderBoard(); }, text: "Close" })),
    el("div", { class: "kpis" },
      kpi("Rating", b.matches ? fmt.format(b.rating) : "new"),
      kpi("Won-chopped-lost", record(b)),
      kpi("Score", scoreText(b.score)),
      kpi("bb/deal", bbText(b.bb_deal), tone(b.bb_deal || 0))),
    b.note ? el("p", { class: "note", text: b.note }) : null,
    problems.length ? el("p", { class: "note", text: `${problems.join(", ")}. Latest: ${b.last_problem || "see the bot's output"}` }) : null,
    b.matches ? el("p", { class: "sub", text: `Slowest half used ${b.think_max.toFixed(2)}s of the ${state ? state.settings.time_bank : 8}s time bank.` }) : null,
    d.h2h.length ? el("div", {}, el("h3", { text: "Head to head" }), el("div", { class: "table-wrap" }, el("table", {},
      el("thead", {}, el("tr", {}, el("th", { text: "Opponent" }), el("th", { class: "num", text: "Matches" }),
        el("th", { class: "num", text: "W-C-L" }), el("th", { class: "num", text: "Score" }), el("th", { class: "num", text: "bb/deal" }))),
      el("tbody", {}, d.h2h.map((h) => el("tr", {},
        el("td", {}, el("span", { class: "name", text: h.opp_name }), el("span", { class: "sub", text: ` ${h.opp_rating}` })),
        el("td", { class: "num", text: h.n }),
        el("td", { class: "num", text: `${h.wins}-${h.chops}-${h.losses}` }),
        el("td", { class: "num", text: scoreText(h.points / h.n) }),
        el("td", { class: `num ${tone(h.bb_deal || 0)}`, text: bbText(h.bb_deal) }))))))) : el("p", { class: "sub", text: "No matches yet. The first one starts within a minute." }),
    d.recent.length ? el("div", {}, el("h3", { text: "Recent matches" }), el("ul", { class: "feed" }, d.recent.slice(0, 10).map((m) => matchItem(m, b.id)))) : null,
    keys[b.id] ? ownerTools(b) : null,
  ].filter(Boolean));
  if (logBox) box.append(logBox);
}

function ownerTools(b) {
  const key = keys[b.id];
  const showLog = async () => {
    const r = await fetch(`/api/bots/${b.id}/log`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ key }) });
    const res = await r.json();
    $("#detail").querySelector("pre.log")?.remove();
    $("#detail").append(el("pre", { class: "log", text: res.log || res.error }));
  };
  const retire = async (e) => {
    if (!retireArmed) { retireArmed = true; e.target.textContent = "Click again to retire"; return; }
    const r = await fetch(`/api/bots/${b.id}/retire`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ key }) });
    retireArmed = false;
    if (r.ok) refresh();
  };
  return el("div", { class: "actions" },
    el("button", { class: "ghost", onclick: showLog, text: "Show errors and output from the latest match" }),
    b.active ? el("button", { class: "danger", onclick: retire, text: retireArmed ? "Click again to retire" : "Retire this bot" }) : null);
}

function showResult(kind, message, details) {
  const out = $("#enter-result");
  out.hidden = false;
  out.className = `result ${kind}`;
  out.replaceChildren(...[el("div", { text: message }), details ? el("pre", { text: details }) : null].filter(Boolean));
}

$("#enter").addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = e.target;
  const btn = f.querySelector("button[type=submit]");
  const file = f.elements.file.files[0];
  if (!file) return showResult("error", "Pick your agent.py.");
  if (file.size > 300000) return showResult("error", "That file is over 300 KB.");
  const name = f.elements.botname.value.trim() || file.name.replace(/\.py$/i, "");
  const author = f.elements.author.value.trim();
  const invite = f.elements.invite.value.trim();
  store.set("ladderAuthor", author);
  store.set("ladderInvite", invite);
  btn.disabled = true;
  showResult("busy", `Testing ${name} with the official judge against calling_station. This takes a few seconds.`);
  try {
    const code = await file.text();
    const r = await fetch("/api/bots", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name, author, invite, code }) });
    const res = await r.json();
    if (!r.ok) return showResult("error", res.error || "Upload failed.", res.details);
    keys[res.id] = res.key;
    store.set("ladderKeys", keys);
    showResult("ok", `${name} is on the ladder. Its first match starts within a minute.`,
      `Owner key (saved in this browser; copy it to manage the bot from another device):\n${res.key}`);
    f.elements.file.value = "";
    f.elements.botname.value = "";
    await refresh();
    select(res.id);
  } catch {
    showResult("error", "Couldn't reach the ladder. Try again.");
  } finally {
    btn.disabled = false;
  }
});

$("#show-retired").addEventListener("change", (e) => { showRetired = e.target.checked; if (state) renderBoard(); });
$("#enter").elements.author.value = store.get("ladderAuthor", "");
$("#enter").elements.invite.value = store.get("ladderInvite", "");

refresh();
setInterval(() => { if (!document.hidden) refresh(); }, 4000);
document.addEventListener("visibilitychange", () => { if (!document.hidden) refresh(); });
