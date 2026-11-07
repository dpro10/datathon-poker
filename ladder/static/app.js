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

const signed = (n) => (n > 0 ? "+" : n < 0 ? "-" : "") + fmt.format(Math.abs(n));
const tone = (n) => (n > 0 ? "pos" : n < 0 ? "neg" : "");
const bb = (v) => (v == null ? "-" : (v > 0 ? "+" : "") + v.toFixed(1));
function ago(ts) {
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 60) return `${Math.round(s)}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  return `${Math.round(s / 3600)}h ago`;
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
  $("#rules").textContent = s.rules_text;
  $("#limits").replaceChildren(
    el("li", { text: `${fmt.format(s.hands)} hands per match, ${s.timeout}s per decision, ${s.time_bank}s of thinking per half.` }),
    el("li", { text: "Python standard library, numpy and pokerlib only. No internet, no files outside the bot's folder." }),
    el("li", { text: "Crashes, timeouts and illegal moves play as check or fold. Three crashed matches in a row retire the bot." }),
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
    body.append(el("tr", {
      class: [b.active ? "" : "retired", selected === b.id ? "sel" : ""].join(" ").trim() || null,
      tabindex: "0", onclick: open, onkeydown: (e) => { if (e.key === "Enter") open(); },
    },
      el("td", { class: "num sub", text: b.active ? rank : "-" }),
      el("td", {},
        el("span", { class: "name", text: b.name }),
        b.house ? el("span", { class: "tag", text: "practice" }) : null,
        keys[b.id] ? el("span", { class: "tag mine", text: "yours" }) : null,
        el("div", { class: "sub", text: b.house ? "house bot" : `by ${b.author}` })),
      el("td", { class: "num strong", text: b.matches ? fmt.format(b.rating) : "new" }),
      el("td", { class: "num", text: `${b.wins}-${b.losses}${b.draws ? `-${b.draws}` : ""}` }),
      el("td", { class: `num ${tone(b.bb100 || 0)}`, text: bb(b.bb100) }),
      el("td", { class: "num", text: fmt.format(b.matches) }),
      el("td", {}, b.problems
        ? el("span", { class: "warn", title: b.last_problem || "", text: `${b.problems} issue${b.problems === 1 ? "" : "s"}` })
        : el("span", { class: "ok", text: "ok" })),
    ));
  }
  if (!body.children.length) body.append(el("tr", {}, el("td", { colspan: "7", class: "sub", text: "No bots yet." })));
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

function matchItem(m, me) {
  const meta = ` · ${fmt.format(m.hands)} hands${m.t != null ? ` · t ${Math.abs(m.t).toFixed(1)}` : ""} · ${ago(m.created_at)}`;
  if (me != null) {
    const mine = m.a === me ? m.net_a : -m.net_a;
    const opp = m.a === me ? m.b_name : m.a_name;
    return el("li", {},
      el("span", { class: tone(mine), text: mine > 0 ? "Beat " : mine < 0 ? "Lost to " : "Tied " }),
      el("b", { text: opp }), " ", el("span", { class: tone(mine), text: `${signed(mine)} chips` }),
      el("span", { class: "sub", text: meta }));
  }
  if (m.net_a === 0) return el("li", {}, el("b", { text: m.a_name }), " tied ", el("b", { text: m.b_name }), el("span", { class: "sub", text: meta }));
  const [w, l] = m.net_a > 0 ? [m.a_name, m.b_name] : [m.b_name, m.a_name];
  return el("li", {}, el("b", { text: w }), " beat ", el("b", { text: l }),
    ` by ${fmt.format(Math.abs(m.net_a))} chips`, el("span", { class: "sub", text: meta }));
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
  box.hidden = false;
  box.replaceChildren(...[
    el("div", { class: "detail-head" },
      el("div", {}, el("h2", { text: b.name }),
        el("div", { class: "sub", text: b.house ? "House practice bot" : `by ${b.author} · entered ${ago(b.created_at)}` })),
      el("button", { class: "ghost", onclick: () => { selected = null; box.hidden = true; renderBoard(); }, text: "Close" })),
    el("div", { class: "kpis" },
      kpi("Rating", b.matches ? fmt.format(b.rating) : "new"),
      kpi("Won-lost", `${b.wins}-${b.losses}${b.draws ? `-${b.draws}` : ""}`),
      kpi("bb/100", bb(b.bb100), tone(b.bb100 || 0)),
      kpi("Hands", fmt.format(b.hands))),
    b.note ? el("p", { class: "note", text: b.note }) : null,
    b.problems ? el("p", { class: "note", text: `${b.problems} problem${b.problems === 1 ? "" : "s"} so far. Latest: ${b.last_problem || "unknown"}` }) : null,
    d.h2h.length ? el("div", {}, el("h3", { text: "Head to head" }), el("div", { class: "table-wrap" }, el("table", {},
      el("thead", {}, el("tr", {}, el("th", { text: "Opponent" }), el("th", { class: "num", text: "Matches" }),
        el("th", { class: "num", text: "W-L" }), el("th", { class: "num", text: "Chips" }), el("th", { class: "num", text: "bb/100" }))),
      el("tbody", {}, d.h2h.map((h) => el("tr", {},
        el("td", {}, el("span", { class: "name", text: h.opp_name }), el("span", { class: "sub", text: ` ${h.opp_rating}` })),
        el("td", { class: "num", text: h.n }),
        el("td", { class: "num", text: `${h.wins}-${h.losses}` }),
        el("td", { class: `num ${tone(h.chips)}`, text: signed(h.chips) }),
        el("td", { class: `num ${tone(h.bb100 || 0)}`, text: bb(h.bb100) }))))))) : el("p", { class: "sub", text: "No matches yet. The first one starts within a minute." }),
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
    el("button", { class: "ghost", onclick: showLog, text: "Show my bot's latest output" }),
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
  if (!file) return showResult("error", "Pick your bot's .py file.");
  if (file.size > 200000) return showResult("error", "That file is over 200 KB.");
  const name = f.elements.botname.value.trim() || file.name.replace(/\.py$/i, "");
  const author = f.elements.author.value.trim();
  const invite = f.elements.invite.value.trim();
  store.set("ladderAuthor", author);
  store.set("ladderInvite", invite);
  btn.disabled = true;
  showResult("busy", `Testing ${name} against calling_station. This takes a few seconds.`);
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
