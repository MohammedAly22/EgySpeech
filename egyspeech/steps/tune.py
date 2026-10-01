"""tune — interactive filter tuner: <work_dir>/review/tuner.html.

A self-contained page (open it in Chrome / Edge) with every quality metric of the
clips: switch each metric on / off, move its threshold and see instantly how many
clips are kept, the histogram, and the accepted / rejected clips with audio players
(borderline clips first). Mark clips ✓ / ✗ while listening: the page counts how many
bad clips would be accepted and good clips lost. When happy, copy the `filter:` block
it shows into configs/config.yaml and run the `filter` step.

    python -m egyspeech.cli tune [--max-clips 30000]

Labels are kept in the browser; review/labels.json (if present) pre-fills them.
"""

import json
import logging
import os
import random

from egyspeech.config import Section
from egyspeech.steps import layout, step_main, videos
from egyspeech.steps.filter import RULES, edge_db, load_joined

logger = logging.getLogger("tune")

# Ordered by how directly a metric targets what must be removed (noise / music / effects,
# then a second voice), then general quality scores.
METRICS = [
    ("min_dnsmos_bak", "Background (DNSMOS BAK)", 0.01, 3.0,
     "How clean the background is, 1-5 (Microsoft DNSMOS, P.835). Anything that is not the voice lowers it: "
     "background noise, music beds, sound effects, room hum, traffic. <b>The main detector for music, effects and "
     "noise.</b> Clean studio speech is usually above 3.8; 3.3-3.8 has a light noise floor."),
    ("min_window_similarity", "Single speaker (voice consistency)", 0.005, 0.70,
     "The clip is cut into 3 s windows and each window's voice (TitaNet speaker embedding) is compared with the "
     "clip's average voice; this is the <b>lowest</b> similarity of any window (cosine, up to 1). A second speaker, "
     "an interjection or someone else laughing pulls it down. <b>The main detector for multi-speaker clips.</b> "
     "Same voice is usually above 0.8."),
    ("min_dnsmos_sig", "Speech signal (DNSMOS SIG)", 0.01, 3.0,
     "Quality of the voice itself, 1-5, ignoring the background: distortion, muffled or phone-like sound, codec "
     "artifacts, heavy reverb. Low = the voice sounds damaged."),
    ("min_dnsmos_ovrl", "Overall (DNSMOS OVRL)", 0.01, 2.7,
     "Overall listening quality, 1-5, predicted from SIG and BAK together. Mostly redundant when BAK and SIG are "
     "used; it also marks lively, unpolished voices lower."),
    ("min_dnsmos_p808", "Overall (DNSMOS P.808)", 0.01, 3.3,
     "A second overall-quality model (ITU-T P.808 ratings), 1-5. Similar to OVRL, trained on different ratings."),
    ("min_utmos", "Naturalness (UTMOS)", 0.01, 2.0,
     "Predicted naturalness MOS, 1-5, from a model trained on English speech: Arabic scores lower in general. "
     "Very low values flag broken or robotic audio; in between it also penalizes normal speech."),
    ("min_speech_ratio", "Speech ratio", 0.01, 0.55,
     "Share of the clip that is speech (voice activity). Low = long silences inside the clip."),
    ("max_clip_ratio", "Clipping", 0.0001, 0.001,
     "Fraction of samples at full scale (the recording was too loud and got flattened). Reject above the threshold."),
    ("max_edge_db", "Edge loudness", 0.5, -5.0,
     "Loudest 10 ms at the start / end of the clip relative to its speech level, in dB. Near 0 = speech right at the "
     "cut. Reject above the threshold."),
]


def main(cfg: Section, args):
    lay = layout(cfg)
    rows = load_joined(cfg)
    if not rows:
        raise SystemExit("nothing to tune: run segment, quality and speaker_check first")
    if len(rows) > args.max_clips:
        rows = random.Random(0).sample(rows, args.max_clips)
    root = lay.review
    root.mkdir(parents=True, exist_ok=True)
    vids = sorted({r["video_id"] for r in rows})
    titles = {v["id"]: v.get("title") or v["id"] for v in videos(cfg)}
    keys = [m[0] for m in METRICS]
    clips = []
    for r in rows:
        vals = []
        for key in keys:
            metric = RULES[key][0]
            x = edge_db(r) if metric == "edge_db" else r.get(metric)
            vals.append(None if x is None else round(float(x), 4))
        clips.append([r["id"], os.path.relpath(r["path"], root).replace(os.sep, "/"), vids.index(r["video_id"]),
                      round(r["start"], 1), round(r["duration"], 2), vals])
    f = cfg.filter
    metrics = [{"key": k, "name": n, "step": st, "dir": RULES[k][1], "explain": ex,
                "on": f.get(k) is not None, "value": f.get(k) if f.get(k) is not None else d}
               for k, n, st, d, ex in METRICS]
    labels_path = root / "labels.json"
    seed_labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.exists() else {}
    data = {"clips": clips, "videos": [[v, titles.get(v, v)] for v in vids], "metrics": metrics,
            "labels": seed_labels, "seg": [cfg.segmentation.min_sec, cfg.segmentation.max_sec]}
    page = PAGE.replace("/*DATA*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    out = root / "tuner.html"
    out.write_text(page, encoding="utf-8")
    logger.info(f"wrote {out} ({len(clips):,} clips from {len(vids)} videos; open it in Chrome / Edge)")


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>EgySpeech filter tuner</title>
<style>
:root{--bg:#f7f7f8;--card:#fff;--fg:#1d1d1f;--muted:#6e6e73;--line:#e3e3e8;--ok:#1f7a3a;--okbg:#e6f4ea;--bad:#b3261e;
--badbg:#fbe9e7;--acc:#2f5bd3;--accbg:#e8eefc;--bar:#9aa7c7;--barbad:#e3a59f}
@media (prefers-color-scheme:dark){:root{--bg:#121214;--card:#1c1c1f;--fg:#f2f2f7;--muted:#a1a1a6;--line:#2e2e33;
--ok:#5ad27d;--okbg:#17301f;--bad:#ff6b5e;--badbg:#3a1d1a;--acc:#8aa8ff;--accbg:#1f2741;--bar:#5d6a8c;--barbad:#8a4640}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--card);border-bottom:1px solid var(--line);padding:10px 16px;
display:flex;flex-wrap:wrap;gap:8px 18px;align-items:center}
header h1{font-size:17px;margin:0 8px 0 0}.big{font-size:20px;font-weight:700}.muted{color:var(--muted)}
.ok{color:var(--ok)}.bad{color:var(--bad)}
main{display:grid;grid-template-columns:minmax(330px,430px) 1fr;gap:16px;padding:16px;max-width:1700px;margin:0 auto}
@media (max-width:900px){main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px;margin-bottom:12px}
.metric.off{opacity:.55}.metric .head{display:flex;align-items:center;gap:8px}.metric .head b{flex:1}
.metric .explain{font-size:12.5px;color:var(--muted);margin:6px 0}.metric .ctrl{display:flex;gap:8px;align-items:center}
.metric input[type=range]{flex:1}.metric input[type=number]{width:84px}.alone{font-size:12px;margin-top:4px}
.focus{outline:2px solid var(--acc)}
svg.hist{width:100%;height:58px;display:block;margin-top:6px}
.tabs{display:flex;gap:6px;flex-wrap:wrap;align-items:center}.tabs button,.btn{border:1px solid var(--line);
background:var(--card);color:var(--fg);border-radius:8px;padding:5px 10px;cursor:pointer;font:inherit}
.tabs button.on{background:var(--accbg);border-color:var(--acc);color:var(--acc);font-weight:600}
select{font:inherit;padding:4px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg)}
.clip{display:grid;grid-template-columns:auto 250px 1fr;gap:10px;align-items:center;padding:8px 0;border-bottom:1px solid var(--line)}
@media (max-width:700px){.clip{grid-template-columns:1fr}}
.clip audio{width:250px;height:32px}.lab button{width:30px;height:28px;border-radius:7px;border:1px solid var(--line);
background:var(--card);cursor:pointer;font-size:14px;color:var(--fg)}.lab .g.on{background:var(--okbg);border-color:var(--ok);color:var(--ok)}
.lab .b.on{background:var(--badbg);border-color:var(--bad);color:var(--bad)}
.vals{font-size:12px;color:var(--muted)}.vals span{margin-right:10px;white-space:nowrap}.vals .fail{color:var(--bad);font-weight:600}
.vals .foc{text-decoration:underline}.chip{display:inline-block;font-size:11px;border-radius:6px;padding:1px 6px;margin-left:6px;
background:var(--badbg);color:var(--bad)}.id{font-family:ui-monospace,monospace;font-size:12px}
textarea{width:100%;height:210px;font:12px ui-monospace,monospace;background:var(--bg);color:var(--fg);border:1px solid var(--line);
border-radius:8px;padding:8px}a{color:inherit}
</style></head><body>
<header>
  <h1>Filter tuner</h1>
  <div><span class="big ok" id="nKept"></span> <span class="muted" id="nTotal"></span></div>
  <div><span class="big" id="hKept"></span> <span class="muted" id="hTotal"></span></div>
  <div id="labStats" class="muted"></div>
</header>
<main>
<section>
  <div class="card muted" style="font-size:12.5px">Switch metrics on one at a time, starting from the top. Move the
  threshold and listen to the clips at the top of both lists: they are the closest to the threshold of the metric you
  touched last (outlined). Mark clips ✓ (valid) / ✗ (must be rejected): the header counts bad clips that would be
  accepted and good clips that would be lost. Settings and labels are remembered in this browser.</div>
  <div id="metrics"></div>
  <div class="card"><b>Settings for configs/config.yaml</b>
    <p class="muted" style="font-size:12.5px;margin:4px 0 8px">Replace the <code>filter:</code> section with this, then run
    <code>python -m egyspeech.cli filter</code> (or send it to me).</p>
    <textarea id="yaml" readonly></textarea>
    <div style="display:flex;gap:8px;margin-top:8px;flex-wrap:wrap"><button class="btn" id="copy">Copy</button>
    <button class="btn" id="exportLabels">Download labels.json</button><button class="btn" id="reset">Reset settings</button></div>
  </div>
</section>
<section>
  <div class="card">
    <div class="tabs">
      <button id="tabA" class="on"></button><button id="tabR"></button>
      <span style="flex:1"></span>
      <label class="muted">order <select id="order"><option value="border">closest to threshold</option>
        <option value="random">random</option><option value="low">lowest value first</option>
        <option value="high">highest value first</option></select></label>
      <label class="muted">video <select id="video"></select></label>
      <label class="muted">show <select id="show"><option value="all">all</option><option value="lab">labeled</option>
        <option value="unlab">not labeled</option><option value="wrong">wrongly decided</option></select></label>
    </div>
    <div id="list"></div>
    <div style="text-align:center;margin-top:10px"><button class="btn" id="more">Show more</button></div>
  </div>
</section>
</main>
<script>
const D = /*DATA*/null;
const M = D.metrics, C = D.clips, STORE = "egyspeech-tuner-v1";
let saved = {}; try { saved = JSON.parse(localStorage.getItem(STORE) || "{}"); } catch (e) {}
const state = {on: {}, val: {}, focus: 0, tab: "A", order: "border", video: "all", show: "all", page: 40,
               labels: Object.assign({}, D.labels || {}, saved.labels || {})};
M.forEach((m, i) => { state.on[i] = saved.on && i in saved.on ? saved.on[i] : m.on;
                      state.val[i] = saved.val && i in saved.val ? saved.val[i] : m.value; });
if (saved.focus != null) state.focus = saved.focus;
const persist = () => { try { localStorage.setItem(STORE, JSON.stringify({on: state.on, val: state.val, focus: state.focus,
  labels: state.labels})); } catch (e) {} };
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const digits = m => m.step >= 0.5 ? 1 : m.step >= 0.01 ? 2 : m.step >= 0.001 ? 3 : 4;
const fails = (c, i) => { const x = c[5][i]; if (x == null) return false;
  return M[i].dir === "min" ? x < state.val[i] : x > state.val[i]; };
const rejectedBy = c => M.map((m, i) => i).filter(i => state.on[i] && fails(c, i));
const ranges = M.map((m, i) => { const xs = C.map(c => c[5][i]).filter(x => x != null);
  return xs.length ? [Math.min(...xs), Math.max(...xs)] : [0, 1]; });

function buildMetrics() {
  const box = document.getElementById("metrics");
  box.innerHTML = M.map((m, i) => `<div class="card metric" id="m${i}">
    <div class="head"><input type="checkbox" id="on${i}"><b>${esc(m.name)}</b>
      <span class="muted" style="font-size:12px">${m.dir === "min" ? "keep ≥" : "keep ≤"}</span></div>
    <div class="explain">${m.explain}</div>
    <div class="ctrl"><input type="range" id="r${i}" min="${ranges[i][0]}" max="${ranges[i][1]}" step="${m.step}">
      <input type="number" id="n${i}" step="${m.step}"></div>
    <svg class="hist" id="h${i}" viewBox="0 0 400 58" preserveAspectRatio="none"></svg>
    <div class="alone muted" id="a${i}"></div></div>`).join("");
  M.forEach((m, i) => {
    const on = document.getElementById("on" + i), r = document.getElementById("r" + i), n = document.getElementById("n" + i);
    on.checked = state.on[i]; r.value = state.val[i]; n.value = (+state.val[i]).toFixed(digits(m));
    const set = v => { state.val[i] = +v; state.focus = i; r.value = v; update(); };
    on.onchange = () => { state.on[i] = on.checked; state.focus = i; update(); };
    r.oninput = () => { n.value = (+r.value).toFixed(digits(m)); set(r.value); };
    n.oninput = () => { if (n.value !== "" && !isNaN(+n.value)) set(n.value); };
    document.getElementById("m" + i).onclick = e => { if (e.target.tagName !== "INPUT" && state.focus !== i) { state.focus = i; update(); } };
  });
}

function hist(i) {
  const m = M[i], [lo, hi] = ranges[i], bins = 50, w = 400 / bins, cnt = new Array(bins).fill(0);
  const bin = x => Math.min(bins - 1, Math.max(0, Math.floor((x - lo) / ((hi - lo) || 1) * bins)));
  C.forEach(c => { const x = c[5][i]; if (x != null) cnt[bin(x)]++; });
  const top = Math.max(...cnt, 1), t = state.val[i], tx = ((t - lo) / ((hi - lo) || 1)) * 400;
  let s = "";
  cnt.forEach((n, b) => { const x0 = lo + (b + 0.5) * (hi - lo) / bins;
    const bad = state.on[i] && (m.dir === "min" ? x0 < t : x0 > t), h = n / top * 44;
    s += `<rect x="${b * w + 0.5}" y="${46 - h}" width="${w - 1}" height="${h}" fill="var(${bad ? "--barbad" : "--bar"})"/>`; });
  C.forEach(c => { const lab = state.labels[c[0]], x = c[5][i]; if (!lab || x == null) return;
    const px = (x - lo) / ((hi - lo) || 1) * 400;
    s += `<rect x="${px - 1}" y="${lab === "good" ? 49 : 53}" width="2.5" height="4" fill="var(${lab === "good" ? "--ok" : "--bad"})"/>`; });
  if (state.on[i]) s += `<line x1="${tx}" x2="${tx}" y1="0" y2="58" stroke="var(--acc)" stroke-width="2"/>`;
  document.getElementById("h" + i).innerHTML = s;
}

function update() {
  const kept = [], rej = [];
  let hk = 0, ht = 0;
  C.forEach(c => { const r = rejectedBy(c); ht += c[4]; (r.length ? rej : kept).push(c); if (!r.length) hk += c[4]; });
  document.getElementById("nKept").textContent = kept.length.toLocaleString() + " kept";
  document.getElementById("nTotal").textContent = `of ${C.length.toLocaleString()} clips (${(100 * kept.length / C.length).toFixed(0)}%)`;
  document.getElementById("hKept").textContent = (hk / 3600).toFixed(2) + " h";
  document.getElementById("hTotal").textContent = `of ${(ht / 3600).toFixed(2)} h`;
  const ids = new Set(kept.map(c => c[0])), L = Object.entries(state.labels);
  const g = L.filter(([, v]) => v === "good"), b = L.filter(([, v]) => v === "bad");
  const lost = g.filter(([k]) => !ids.has(k) && C.some(c => c[0] === k)).length;
  const leak = b.filter(([k]) => ids.has(k)).length;
  document.getElementById("labStats").innerHTML = `labels: <b class="${leak ? "bad" : "ok"}">${leak} bad accepted</b> of ${b.length}
    · <b class="${lost ? "bad" : "ok"}">${lost} good rejected</b> of ${g.length}`;
  M.forEach((m, i) => {
    document.getElementById("m" + i).classList.toggle("off", !state.on[i]);
    document.getElementById("m" + i).classList.toggle("focus", state.focus === i);
    const alone = C.filter(c => fails(c, i)).length;
    document.getElementById("a" + i).textContent = `alone this threshold rejects ${alone.toLocaleString()} clips`
      + (state.on[i] ? "" : " (off)");
    hist(i);
  });
  document.getElementById("tabA").textContent = `Accepted (${kept.length.toLocaleString()})`;
  document.getElementById("tabR").textContent = `Rejected (${rej.length.toLocaleString()})`;
  state.kept = kept; state.rej = rej; state.page = 40;
  renderList(); renderYaml(); persist();
}

let shuffleKey = {};
C.forEach(c => shuffleKey[c[0]] = Math.random());
function renderList() {
  const i = state.focus, t = state.val[i];
  let rows = (state.tab === "A" ? state.kept : state.rej).slice();
  if (state.video !== "all") rows = rows.filter(c => c[2] === +state.video);
  const lab = id => state.labels[id];
  if (state.show === "lab") rows = rows.filter(c => lab(c[0]));
  if (state.show === "unlab") rows = rows.filter(c => !lab(c[0]));
  if (state.show === "wrong") rows = rows.filter(c => lab(c[0]) === (state.tab === "A" ? "bad" : "good"));
  const v = c => c[5][i] == null ? Infinity : c[5][i];
  if (state.order === "random") rows.sort((a, b) => shuffleKey[a[0]] - shuffleKey[b[0]]);
  else if (state.order === "low") rows.sort((a, b) => v(a) - v(b));
  else if (state.order === "high") rows.sort((a, b) => v(b) - v(a));
  else rows.sort((a, b) => Math.abs(v(a) - t) - Math.abs(v(b) - t));
  const html = rows.slice(0, state.page).map(c => {
    const why = rejectedBy(c), [vid, title] = D.videos[c[2]], s = Math.floor(c[3]);
    const url = vid.length === 11 ? `https://www.youtube.com/watch?v=${vid}&t=${s}s` : null;
    const vals = M.map((m, k) => c[5][k] == null ? "" : `<span class="${state.on[k] && fails(c, k) ? "fail" : ""} ${k === i ? "foc" : ""}"
      title="${esc(m.name)}">${esc(m.name.replace(/ \(.*\)/, "").replace("Single speaker", "voice"))} ${(+c[5][k]).toFixed(digits(m))}</span>`).join("");
    const L = lab(c[0]);
    return `<div class="clip"><div class="lab"><button class="g ${L === "good" ? "on" : ""}" data-id="${esc(c[0])}" data-l="good" title="valid">✓</button>
      <button class="b ${L === "bad" ? "on" : ""}" data-id="${esc(c[0])}" data-l="bad" title="must be rejected">✗</button></div>
      <audio controls preload="none" src="${esc(c[1])}"></audio>
      <div><span class="id">${esc(c[0])}</span> <span class="muted">${c[4].toFixed(1)} s</span>
      ${why.map(k => `<span class="chip">${esc(M[k].name.replace(/ \(.*\)/, ""))}</span>`).join("")}
      <div class="vals">${vals}</div>
      <div class="muted" style="font-size:12px">${url ? `<a href="${url}" target="_blank">${esc(title.slice(0, 70))} @ ${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}</a>` : esc(title.slice(0, 70))}</div></div></div>`;
  }).join("");
  document.getElementById("list").innerHTML = html || '<p class="muted">no clips</p>';
  document.getElementById("more").style.display = rows.length > state.page ? "" : "none";
}

function renderYaml() {
  const lines = ["filter:"];
  M.forEach((m, i) => { const v = state.on[i] ? (+state.val[i]).toFixed(digits(m)) : "null";
    lines.push(`  ${m.key}: ${v}`.padEnd(34) + `# ${m.name}${state.on[i] ? "" : " (off)"}`); });
  lines.push("  drop_weak_cuts: false");
  document.getElementById("yaml").value = lines.join("\n");
}

document.getElementById("list").onclick = e => {
  const btn = e.target.closest("button[data-id]"); if (!btn) return;
  const id = btn.dataset.id, l = btn.dataset.l;
  if (state.labels[id] === l) delete state.labels[id]; else state.labels[id] = l;
  const page = state.page; update(); state.page = page; renderList();
};
document.getElementById("tabA").onclick = () => { state.tab = "A"; state.page = 40; tabs(); renderList(); };
document.getElementById("tabR").onclick = () => { state.tab = "R"; state.page = 40; tabs(); renderList(); };
const tabs = () => { document.getElementById("tabA").classList.toggle("on", state.tab === "A");
                     document.getElementById("tabR").classList.toggle("on", state.tab === "R"); };
["order", "video", "show"].forEach(k => document.getElementById(k).onchange = e => { state[k] = e.target.value; state.page = 40; renderList(); });
document.getElementById("video").innerHTML = '<option value="all">all</option>' +
  D.videos.map(([v, t], i) => `<option value="${i}">${esc(t.slice(0, 40))}</option>`).join("");
document.getElementById("more").onclick = () => { state.page += 40; renderList(); };
document.getElementById("copy").onclick = () => { const t = document.getElementById("yaml"); t.select();
  (navigator.clipboard ? navigator.clipboard.writeText(t.value) : Promise.reject()).catch(() => document.execCommand("copy")); };
document.getElementById("exportLabels").onclick = () => { const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(state.labels, null, 1)], {type: "application/json"}));
  a.download = "labels.json"; a.click(); };
document.getElementById("reset").onclick = () => { M.forEach((m, i) => { state.on[i] = m.on; state.val[i] = m.value; });
  buildMetrics(); update(); };
buildMetrics(); tabs(); update();
</script></body></html>
"""

if __name__ == "__main__":
    step_main(main, lambda p: p.add_argument("--max-clips", type=int, default=30000,
                                             help="random sample of clips shown (the page stays fast)"))
