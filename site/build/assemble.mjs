import { writeFileSync } from "node:fs";
import { ISO_CSS } from "../vendor/anatomy/kit/iso-kit.mjs";
import { kitScript } from "../vendor/anatomy/scripts/inline-kit.mjs";
import * as scale from "./fig-scale.mjs";
import * as kase from "./fig-case.mjs";
import * as chest from "./fig-chest.mjs";

const fig = (id, m) => `<div class="iso fig" id="${id}">${m.html}</div>`;

// Artificial Analysis Intelligence Index, medium reasoning effort for API models (Oct 2026)
const LADDER = [
  ["Claude Opus 5.5", "anthropic", 51, "medium"],
  ["GPT-6 Astra", "openai", 50, "medium"],
  ["Claude Fable 5.1", "anthropic", 49, "medium"],
  ["GPT-6.1 Sol", "openai", 48, "medium"],
  ["GLM-5.3-Flash (original)", "base", 42, "full-precision base"],
  ["GLM 5.3 Flash-sauhard", "ours", 40.5, "estimated"],
  ["Claude Sonnet 5.5", "anthropic", 41, "medium"],
  ["Qwen3.8-Flash-Next", "open", 40, "open weights"],
  ["Claude Haiku 5.5", "anthropic", 34, "medium"],
  ["GPT-6 Luna", "openai", 30, "medium"],
];
// [benchmark, GLM-5.3-Flash, Sonnet 5.5 medium, GPT-6.1 Sol medium, unit, higherBetter]
const H2H = [
  ["Intelligence Index", 42, 41, 48, ""],
  ["GDPval-AA (real work, Elo)", 1642, 1323, 1450, ""],
  ["AA-Briefcase (Elo)", 1453, 1441, 1361, ""],
  ["AutomationBench-AA", 60, 55, 63, "%"],
  ["AA-LCR long context", 80, 76, 83, "%"],
  ["Terminal-Bench 4.0", 33, 30, 48, "%"],
  ["SciCode", 52, 53, 53, "%"],
  ["Humanity's Last Exam", 40, 40, 50, "%"],
  ["CritPt", 15, 17, 28, "%"],
  ["GDP.pdf", 15, 20, 30, "%"],
  ["AA-Omniscience (−100 to 100)", 7, 20, 40, ""],
];
const cell = (v, u, best) => `<td class="n${best ? " best" : ""}">${v.toLocaleString("en-US")}${u}</td>`;
const h2hRows = H2H.map(([n, g, s, o, u]) => {
  const m = Math.max(g, s, o);
  return `<tr><td>${n}</td>${cell(g, u, g === m)}${cell(s, u, s === m)}${cell(o, u, o === m)}</tr>`;
}).join("");
const wins = H2H.filter(([, g, s]) => g > s).length, ties = H2H.filter(([, g, s]) => g === s).length;

const ladder = LADDER.map(([n, who, v, eff]) => `<li class="rung ${who}"><span class="rn">${n}<small>${eff}</small></span><span class="rt"><i style="--w:${(v / 56) * 100}%"></i></span><b>${who === "ours" ? "40–41*" : v}</b></li>`).join("");

const LESSONS = [
  ["Evaluation", "A mean KL of 2.9 means the build is broken.", "The error lives where the model reads, not where it writes.", "Split by chat role, assistant tokens sat at KL 0.12 with 92% top-1 agreement. User and tool positions run at KL 3 to 7 for every build, the reference included, because no chat model predicts your prompt word for word."],
  ["Precision", "Recalibrating in fp32 will fix it.", "fp32 measured no better than bf16.", "Splitting the metric by role made a five-hour fp32 recalibration pointless. We ran a slice anyway to be sure."],
  ["Pruning", "REAP saliency picks the right experts.", "On GLM, REAP scores are almost flat.", "Median 5.7 against a cutoff of 5.5. The 153 it keeps carry 54% of routing on average, where the busiest 153 would carry 68%."],
  ["Rounding", "Pruning choices matter most.", "How you round matters more than what you cut.", "Refitting every scale and bias by importance-weighted least squares cut assistant KL 10.5% at the same size. Pruning variants moved it 2 to 3%."],
  ["Recovery", "Fixing every layer locally fixes the model.", "All 45 layers improved; the model did not.", "Blockwise distillation lowered every layer's error and nudged assistant KL to 0.106, but read prompts worse and coded slightly worse. We shipped the simpler build."],
  ["Engineering", "The trainer's memory is fine.", "It leaked 1.46 GB per layer.", "MLX keeps a custom function's inputs alive, so each layer's expert weights stayed pinned: 110 GB projected by layer 43. Fixed, it ran flat at 24 GB, and sparse layers went from 17 minutes to 2."],
];
const lessons = LESSONS.map(([tag, wrong, right, body]) => `<article class="lesson"><span class="tag">${tag}</span><p class="wrong">${wrong}</p><p class="right">${right}</p><p class="why">${body}</p></article>`).join("");

const KL = [
  ["REAP-153, round to nearest", 0.1206, "base"],
  ["Mass-153, round to nearest", 0.1231, "base"],
  ["Mass-162, 80 GB", 0.1176, "base"],
  ["Mass-162, LSQ, 80 GB", 0.1118, "base"],
  ["GLM 5.3 Flash-sauhard", 0.1079, "win"],
  ["+ DWQ, cosine", 0.1064, "dwq"],
  ["+ DWQ, MSE", 0.1069, "dwq"],
];
const klRows = KL.map(([n, v, c]) => `<li class="kl ${c}"><span>${n}</span><span class="kt"><i style="--w:${(v / 0.13) * 100}%"></i></span><b>${v.toFixed(3)}</b></li>`).join("");


// open builds that target a 128 GB Mac (Hugging Face, October 2026). fidelity: [top1 %, KL], author-published unless ours
const FIELD = [
  ["GLM 5.3 Flash-sauhard", "GLM-5.3-Flash", 42, 71, "MLX · REAP 153 + 3.48-bit LSQ", "92.4% / 0.108", "measured, assistant tokens", 1],
  ["ISTA-DASLab GSQ-RCO IQ3_S", "Qwen3.8-Flash-Next", 40, 83.6, "llama.cpp · non-uniform 3-bit", "task avg 93.3 vs 93.1 BF16", "author, task suite", 0],
  ["unsloth UD-IQ1_S", "GLM-5.3-Flash", 42, 93.1, "llama.cpp · all experts, ~1.6-bit", "70.9% / 0.670", "author, generic text", 0],
  ["unsloth UD-IQ4_XS", "Qwen3.8-Flash-Next", 40, 93.7, "llama.cpp · all experts", "not published", "", 0],
  ["patrickbdevaney REAP50", "GLM-5.3-Flash", 42, 72, "llama.cpp · 144 experts, imatrix", "84.3% (unquantized parent)", "author", 0],
  ["orcarouter 2bit-lite", "GLM-5.3-Flash", 42, 102, "MLX · all experts", "77.2%, PPL +141%", "author", 0],
  ["pfeifferj GSQ-RCO 3.0-bit", "GLM-5.3-Flash", 42, 117.5, "llama.cpp (patched)", "KL ~0.142", "author", 0],
  ["unsloth UD-IQ3_XXS", "GLM-5.3-Flash", 42, 120.4, "llama.cpp · all experts", "81.6% / 0.284", "author, generic text", 0],
  ["sh0wie REAP-288 4-bit", "Qwen3.8-Flash-Next", 40, 68, "MLX · 288 of 512 experts", "HumanEval 91.5 vs 93.9", "author", 0],
  ["Litwein REAP320 oQ3e DWQ", "Qwen3.8-Flash-Next", 40, 70.8, "oMLX only", "KL 0.104 (held-out)", "author", 0],
];
const fieldRows = FIELD.map(([n, base, aa, gb, how, fid, src, us]) => `<tr${us ? ' class="us-row"' : ""}><td><b>${n}</b><small>${how}</small></td><td>${base}<small>AA ${aa}</small></td><td class="n${gb > 84 ? " over" : ""}">${gb}</td><td>${fid}<small>${src}</small></td></tr>`).join("");

const page = `<title>GLM 5.3 Flash-sauhard</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Familjen+Grotesk:wght@400;500;600;700&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
${ISO_CSS}
/* Layout: one long scroll in seven chapters. A fixed chapter rail on wide screens; each chapter is prose on the left
   and its instrument on the right, stacking to one column on phones. Figures keep the Anatomy kit's own palette. */
:root{
  --bg:#f2f2ef; --card:#fbfbfa; --ink:#18181a; --text:#46464a; --muted:#8a8a8f; --line:#dcdcd9; --faint:#e8e8e5;
  --accent:#3e63dd; --accent-soft:#e3e9fb; --warn:#c2410c; --good:#30a46c;
  --anth:#c4693d; --oai:#4b5563; --open:#9a9aa0;
  --display:"Familjen Grotesk","Helvetica Neue",Arial,sans-serif;
  --body:"Newsreader",Georgia,"Times New Roman",serif;
  --mono:"JetBrains Mono",ui-monospace,Menlo,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#0b0b0b; --card:#131313; --ink:#ececec; --text:#b4b4b4; --muted:#6e6e6e; --line:#232323; --faint:#1a1a1a;
  --accent:#7b9dff; --accent-soft:#16203a; --warn:#fb923c; --good:#6be0a0; --anth:#d9825a; --oai:#8b95a5; --open:#5c5c62; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#0b0b0b; --card:#131313; --ink:#ececec; --text:#b4b4b4; --muted:#6e6e6e; --line:#232323; --faint:#1a1a1a;
  --accent:#7b9dff; --accent-soft:#16203a; --warn:#fb923c; --good:#6be0a0; --anth:#d9825a; --oai:#8b95a5; --open:#5c5c62; color-scheme:dark}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{background:var(--bg);color:var(--text);font:18px/1.62 var(--body);-webkit-font-smoothing:antialiased}
body::before{content:"";position:fixed;inset:0 0 auto;height:env(safe-area-inset-top,0px);z-index:20;background:var(--bg)}
a{color:var(--accent);text-underline-offset:3px}
:focus-visible{outline:2px solid var(--accent);outline-offset:3px}
h1,h2,h3{font-family:var(--display);color:var(--ink);text-wrap:balance;margin:0;letter-spacing:-.02em;line-height:1.05}
p{margin:0}
.wrap{max-width:1180px;margin:0 auto;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
.mono{font-family:var(--mono)}
.eyebrow{font:500 12px/1.4 var(--mono);letter-spacing:.14em;text-transform:uppercase;color:var(--muted)}

/* rail */
.rail{position:fixed;left:max(14px,env(safe-area-inset-left,0px));top:50%;transform:translateY(-50%);z-index:15;display:flex;flex-direction:column;gap:4px;list-style:none;margin:0;padding:0}
.rail a{display:flex;align-items:center;gap:10px;text-decoration:none;color:var(--muted);font:500 11px var(--mono);letter-spacing:.08em;text-transform:uppercase;padding:3px 0}
.rail a i{width:14px;height:2px;background:var(--line);transition:width .3s,background .3s}
.rail a span{opacity:0;transform:translateX(-4px);transition:opacity .2s,transform .2s}
.rail:hover a span,.rail a.on span{opacity:1;transform:none}
.rail a.on{color:var(--ink)} .rail a.on i{width:28px;background:var(--accent)}
@media (max-width:1320px){.rail{display:none}}

/* hero */
.hero{padding-block:clamp(56px,10vw,120px) 48px}
.hero h1{font-size:clamp(46px,8.6vw,118px);font-weight:700;letter-spacing:-.045em;line-height:.92}
.hero h1 .q{color:var(--muted);font-weight:500}
.hero .lede{max-width:62ch;font-size:clamp(19px,2vw,22px);margin-top:28px;color:var(--text)}
.hero .lede b{color:var(--ink);font-weight:500}
.numbers{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:0;margin-top:48px;border-top:1px solid var(--line)}
.numbers div{padding:18px 18px 0 0;display:flex;flex-direction:column;gap:2px;min-width:0}
.numbers b{font:600 clamp(30px,4vw,46px)/1 var(--display);color:var(--ink);letter-spacing:-.03em;font-variant-numeric:tabular-nums}
.numbers b small{font-size:.42em;color:var(--muted);margin-left:3px;font-weight:500}
.numbers span{font:12px/1.45 var(--mono);color:var(--muted)}
@media (max-width:760px){.numbers{grid-template-columns:repeat(2,minmax(0,1fr))}}
.begin{display:inline-flex;gap:10px;align-items:center;margin-top:40px;font:500 14px var(--display);color:var(--ink);text-decoration:none;border:1px solid var(--line);padding:10px 16px;border-radius:999px;background:var(--card)}
.begin:hover{border-color:var(--accent)}
.begin svg{width:14px;height:14px;fill:none;stroke:currentColor;stroke-width:1.6}

/* chapters */
.ch{padding-block:clamp(64px,9vw,120px);border-top:1px solid var(--line)}
.ch-grid{display:grid;grid-template-columns:minmax(0,5fr) minmax(0,7fr);gap:clamp(24px,4vw,56px);align-items:start}
.ch-grid.flip{grid-template-columns:minmax(0,7fr) minmax(0,5fr)}
.ch-grid.flip .story{order:2}
@media (max-width:980px){.ch-grid,.ch-grid.flip{grid-template-columns:1fr}.ch-grid.flip .story{order:0}}
.story{display:flex;flex-direction:column;gap:18px;min-width:0;position:sticky;top:calc(env(safe-area-inset-top,0px) + 32px)}
@media (max-width:980px){.story{position:static}}
.story h2{font-size:clamp(32px,4.4vw,52px);font-weight:650}
.story p{max-width:56ch}
.story .num{font:500 13px var(--mono);color:var(--accent);letter-spacing:.06em}
.facts{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1px;background:var(--line);border:1px solid var(--line);border-radius:10px;overflow:hidden;margin-top:6px}
.facts div{background:var(--card);padding:12px 14px;display:flex;flex-direction:column;min-width:0}
.facts b{font:600 20px var(--display);color:var(--ink);font-variant-numeric:tabular-nums}
.facts span{font:12px/1.4 var(--mono);color:var(--muted)}
.fig{min-width:0}
.fig .iso-figure{margin:0}
.fig .iso-legend{font-family:var(--mono)}
.fmt-row{display:flex;flex-wrap:wrap;gap:6px;padding:4px 16px 16px;position:relative;z-index:2}
.fmt{font:500 12px var(--mono);background:transparent;border:1px solid var(--anatomy-faint);color:var(--anatomy-text);border-radius:6px;padding:6px 9px;cursor:pointer;display:inline-flex;gap:8px;align-items:baseline}
.fmt:hover{border-color:var(--anatomy-mid)}
.fmt[aria-pressed="true"]{color:var(--anatomy-strong);border-color:var(--anatomy-hi)}
.fmt b{font-weight:500;color:var(--anatomy-muted)}
.fmt[aria-pressed="true"] b{color:var(--anatomy-strong)}
.fmt-note{font:13px/1.5 var(--mono);color:var(--anatomy-text);padding:0 16px 14px;margin:0}

/* streaming strip (calibration) */
.stream{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:22px;display:flex;flex-direction:column;gap:16px}
.layers{display:grid;grid-template-columns:repeat(45,1fr);gap:3px;height:64px;align-items:end}
.layers i{display:block;height:28%;background:var(--line);border-radius:2px;transition:height .35s,background .35s}
.layers i.dense{height:20%}
.layers i.done{background:color-mix(in srgb,var(--accent) 40%,var(--line))}
.layers i.now{height:100%;background:var(--accent)}
.stream-meta{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;font:12px var(--mono);color:var(--muted)}
.stream-meta b{color:var(--ink);font-weight:500}
.mix{display:flex;height:12px;border-radius:6px;overflow:hidden}
.mix i{display:block;height:100%}
.mixkey{display:flex;flex-wrap:wrap;gap:4px 16px;font:12px var(--mono);color:var(--muted)}
.mixkey i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:6px}

/* KL bars */
.panel{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:22px}
.panel h3{font-size:18px;font-weight:600;margin-bottom:4px}
.panel .sub{font:12px var(--mono);color:var(--muted);margin-bottom:14px}
.klist{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:7px}
.kl{display:grid;grid-template-columns:minmax(0,190px) minmax(0,1fr) 52px;gap:12px;align-items:center;font:13px var(--mono);color:var(--text)}
.kl .kt{height:12px;background:var(--faint);border-radius:3px;overflow:hidden}
.kl .kt i{display:block;height:100%;width:var(--w);background:var(--open);border-radius:3px}
.kl.win .kt i{background:var(--accent)} .kl.dwq .kt i{background:color-mix(in srgb,var(--warn) 70%,var(--card))}
.kl.win span,.kl.win b{color:var(--ink)} .kl b{font-weight:500;text-align:right;font-variant-numeric:tabular-nums}
@media (max-width:560px){.kl{grid-template-columns:1fr 52px}.kl .kt{grid-column:1/-1;order:3}}

/* lessons */
.lessons{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin-top:40px}
@media (max-width:980px){.lessons{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media (max-width:620px){.lessons{grid-template-columns:1fr}}
.lesson{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px;display:flex;flex-direction:column;gap:10px;min-width:0}
.lesson .tag{font:500 11px var(--mono);letter-spacing:.1em;text-transform:uppercase;color:var(--warn)}
.lesson .wrong{font:500 16px/1.35 var(--display);color:var(--muted);text-decoration:line-through;text-decoration-color:color-mix(in srgb,var(--warn) 70%,transparent)}
.lesson .right{font:600 18px/1.3 var(--display);color:var(--ink)}
.lesson .why{font-size:16px;color:var(--text)}

/* standing */
.ladder{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:6px}
.rung{display:grid;grid-template-columns:minmax(0,200px) minmax(0,1fr) 52px;gap:12px;align-items:center}
.rn{font:500 14px var(--display);color:var(--ink);display:flex;flex-direction:column;line-height:1.2}
.rn small{font:11px var(--mono);color:var(--muted)}
.rt{height:22px;background:var(--faint);border-radius:5px;overflow:hidden}
.rt i{display:block;height:100%;width:var(--w);border-radius:5px;background:var(--open);transform-origin:left;animation:grow 1s cubic-bezier(.2,.7,.2,1) both}
.rung.anthropic .rt i{background:var(--anth)} .rung.openai .rt i{background:var(--oai)}
.rung.base .rt i{background:var(--accent)} .rung.base .rn{color:var(--accent)}
.rung.ours .rt i{background:repeating-linear-gradient(135deg,var(--accent) 0 6px,color-mix(in srgb,var(--accent) 45%,var(--card)) 6px 12px)} .rung.ours .rn{color:var(--accent)}
.hero .name{color:var(--accent);font-weight:500}
.rung b{font:600 15px var(--mono);color:var(--ink);text-align:right;font-variant-numeric:tabular-nums}
@keyframes grow{from{transform:scaleX(.2)}to{transform:none}}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;margin-top:14px;font:12px var(--mono);color:var(--muted)}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
.tbl{overflow-x:auto;margin-top:10px}
table{border-collapse:collapse;width:100%;font-size:14px;font-family:var(--mono)}
th,td{padding:9px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{font-weight:500;font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
td.best{color:var(--ink);font-weight:600}
th.us,td:nth-child(2){background:color-mix(in srgb,var(--accent-soft) 60%,transparent)}
.note{font:13px/1.55 var(--mono);color:var(--muted);margin-top:12px}
.verdicts{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px;margin-top:28px}
@media (max-width:860px){.verdicts{grid-template-columns:1fr}}
.verdicts div{border-top:2px solid var(--ink);padding-top:12px}
.verdicts b{display:block;font:600 16px var(--display);color:var(--ink);margin-bottom:4px}
.verdicts p{font-size:16px}
.stack2{display:flex;flex-direction:column;gap:18px;min-width:0}

table.field td{white-space:normal;vertical-align:top;min-width:120px}
table.field td small{display:block;color:var(--muted);font-size:11px;margin-top:2px}
table.field td b{color:var(--ink);font-weight:500}
table.field td.over{color:var(--warn)}
table.field td:nth-child(2){background:transparent}
tr.us-row td{background:color-mix(in srgb,var(--accent-soft) 70%,transparent)!important}
tr.us-row td b{color:var(--accent)}
/* run */
pre{margin:0;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;overflow-x:auto;font:13px/1.65 var(--mono);color:var(--ink)}
.copy{float:right;font:500 12px var(--display);border:1px solid var(--line);background:var(--bg);color:var(--ink);padding:4px 10px;border-radius:6px;cursor:pointer}
.open-items{list-style:none;padding:0;margin:0;display:flex;flex-direction:column;gap:10px}
.open-items li{padding-left:18px;position:relative}
.open-items li::before{content:"";position:absolute;left:0;top:.65em;width:8px;height:8px;border:1.5px solid var(--warn);border-radius:50%}
footer{border-top:1px solid var(--line);padding-block:48px 72px;font:13px/1.6 var(--mono);color:var(--muted)}
footer ol{padding-left:18px;margin:12px 0 0;columns:2;column-gap:32px}
footer li{break-inside:avoid;margin-bottom:6px}
@media (max-width:700px){footer ol{columns:1}}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important;scroll-behavior:auto!important}}
</style>

<nav aria-label="Chapters"><ul class="rail" id="rail">
<li><a href="#problem"><i></i><span>Problem</span></a></li>
<li><a href="#learn"><i></i><span>Calibrate</span></a></li>
<li><a href="#prune"><i></i><span>Prune</span></a></li>
<li><a href="#squeeze"><i></i><span>Quantize</span></a></li>
<li><a href="#detours"><i></i><span>Detours</span></a></li>
<li><a href="#standing"><i></i><span>Standing</span></a></li>
<li><a href="#field"><i></i><span>Open field</span></a></li>
<li><a href="#run"><i></i><span>Run it</span></a></li>
</ul></nav>

<header class="hero"><div class="wrap">
  <p class="eyebrow">GLM 5.3 Flash-sauhard · MacBook Pro M5 Max · 128 GB · MLX</p>
  <h1 style="margin-top:20px">321 billion parameters.<br><span class="q">One laptop.</span> Room to keep working.</h1>
  <p class="lede">GLM-5.3-Flash ships as a 305 GB checkpoint. We rebuilt it for one machine as <b class="name">GLM 5.3 Flash-sauhard</b>: <b>calibrated on real coding, agent and math work, pruned to the experts it needs, given a bit width per tensor and refit scale by scale</b>. GLM 5.3 Flash-sauhard runs offline in 71.6 GB and leaves the rest of the Mac to you. The model it is built from edges out Claude Sonnet 5.5 at medium effort on the Artificial Analysis index.</p>
  <div class="numbers">
    <div><b>71.6<small>GB</small></b><span>GLM 5.3 Flash-sauhard resident, of 128</span></div>
    <div><b>92.4<small>%</small></b><span>GLM 5.3 Flash-sauhard: next-token agreement with the original</span></div>
    <div><b>41.5<small>tok/s</small></b><span>GLM 5.3 Flash-sauhard decode, measured</span></div>
    <div><b>42<small>vs 41</small></b><span>AA index: original GLM-5.3-Flash vs Sonnet 5.5 (medium)</span></div>
  </div>
  <a class="begin" href="#problem">Follow the build <svg viewBox="0 0 16 16"><path d="M8 2v12M3 9l5 5 5-5"/></svg></a>
</div></header>

<section class="ch" id="problem"><div class="wrap ch-grid">
  <div class="story">
    <span class="num">01 · The problem</span>
    <h2>The model is four times the machine</h2>
    <p>GLM-5.3-Flash is a mixture-of-experts model: 321 billion weights, of which about 18 billion fire for any one token. Every expert still has to sit in memory, because the next token may need any of them.</p>
    <p>The Mac has 128 GB, and it is also the machine for the editor, the builds and Claude Code. So we set a hard rule: the model process gets at most 84 GB, and at least 45 GB stays free.</p>
    <p>Turn the knob on the scale. A plain 4-bit quantization still weighs 168 GB. Getting under the line took three more steps.</p>
  </div>
  ${fig("fig-scale", scale)}
</div></section>

<section class="ch" id="learn"><div class="wrap ch-grid flip">
  <div class="story">
    <span class="num">02 · Calibrate</span>
    <h2>First, watch it do our work</h2>
    <p>Generic quantizers calibrate on Wikipedia. We built 712 samples from the work this Mac actually does: GitHub files and commit edits, multi-turn agent traces with real tool calls, competition math, and 16K to 64K-token documents. 512 drive calibration; 200 are held out for scoring.</p>
    <p>The 305 GB checkpoint never fits, so calibration streams it one layer at a time: load, run every sample through, record, free. For each layer it records how often each expert is chosen, how much it contributes, and how much error every tensor takes at 2 to 8 bits.</p>
    <div class="facts">
      <div><b>36,764</b><span>sensitivity measurements</span></div>
      <div><b>&lt; 40 GB</b><span>peak memory while streaming</span></div>
      <div><b>5 h 27 m</b><span>for all 45 layers</span></div>
      <div><b>top-64</b><span>reference logprobs kept for scoring</span></div>
    </div>
  </div>
  <div class="stream" aria-label="Calibration streams one decoder layer at a time">
    <div class="stream-meta"><span>Decoder layers, streamed one at a time</span><span id="layer-now"><b>layer 0</b> of 45</span></div>
    <div class="layers" id="layers"></div>
    <div class="stream-meta"><span>3 dense layers, then 42 mixture-of-experts layers</span><span>only the lit layer is in memory</span></div>
    <div style="margin-top:8px" class="eyebrow">Calibration mix · 512 samples</div>
    <div class="mix"><i style="width:37.5%;background:var(--accent)"></i><i style="width:31.25%;background:color-mix(in srgb,var(--accent) 60%,var(--card))"></i><i style="width:28.1%;background:color-mix(in srgb,var(--accent) 32%,var(--card))"></i><i style="width:3.15%;background:var(--muted)"></i></div>
    <div class="mixkey"><span><i style="background:var(--accent)"></i>code · 192</span><span><i style="background:color-mix(in srgb,var(--accent) 60%,var(--card))"></i>agentic · 160</span><span><i style="background:color-mix(in srgb,var(--accent) 32%,var(--card))"></i>reasoning · 144</span><span><i style="background:var(--muted)"></i>long documents · 16</span></div>
  </div>
</div></section>

<section class="ch" id="prune"><div class="wrap ch-grid">
  <div class="story">
    <span class="num">03 · Prune</span>
    <h2>Keep 153 of 288 experts in every layer</h2>
    <p>Each layer's router sends a token to 8 of its 288 experts. Some are busy, some almost idle. Removing the quiet ones cuts 47% of the expert weight.</p>
    <p>We ranked experts with REAP, which scores how much an expert changes the output when it is chosen. The figure is real data from layer 20: sort height is how often the router picked each expert while it read our samples, and the stick in front shows the 8 picked for each new token.</p>
    <p>Switch to "Busiest 153" and the kept share of traffic rises from 54% to 70%. That gap was our first surprise, and the third detour below.</p>
  </div>
  ${fig("fig-case", kase)}
</div></section>

<section class="ch" id="squeeze"><div class="wrap ch-grid flip">
  <div class="story">
    <span class="num">04 · Quantize</span>
    <h2>Spend bits where the errors are</h2>
    <p>A uniform 4-bit model treats an attention projection and a rarely used expert the same. We gave each of 490 tensor groups its own width from the measured errors, with a solver that minimises total error under a 76 GB budget. The average came out at 3.48 bits per weight.</p>
    <p>Then we changed how each group is rounded. Instead of taking a scale from the group's minimum and maximum, we search the clipping range and solve a least-squares fit weighted by how strongly real inputs use each channel. Same bytes, 10.5% less divergence from the full model: the largest single gain of the project.</p>
    <div class="panel">
      <h3>Divergence from the full model</h3>
      <p class="sub">Assistant-token KL on 200 held-out samples · lower is better</p>
      <ul class="klist">${klRows}</ul>
      <p class="note">GLM 5.3 Flash-sauhard is the highlighted row. DWQ lowered this number a little more but read prompts and wrote code slightly worse, so it was not shipped.</p>
    </div>
  </div>
  ${fig("fig-chest", chest)}
</div></section>

<section class="ch" id="detours"><div class="wrap">
  <div class="story" style="position:static;max-width:760px">
    <span class="num">05 · Detours</span>
    <h2>Six things we believed that measurements overturned</h2>
    <p>Most of the time went into finding out why a number looked wrong. Each card is a belief we started with, struck through, and what we found instead.</p>
  </div>
  <div class="lessons">${lessons}</div>
</div></section>

<section class="ch" id="standing"><div class="wrap ch-grid">
  <div class="story">
    <span class="num">06 · Where it lands</span>
    <h2>Sonnet-class at the effort people actually use</h2>
    <p>Most people run hosted models at their default, medium reasoning effort, so that is the comparison here. On the Artificial Analysis Intelligence Index, the original GLM-5.3-Flash scores 42: one point above Claude Sonnet 5.5 at medium, and six below GPT-6.1 Sol at medium.</p>
    <p>Those are the original model's scores. GLM 5.3 Flash-sauhard matches it on 92.4% of next tokens, so expect a point or two lower until someone runs the suite on it. On our own tasks it scored 95% on HumanEval+, 92.5% on MBPP+, 84.8% on 105 tool calls and 97.5% on MATH-500 problems.</p>
    <div class="verdicts" style="grid-template-columns:1fr">
      <div><b>Original GLM-5.3-Flash vs Sonnet 5.5 (medium)</b><p>Ahead on real-work tasks (GDPval 1642 vs 1323), automation, long context and terminal agents. ${wins} of 11 benchmarks, ${ties} tie.</p></div>
      <div><b>Original GLM-5.3-Flash vs GPT-6.1 Sol (medium)</b><p>Behind on knowledge recall, hard science and terminal agents. Ahead of Sol only on real-work and briefcase tasks.</p></div>
    </div>
  </div>
  <div class="stack2">
    <div class="panel">
      <h3>Artificial Analysis Intelligence Index</h3>
      <p class="sub">API models at medium reasoning effort · October 2026</p>
      <ol class="ladder">${ladder}</ol>
      <div class="legend"><span><i style="background:var(--anth)"></i>Anthropic</span><span><i style="background:var(--oai)"></i>OpenAI</span><span><i style="background:var(--open)"></i>Open weights</span><span><i style="background:var(--accent)"></i>Original GLM-5.3-Flash</span><span><i style="background:repeating-linear-gradient(135deg,var(--accent) 0 3px,transparent 3px 6px)"></i>GLM 5.3 Flash-sauhard*</span></div>
    <p class="note">* Estimated from GLM 5.3 Flash-sauhard's 92.4% next-token agreement with the original; it has not been run on the Artificial Analysis suite.</p>
</div>
    <div class="panel">
      <h3>Head to head at medium effort</h3>
      <p class="sub">Independent Artificial Analysis results · best in each row in bold · GLM base is the original GLM-5.3-Flash</p>
      <div class="tbl"><table>
        <thead><tr><th>Benchmark</th><th class="n us">GLM base</th><th class="n">Sonnet 5.5</th><th class="n">GPT-6.1 Sol</th></tr></thead>
        <tbody>${h2hRows}</tbody>
      </table></div>
      <p class="note">Original, full-precision GLM-5.3-Flash. GLM 5.3 Flash-sauhard has not been run on these suites yet.</p>
    </div>
  </div>
</div></section>

<section class="ch" id="field"><div class="wrap">
  <div class="story" style="position:static;max-width:780px">
    <span class="num">07 · The open field</span>
    <h2>Against everything else built for a 128 GB Mac</h2>
    <p>We surveyed what people have published on Hugging Face for this class of machine. GLM-5.3-Flash is the highest-ranked open model that fits at all, and every other public build of it is either far larger or reports lower fidelity. At an 80 GB budget we found nothing clearly better than GLM 5.3 Flash-sauhard.</p>
    <p>The strongest challenger is a different model: ISTA-DASLab's 3-bit Qwen3.8-Flash-Next, 84 GB, matching its full-precision parent on its authors' task suite. Its base model scores two points lower than the base of GLM 5.3 Flash-sauhard, which is close to noise, so it is the next build we will test head to head.</p>
  </div>
  <div class="panel" style="margin-top:32px">
    <h3>Open builds for 128 GB Macs</h3>
    <p class="sub">Size in GB; red is over our 84 GB server cap · fidelity is top-1 agreement / KL against the full model</p>
    <div class="tbl"><table class="field">
      <thead><tr><th>Build</th><th>Base model</th><th class="n">GB</th><th>Published fidelity</th></tr></thead>
      <tbody>${fieldRows}</tbody>
    </table></div>
    <p class="note">These numbers are not like for like. GLM 5.3 Flash-sauhard's are measured on assistant tokens of held-out samples from our own workload; most others are measured on generic text over all tokens, and some only on tasks. The fair test is to run one harness over every build, which we have not done yet.</p>
  </div>
</div></section>

<section class="ch" id="run"><div class="wrap ch-grid flip">
  <div class="story">
    <span class="num">08 · Run it</span>
    <h2>A dashboard, two models, nothing in the cloud</h2>
    <p>A launchd service keeps a local control panel up. Start, stop or switch between GLM and a tuned Qwen3.8-Flash-Next profile, watch memory, and point any OpenAI- or Anthropic-compatible client at it, Claude Code included.</p>
    <p>The server clamps how much memory it may pin, caps itself at 84 GB, and skips a prefill copy that costs 22 to 24 GB at 32K context without changing a single output token.</p>
    <ul class="open-items">
      <li>Speculative decoding with GLM's own draft head is built in but not yet verified, so 41.5 tok/s is plain decode.</li>
      <li>Long-context retrieval at 64K to 128K has not been measured on this build.</li>
      <li>The index score for GLM 5.3 Flash-sauhard is inferred from its fidelity, not measured.</li>
    </ul>
  </div>
  <div class="stack2">
    <div><button class="copy" type="button" data-copy="c1">Copy</button><p class="eyebrow" style="margin-bottom:10px">Endpoints</p>
<pre id="c1">dashboard   http://127.0.0.1:8090
OpenAI API  http://127.0.0.1:8080/v1
Anthropic   http://127.0.0.1:8080/v1/messages
model       models/glm-5.3-flash-custom</pre></div>
    <div><button class="copy" type="button" data-copy="c2">Copy</button><p class="eyebrow" style="margin-bottom:10px">Memory guardrails</p>
<pre id="c2">mx.set_memory_limit(84 GB - 4 GB)
mx.set_wired_limit(min(n, WIRE))   # stock wires 107 GB
mx.set_cache_limit(2 GB)
_MAX_PROJECTED_PREFILL_TOKENS = 0  # -22 GB at 32K</pre></div>
    <div class="facts">
      <div><b>71.6 GB</b><span>resident while serving</span></div>
      <div><b>~45 GB</b><span>left for your work</span></div>
      <div><b>7</b><span>builds compared end to end</span></div>
      <div><b>54 tok/s</b><span>Qwen profile, for fast agent loops</span></div>
    </div>
  </div>
</div></section>

<footer><div class="wrap">
  <p class="eyebrow">Sources and method</p>
  <p style="margin-top:10px;max-width:90ch">Numbers for GLM 5.3 Flash-sauhard (size, divergence, speed, tasks) are our own measurements on this machine; Artificial Analysis numbers are for the original GLM-5.3-Flash. Figures 1 to 3 are drawn with the open-source Anatomy kit; every number they show comes from the build. Public benchmark numbers are from Artificial Analysis. GB means GiB as macOS reports it.</p>
  <ol>
    <li><a href="https://artificialanalysis.ai/leaderboards/models">Artificial Analysis, model leaderboard</a></li>
    <li><a href="https://artificialanalysis.ai/models/comparisons/glm-5-3-flash-vs-claude-sonnet-5-5-medium">GLM-5.3-Flash vs Claude Sonnet 5.5 (medium)</a></li>
    <li><a href="https://artificialanalysis.ai/models/comparisons/glm-5-3-flash-vs-gpt-6-1-sol-medium">GLM-5.3-Flash vs GPT-6.1 Sol (medium)</a></li>
    <li><a href="https://artificialanalysis.ai/models/comparisons/glm-5-3-flash-vs-claude-opus-5-5-medium">GLM-5.3-Flash vs Claude Opus 5.5 (medium)</a></li>
    <li><a href="https://artificialanalysis.ai/models/glm-5-3-flash">Artificial Analysis, GLM-5.3-Flash</a></li>
    <li><a href="https://huggingface.co/zai-org/GLM-5.3-Flash">zai-org/GLM-5.3-Flash</a></li>
    <li><a href="https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF">ISTA-DASLab Qwen3.8-Flash-Next GSQ-RCO</a></li>
    <li><a href="https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF">unsloth GLM-5.3-Flash GGUF</a></li>
    <li><a href="https://huggingface.co/pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF">pfeifferj GLM-5.3-Flash GSQ-RCO</a></li>
    <li><a href="https://huggingface.co/orcarouter/GLM-5.3-Flash-MLX">orcarouter GLM-5.3-Flash MLX</a></li>
    <li><a href="https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF">patrickbdevaney GLM-5.3-Flash REAP50</a></li>
    <li><a href="https://huggingface.co/sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit">sh0wie Qwen3.8-Flash-Next REAP-288</a></li>
    <li><a href="https://huggingface.co/Litwein/Qwen3.8-Flash-Next-REAP320-oQ3e-DWQ-MTP-Vision-MLX">Litwein Qwen3.8-Flash-Next REAP320</a></li>
    <li><a href="https://github.com/wheresryan22/anatomy">Anatomy, isometric figure kit</a></li>
  </ol>
</div></footer>

<script type="module">
${kitScript().replace(/<\/(script)/gi, "<\\/$1")}
// figures follow the page theme (explicit choice first, then the system)
const isoNodes = [...document.querySelectorAll(".iso")];
const mq = matchMedia("(prefers-color-scheme: dark)");
const syncTheme = () => { const t = document.documentElement.getAttribute("data-theme"); const dark = t ? t === "dark" : mq.matches; isoNodes.forEach((n) => n.setAttribute("data-theme", dark ? "dark" : "light")); };
syncTheme(); mq.addEventListener("change", syncTheme);
new MutationObserver(syncTheme).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
${scale.live}
${kase.live}
${chest.live}
// calibration strip: a scanner that walks the 45 layers
{
  const host = document.getElementById("layers"), label = document.getElementById("layer-now");
  for (let i = 0; i < 45; i++) { const b = document.createElement("i"); if (i < 3) b.className = "dense"; host.appendChild(b); }
  const bars = [...host.children]; let i = 0, t = 0;
  const show = () => { bars.forEach((b, j) => { b.classList.toggle("now", j === i); b.classList.toggle("done", j < i); }); label.innerHTML = "<b>layer " + i + "</b> of 45"; };
  show();
  if (!matchMedia("(prefers-reduced-motion: reduce)").matches) {
    new IntersectionObserver((e) => { clearInterval(t); if (e[0].isIntersecting) t = setInterval(() => { i = (i + 1) % 46; if (i === 45) i = 0; show(); }, 420); }).observe(host);
  }
}
// chapter rail
{
  const links = [...document.querySelectorAll("#rail a")];
  const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) links.forEach((a) => a.classList.toggle("on", a.getAttribute("href") === "#" + e.target.id)); }), { rootMargin: "-45% 0px -50% 0px" });
  links.forEach((a) => { const s = document.querySelector(a.getAttribute("href")); if (s) io.observe(s); });
}
// copy buttons
document.querySelectorAll("[data-copy]").forEach((b) => b.addEventListener("click", () => {
  const el = document.getElementById(b.dataset.copy);
  const sel = () => { const r = document.createRange(); r.selectNodeContents(el); const s = getSelection(); s.removeAllRanges(); s.addRange(r); };
  try { navigator.clipboard.writeText(el.textContent).then(() => { b.textContent = "Copied"; setTimeout(() => (b.textContent = "Copy"), 1500); }, sel); } catch { sel(); }
}));
</script>
`;
// the page is written as head (title, meta, fonts, styles) + body, for static hosting
const split = page.indexOf('<nav aria-label="Chapters">');
const DESC = "How GLM 5.3 Flash-sauhard fits the 321B GLM-5.3-Flash into 71.6 GB on a 128 GB MacBook Pro M5 Max: calibration, expert pruning, mixed-precision LSQ quantization, and how it compares.";
const doc = `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="description" content="${DESC}"><meta property="og:title" content="GLM 5.3 Flash-sauhard"><meta property="og:description" content="${DESC}"><meta name="twitter:card" content="summary">
${page.slice(0, split)}</head><body>${page.slice(split)}</body></html>`;
writeFileSync(new URL("../public/index.html", import.meta.url), doc);
console.log("wrote public/index.html", doc.length);
