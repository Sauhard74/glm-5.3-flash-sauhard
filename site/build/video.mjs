// Writes build/out-video.html: a 59-second, 1920x1080 cut of the journey, built from the three Anatomy figures.
// The page runs on a virtual clock (window.__advance(ms)) so record.mjs can step it frame by frame.
import { readFileSync, writeFileSync } from "node:fs";
import { ISO_CSS } from "../vendor/anatomy/kit/iso-kit.mjs";
import { kitScript } from "../vendor/anatomy/scripts/inline-kit.mjs";
import * as scale from "./fig-scale.mjs";
import * as kase from "./fig-case.mjs";
import * as chest from "./fig-chest.mjs";

export const SECONDS = 59;
const ANSWER = JSON.parse(readFileSync(new URL("./hook-answer.json", import.meta.url))).answer;
const LOGO = readFileSync(new URL("../public/favicon.svg", import.meta.url), "utf8").replace("<svg ", '<svg class="logo" ');

// scenes: [id, start, end] in seconds; a figure layer may show in several windows
const S = { hook: [0, 6.5], problem: [6.5, 15], calib: [15, 22], prune: [22, 30], quant: [30, 37.5], land: [37.5, 44], results: [44, 52.5], cta: [52.5, 59] };
const win = (...ids) => ids.map((i) => S[i].join("-")).join(",");
// actions: click a figure's buttons at a time
const ACTIONS = [
  [0.1, "fmt-bf16"], [8.6, "fmt-fp8"], [10.8, "fmt-q4"],
  [22.1, "keep-all"], [25.6, "keep-reap"],
  [30.2, "bits-3"], [32.4, "bits-4"], [34.2, "bits-6"], [36.0, "bits-8"],
  [37.7, "fmt-prune"], [39.8, "fmt-ours"],
];
const CHAPTERS = [["hook", ""], ["problem", "01 · The problem"], ["calib", "02 · Calibrate"], ["prune", "03 · Prune"], ["quant", "04 · Quantize"], ["land", "05 · The result"], ["results", "06 · Benchmarks"], ["cta", ""]];

const tile = (to, dec, unit, label, at) => `<div class="tile" data-in="${at}"><b><span data-count="${to}" data-dec="${dec}" data-at="${at}">0</span><small>${unit}</small></b><span>${label}</span><i><em data-bar="${Math.min(1, to / 100)}" data-at="${at}"></em></i></div>`;

const html = `<!doctype html><html lang="en"><head><meta charset="utf-8">
<script>
// virtual clock: timers, animation frames, Date and CSS animations all follow __advance
(() => {
  let now = 0, rid = 0, tid = 0;
  const rafs = new Map(), timers = new Map(), started = new WeakMap(), realRAF = requestAnimationFrame.bind(window), D0 = Date.now();
  performance.now = () => now; Date.now = () => D0 + now;
  window.requestAnimationFrame = (cb) => (rafs.set(++rid, cb), rid);
  window.cancelAnimationFrame = (id) => rafs.delete(id);
  window.setTimeout = (cb, ms = 0, ...a) => (timers.set(++tid, { at: now + (+ms || 0), cb, a }), tid);
  window.setInterval = (cb, ms = 0, ...a) => (timers.set(++tid, { at: now + (+ms || 0), cb, a, every: Math.max(+ms || 0, 1) }), tid);
  window.clearTimeout = window.clearInterval = (id) => timers.delete(id);
  window.__advance = (dt) => {
    const end = now + dt;
    for (;;) {
      let next = null;
      for (const e of timers) if (e[1].at <= end && (!next || e[1].at < next[1].at)) next = e;
      if (!next) break;
      const [id, t] = next; now = Math.max(now, t.at);
      if (t.every) t.at += t.every; else timers.delete(id);
      try { typeof t.cb === "function" && t.cb(...t.a); } catch (e) { console.error(e); }
    }
    now = end;
    const cbs = [...rafs.values()]; rafs.clear();
    for (const cb of cbs) try { cb(now); } catch (e) { console.error(e); }
    window.__scene && window.__scene(now / 1000);
    for (const a of document.getAnimations()) { if (!started.has(a)) started.set(a, now); a.pause(); a.currentTime = now - started.get(a); }
  };
  window.__settle = () => new Promise((r) => realRAF(() => realRAF(r)));
})();
matchMedia = ((m) => (q) => (q.includes("reduced-motion") ? { matches: false, addEventListener() {}, removeEventListener() {} } : m(q)))(matchMedia.bind(window));
</script>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Familjen+Grotesk:wght@500;600;700&family=Newsreader:opsz,wght@6..72,400&family=JetBrains+Mono:wght@400;500&display=block">
<style>${ISO_CSS}
*{box-sizing:border-box}
html,body{margin:0;width:1920px;height:1080px;overflow:hidden;background:#0b0b0b}
body{font-family:"Familjen Grotesk",sans-serif;color:#ececec;--accent:#7b9dff;--muted:#7a7a80;--line:#26262a}
.bg{position:absolute;inset:0;background:radial-gradient(90% 90% at 0% 0%,#141b31 0%,#0b0b0b 55%),#0b0b0b}
.grain{position:absolute;inset:0;background-image:radial-gradient(rgba(255,255,255,.035) 1px,transparent 1px);background-size:26px 26px}
.scene{position:absolute;inset:0;opacity:0;will-change:opacity}
.left{position:absolute;left:120px;top:0;bottom:0;width:760px;display:flex;flex-direction:column;justify-content:center;gap:26px}
.kicker{font:500 22px "JetBrains Mono",monospace;letter-spacing:.12em;text-transform:uppercase;color:var(--accent)}
h1{margin:0;font-size:150px;line-height:.9;letter-spacing:-.05em;font-weight:700}
h1 .q{color:#6e6e74;font-weight:600}
h2{margin:0;font-size:92px;line-height:.95;letter-spacing:-.045em;font-weight:700}
h2 em{font-style:normal;color:var(--accent)}
.sub{font:400 34px/1.35 "Newsreader",serif;color:#b9b9be;max-width:720px}
.sub b{color:#fff;font-weight:400}
.big{font:700 200px/.9 "Familjen Grotesk";letter-spacing:-.05em;font-variant-numeric:tabular-nums}
.big small{font-size:.32em;color:var(--muted);margin-left:12px;letter-spacing:-.02em}
.fig{position:absolute;right:60px;top:52%;width:1040px;transform:translateY(-50%)}
.fig .iso-plate-corners,.fig .fmt-row,.fig figcaption,.fig .iso-legend,.fig .fmt-note{display:none}
.fig{filter:brightness(1.55) contrast(1.08)}
.fig .iso-plate{background:transparent;box-shadow:none;border:0}
.fig [data-readout]{font:500 22px "JetBrains Mono",monospace!important;color:#d4d4d8!important}
.hero{position:absolute;inset:0;display:flex;flex-direction:column;justify-content:center;padding-left:120px;gap:34px}
.brand{display:flex;align-items:center;gap:18px;font:500 24px "JetBrains Mono",monospace;letter-spacing:.1em;text-transform:uppercase;color:#a9a9ae}
.brand .logo{width:64px;height:64px}
.strip{display:grid;grid-template-columns:repeat(45,1fr);gap:7px;height:300px;align-items:end;width:100%}
.strip i{display:block;height:24%;background:#232327;border-radius:4px}
.strip i.done{background:#34436e}.strip i.now{height:100%;background:var(--accent);box-shadow:0 0 40px rgba(123,157,255,.5)}
.wide{position:absolute;left:120px;right:120px;top:0;bottom:0;display:flex;flex-direction:column;justify-content:center;gap:40px}
.meta{display:flex;justify-content:space-between;font:500 26px "JetBrains Mono",monospace;color:var(--muted)}
.meta b{color:#fff;font-weight:500}
.mix{display:flex;height:22px;gap:4px;border-radius:6px;overflow:hidden}
.mix i{display:block;height:100%}
.key{display:flex;gap:36px;font:22px "JetBrains Mono",monospace;color:var(--muted)}.key i{display:inline-block;width:16px;height:16px;border-radius:3px;margin-right:10px;vertical-align:-2px}
.tiles{display:grid;grid-template-columns:repeat(5,1fr);gap:24px}
.tile{border:1px solid var(--line);border-radius:22px;padding:30px 28px;background:rgba(255,255,255,.025);display:flex;flex-direction:column;gap:12px}
.tile b{font:700 84px/1 "Familjen Grotesk";letter-spacing:-.04em;font-variant-numeric:tabular-nums}
.tile b small{font-size:.4em;color:var(--muted);margin-left:6px}
.tile>span{font:22px/1.35 "JetBrains Mono",monospace;color:#a9a9ae}
.tile i{display:block;height:8px;background:#1d1d21;border-radius:4px;overflow:hidden}.tile em{display:block;height:100%;width:0;background:var(--accent)}
.vs{display:flex;flex-direction:column;gap:16px}
.row{display:grid;grid-template-columns:420px 1fr 90px;gap:24px;align-items:center;font:26px "JetBrains Mono",monospace;color:#c9c9ce}
.row .t{height:30px;background:#18181b;border-radius:0 6px 6px 0;overflow:hidden}.row .t em{display:block;height:100%;width:0;background:#55555c}
.row.us{color:#fff}.row.us .t em{background:var(--accent)}.row b{text-align:right;font-weight:500}
.url{font:500 40px "JetBrains Mono",monospace;color:var(--accent)}
.links{font:26px "JetBrains Mono",monospace;color:var(--muted)}
.chapter{position:absolute;left:120px;top:64px;font:500 22px "JetBrains Mono",monospace;letter-spacing:.14em;text-transform:uppercase;color:#8a8a90}
.mark{position:absolute;right:80px;top:56px;display:flex;align-items:center;gap:14px;font:500 22px "Familjen Grotesk";color:#a9a9ae}
.mark .logo{width:40px;height:40px}
.progress{position:absolute;left:0;bottom:0;height:6px;background:var(--accent);width:0}
.hook-head{position:absolute;left:120px;top:90px;right:120px;display:flex;flex-direction:column;gap:16px}
.hook-head h2{font-size:104px}
.term{position:absolute;left:120px;top:330px;width:1080px;height:620px;border:1px solid #2a2a30;border-radius:18px;background:#0e0e11;overflow:hidden;box-shadow:0 40px 120px -40px rgba(80,110,255,.35)}
.term .bar{display:flex;align-items:center;gap:9px;padding:14px 18px;background:#161619;border-bottom:1px solid #232327}
.term .bar i{width:13px;height:13px;border-radius:50%;background:#3a3a40}.term .bar span{margin-left:12px;font:18px "JetBrains Mono",monospace;color:#8a8a90}
.term .body{padding:22px 26px;font:21px/1.5 "JetBrains Mono",monospace;font-variant-ligatures:none}
.term .ask{color:var(--accent);margin-bottom:12px}
.term pre{margin:0;white-space:pre-wrap;color:#dcdce0;font:inherit;height:470px;overflow:hidden;display:flex;flex-direction:column;justify-content:flex-end}
.side{position:absolute;left:1260px;right:120px;top:330px;display:flex;flex-direction:column;gap:30px}
.stat{border-top:1px solid #2a2a30;padding-top:18px;display:flex;flex-direction:column;gap:8px}
.stat b{font:700 96px/1 "Familjen Grotesk";letter-spacing:-.04em;font-variant-numeric:tabular-nums}.stat b small{font-size:.34em;color:var(--muted);margin-left:8px}
.stat>span{font:22px "JetBrains Mono",monospace;color:#a9a9ae}
.mem{position:relative;height:14px;background:#1d1d21;border-radius:4px}.mem i{position:absolute;left:0;top:0;bottom:0;background:var(--accent);border-radius:4px}.mem u{position:absolute;top:-6px;bottom:-6px;width:3px;background:#fb923c}
.cap{position:absolute;left:120px;bottom:70px;font:22px "JetBrains Mono",monospace;color:#6e6e74}
</style></head><body>
<div class="bg"></div><div class="grain"></div>
<div class="chapter" id="chapter"></div>
<div class="mark" id="mark">${LOGO}<span>GLM 5.3 Flash-sauhard</span></div>

<section class="scene" data-win="${win("hook")}">
  <div class="hook-head"><h2 data-in="0"><span>A 321B model.</span> <em>On my MacBook.</em></h2><div class="brand" data-in="0.3" style="font-size:22px">No cloud · no API bill · MacBook Pro M5 Max, 128 GB</div></div>
  <div class="term" data-in="0.15"><div class="bar"><i></i><i></i><i></i><span>claude · GLM 5.3 Flash-sauhard · 127.0.0.1:8080</span></div>
    <div class="body"><div class="ask">› Write separate_paren_groups(paren_string) and explain it.</div><pre id="stream"></pre></div></div>
  <div class="side">
    <div class="stat" data-in="0.4"><b><span id="tps">0.0</span><small>tok/s</small></b><span>peak decode, local</span></div>
    <div class="stat" data-in="0.6"><b>71.6<small>GB</small></b><span>of 128 GB unified memory</span><div class="mem"><i style="width:${71.6 / 128 * 100}%"></i><u style="left:${84 / 128 * 100}%"></u></div></div>
    <div class="stat" data-in="0.8"><b>321<small>B&nbsp;params</small></b><span>GLM-5.3-Flash, pruned + re-quantized</span></div>
  </div>
  <p class="cap" data-in="1">Replay of a real answer from the shipped weights, at the measured peak speed.</p>
</section>

<section class="scene fig-layer" data-win="${win("problem", "land", "cta")}"><div class="fig iso" data-theme="dark" id="fig-scale">${scale.html}</div></section>

<section class="scene" data-win="${win("problem")}"><div class="left">
  <h2 data-in="0.3">The model is <em>4×</em> the machine.</h2>
  <div class="big" data-in="0.4"><span data-count-steps="598@0.1,305@2.1,168@4.3" data-dec="0">598</span><small>GB</small></div>
  <p class="sub" data-in="4.6">Even plain 4-bit weighs 168 GB. The Mac has 128, and <b>45 must stay free</b> for the editor, builds and Claude Code.</p>
</div></section>

<section class="scene" data-win="${win("calib")}"><div class="wide">
  <h2 data-in="0.2">First, watch it <em>work</em>.</h2>
  <p class="sub" data-in="0.6" style="max-width:1300px">Every one of 45 layers streamed through <b>3.3 million tokens</b> of real code, agent, math and long-document work, recording which experts fire and which weights matter.</p>
  <div class="strip" id="strip" data-in="0.9"></div>
  <div class="meta" data-in="1.2"><span id="layer-now"><b>layer 0</b> of 45</span><span><b>5.5 hours</b> on the laptop, one layer at a time</span></div>
  <div data-in="1.6"><div class="mix"><i style="flex:378;background:#7b9dff"></i><i style="flex:737;background:#5a74c9"></i><i style="flex:593;background:#3d4f8c"></i><i style="flex:561;background:#2b3660"></i></div>
  <div class="key" style="margin-top:14px"><span><i style="background:#7b9dff"></i>code</span><span><i style="background:#5a74c9"></i>agentic tool use</span><span><i style="background:#3d4f8c"></i>reasoning</span><span><i style="background:#2b3660"></i>long documents</span></div></div>
</div></section>

<section class="scene" data-win="${win("prune")}"><div class="left">
  <h2 data-in="0.3">Keep the experts the work <em>uses</em>.</h2>
  <div class="big" data-in="0.6"><span data-count-steps="288@0.1,153@3.6" data-dec="0">288</span><small>of 288</small></div>
  <p class="sub" data-in="3.8">Each layer has 288 experts. REAP scored them on the calibration run; <b>153 stay</b>, the rest go.</p>
</div><div class="fig iso" data-theme="dark" id="fig-case">${kase.html}</div></section>

<section class="scene" data-win="${win("quant")}"><div class="left">
  <h2 data-in="0.3">A bit width for <em>every</em> tensor.</h2>
  <div class="big" data-in="0.7"><span data-count="3.48" data-dec="2" data-at="0.7" data-dur="2.4">0</span><small>bits avg</small></div>
  <p class="sub" data-in="3.4">Fragile tensors get 6 to 8 bits, robust experts get 3. Then <b>LSQ rounding</b> refits every scale against real activations: 11% closer to the original, same bytes.</p>
</div><div class="fig iso" data-theme="dark" id="fig-chest">${chest.html}</div></section>

<section class="scene" data-win="${win("land")}"><div class="left">
  <h2 data-in="0.3">Under the line.</h2>
  <div class="big" data-in="0.4"><span data-count-steps="92@0.2,71.6@2.3" data-dec="1">92</span><small>GB</small></div>
  <p class="sub" data-in="2.8">Under the 84 GB cap, with <b>44 GB left</b> for everything else. 598 GB down to 71.6.</p>
</div></section>

<section class="scene" data-win="${win("results")}"><div class="wide">
  <h2 data-in="0.2">And it still <em>thinks</em>.</h2>
  <div class="tiles">
    ${tile(92.4, 1, "%", "matches the original's next token", 0.6)}
    ${tile(95, 0, "%", "HumanEval+ pass@1", 1.0)}
    ${tile(97.5, 1, "%", "MATH-500", 1.4)}
    ${tile(84.8, 1, "%", "BFCL function calling", 1.8)}
    ${tile(41.5, 1, "tok/s", "peak decode, on the laptop", 2.2)}
  </div>
  <div class="vs" data-in="3.4">
    <div class="row us"><span>GLM-5.3-Flash (base)</span><span class="t"><em data-bar="${42 / 52}" data-at="3.6"></em></span><b>42</b></div>
    <div class="row"><span>Claude Sonnet 5.5</span><span class="t"><em data-bar="${41 / 52}" data-at="3.8"></em></span><b>41</b></div>
  </div>
  <p class="links" data-in="4.2" style="font-size:22px;color:#9a9aa0">Artificial Analysis index, medium effort. Task scores: 40-problem subsets, measured on GLM 5.3 Flash-sauhard.</p>
</div></section>

<section class="scene" data-win="${win("cta")}"><div class="hero">
  <div class="brand" data-in="0.2">${LOGO}<span>GLM 5.3 Flash-sauhard</span></div>
  <h1 style="font-size:132px"><span data-in="0.4" style="display:block">Run it on</span><span class="q" data-in="0.9" style="display:block">your Mac.</span></h1>
  <div class="url" data-in="1.6">glm-flash-sauhard.vercel.app</div>
  <div class="links" data-in="2.2">Open weights on Hugging Face · code on GitHub · MIT</div>
</div></section>

<div class="progress" id="progress"></div>

<script type="module">${kitScript().replace(/<\/(script)/gi, "<\\/$1")}
${scale.live}
${kase.live}
${chest.live}
{
  const S = ${JSON.stringify(S)}, ACTIONS = ${JSON.stringify(ACTIONS)}, CHAPTERS = ${JSON.stringify(CHAPTERS)}, TOTAL = ${SECONDS};
  const clamp = (x) => Math.max(0, Math.min(1, x)), ease = (x) => 1 - Math.pow(1 - clamp(x), 3);
  const scenes = [...document.querySelectorAll(".scene")].map((el) => ({ el, wins: el.dataset.win.split(",").map((w) => w.split("-").map(Number)) }));
  const strip = document.getElementById("strip"), layerNow = document.getElementById("layer-now");
  for (let i = 0; i < 45; i++) strip.appendChild(document.createElement("i"));
  const bars = [...strip.children];
  const stream = document.getElementById("stream"), ANSWER = ${JSON.stringify(ANSWER)};
  const fmt = (v, d) => v.toFixed(d);
  let fired = 0, lastT = -1;
  window.__scene = (t) => {
    while (fired < ACTIONS.length && ACTIONS[fired][0] <= t) document.getElementById(ACTIONS[fired++][1])?.click();
    for (const s of scenes) {
      let o = 0, local = -1;
      for (const [a, b] of s.wins) if (t >= a - 0.25 && t <= b + 0.25) { o = Math.max(o, Math.min(clamp((t - a + 0.25) / 0.3), clamp((b + 0.25 - t) / 0.3))); if (t >= a - 0.25) local = t - a; }
      if (t > TOTAL - 0.6) o = Math.min(o, clamp((TOTAL - t) / 0.5) + (s.wins.some(([, b]) => b === TOTAL) ? 1 : 0));
      s.el.style.opacity = o;
      s.el.style.visibility = o > 0 ? "visible" : "hidden";
      if (o === 0) continue;
      s.el.querySelectorAll("[data-in]").forEach((n) => { const e = ease((local - +n.dataset.in) / 0.7); n.style.opacity = e; n.style.transform = "translateY(" + ((1 - e) * 36).toFixed(1) + "px)"; });
      s.el.querySelectorAll("[data-count]").forEach((n) => { const e = ease((local - +n.dataset.at) / (+n.dataset.dur || 1.6)); n.textContent = fmt(+n.dataset.count * e, +n.dataset.dec); });
      s.el.querySelectorAll("[data-count-steps]").forEach((n) => {
        const steps = n.dataset.countSteps.split(",").map((x) => x.split("@").map(Number));
        let v = steps[0][0];
        for (let i = 1; i < steps.length; i++) { const [to, at] = steps[i]; const e = ease((local - at) / 1.2); if (e > 0) v = steps[i - 1][0] + (to - steps[i - 1][0]) * e; }
        n.textContent = fmt(v, +n.dataset.dec);
      });
      s.el.querySelectorAll("[data-bar]").forEach((n) => { n.style.width = (+n.dataset.bar * ease((local - +n.dataset.at) / 1.2) * 100).toFixed(2) + "%"; });
    }
    // hook: the answer streams at the measured peak (about 4 characters a token)
    if (t < S.hook[1] + 0.5) {
      const n = Math.floor(Math.max(0, t - 0.5) * 41.5 * 4);
      stream.textContent = ANSWER.slice(0, n) + (Math.floor(t * 2) % 2 ? "▋" : " ");
      document.getElementById("tps").textContent = t < 0.5 ? "0.0" : Math.min(41.5, 41.5 * ease((t - 0.5) / 0.8)).toFixed(1);
    }
    // calibration strip walks the layers
    const c = t - S.calib[0];
    if (c > -0.5 && c < 8.5) { const i = Math.min(44, Math.max(0, Math.floor(c * 7))); bars.forEach((b, j) => { b.className = j === i ? "now" : j < i ? "done" : ""; }); layerNow.innerHTML = "<b>layer " + i + "</b> of 45"; }
    const ch = CHAPTERS.find(([id]) => t >= S[id][0] && t < S[id][1]);
    document.getElementById("chapter").textContent = ch ? ch[1] : "";
    document.getElementById("mark").style.opacity = t < S.problem[0] - 0.2 || t > S.cta[0] - 0.2 ? 0 : 1;
    document.getElementById("progress").style.width = (t / TOTAL * 100).toFixed(3) + "%";
    lastT = t;
  };
  window.__ready = document.fonts.ready.then(() => true);
}
</script>
</body></html>`;

writeFileSync(new URL("./out-video.html", import.meta.url), html);
console.log("wrote out-video.html", html.length);
