// Fig 2: a printer's type case for one MoE layer. Each of the 288 compartments is one expert; the sorts stacked in it
// are as tall as the share of tokens the router sent there during calibration (layer 20, measured). Every token the
// compositor picks 8 sorts into the composing stick, drawn at random by those shares, as the router does.
import { readFileSync } from "node:fs";
import * as k from "../vendor/anatomy/kit/iso-kit.mjs";

export const ROOT = "fig-case";
const DATA = JSON.parse(readFileSync(new URL("./experts.json", import.meta.url)))["20"];
const FREQ = DATA.freq;
const REAP = DATA.kept;
const E = FREQ.length; // 288
const KEEP = REAP.length; // 153
const BUSY = [...FREQ.keys()].sort((a, b) => FREQ[b] - FREQ[a]).slice(0, KEEP).sort((a, b) => a - b);
const MAXF = Math.max(...FREQ);
export const MODES = [
  { id: "all", name: "All 288 experts", keep: [...FREQ.keys()] },
  { id: "reap", name: "REAP 153 (Flash-sauhard)", keep: REAP },
  { id: "busy", name: "Busiest 153", keep: BUSY },
];
const share = (keep) => keep.reduce((s, i) => s + FREQ[i], 0);
MODES.forEach((m) => { m.share = share(m.keep); });

const COLS = 24, ROWS = 12, PITCH = 9, SORT = 6.2;
const CASE = { x: -COLS * PITCH / 2 - 6, y: -ROWS * PITCH / 2 - 6, w: COLS * PITCH + 12, d: ROWS * PITCH + 12, r: 3 };
const CASE_Z = 0, CASE_H = 10, TOP = CASE_Z + CASE_H;
const PLINTH = { x: CASE.x - 8, y: CASE.y - 8, w: CASE.w + 16, d: CASE.d + 40, r: 6 };
const PLINTH_H = 10;
const STICK = { x: -48, y: CASE.y + CASE.d + 20, w: 96, d: 11, r: 1.4 };
const MAXH = 32;
const heightOf = (i, kept) => (kept ? 1.2 + (FREQ[i] / MAXF) * MAXH : 0.7);
const cellOf = (i) => { const c = i % COLS, r = Math.floor(i / COLS); return [CASE.x + 6 + c * PITCH + PITCH / 2, CASE.y + 6 + r * PITCH + PITCH / 2]; };

const WIDTH = 640, HEIGHT = 430;
const P = k.fitProjection(
  [...k.boxCorners(PLINTH, -PLINTH_H - 4, 0), [CASE.x, CASE.y, TOP + MAXH + 2], [CASE.x + CASE.w, CASE.y, TOP + MAXH + 2], [CASE.x, CASE.y + CASE.d, TOP + MAXH + 2]],
  WIDTH, HEIGHT, { pad: 22, azimuth: 52, elevation: 32 },
);
const at = (p) => k.iso(p, P);
const S = (paths, style) => k.solidSvg(paths, style);
const L = (d, style) => k.lineSvg(d, style);
const D = (pts, style) => k.dotsSvg(pts, style);
const ORDER = [...FREQ.keys()].sort((a, b) => k.depthOf([...cellOf(a), 0], P) - k.depthOf([...cellOf(b), 0], P));
const byDepth = (pts) => [...pts].sort((a, b) => k.depthOf([a[0], a[1], 0], P) - k.depthOf([b[0], b[1], 0], P));

function plinth() {
  const out = [`<path class="iso-halo" d="${k.haloOf(PLINTH, -PLINTH_H - 4, PLINTH_H, P)}"/>`];
  for (const [x, y] of byDepth(k.corners(PLINTH, 16))) out.push(S(k.cylinder(x, y, 5, -PLINTH_H - 4, 4, P, 24), { tone: "lo" }));
  out.push(S(k.slabOf(PLINTH, -PLINTH_H, PLINTH_H, P, 6, 1.4), { tone: "mid", inner: L(k.sideSeam(PLINTH, -4, P, 6), { tone: "faint" }) }));
  out.push(L(k.planOutline(k.insetPlan(PLINTH, 4), 0, P, 6), { tone: "lo" }));
  out.push(D(k.corners(PLINTH, 8).map(([x, y]) => at([x, y, 0])), { size: 0.5 }));
  return out.join("");
}

function caseBody() {
  const out = [`<path class="iso-halo" d="${k.haloOf(CASE, CASE_Z, CASE_H, P)}" style="filter:blur(3px)"/>`];
  out.push(S(k.slabOf(CASE, CASE_Z, CASE_H, P, 4, 1), { tone: "mid", inner: L(k.sideSeam(CASE, 4, P, 4), { tone: "faint" }) }));
  // compartment grooves: one closed outline per cell
  const cells = [];
  for (let i = 0; i < E; i++) { const [x, y] = cellOf(i); cells.push(k.planOutline({ x: x - PITCH / 2 + 0.6, y: y - PITCH / 2 + 0.6, w: PITCH - 1.2, d: PITCH - 1.2, r: 0.8 }, TOP, P, 2)); }
  out.push(L(cells.join(""), { tone: "faint" }));
  out.push(L(k.planOutline(k.insetPlan(CASE, 2.4), TOP, P, 4), { tone: "lo" }));
  // brass pull and label holder on the front face (+y)
  const fy = CASE.y + CASE.d;
  out.push(S(k.slabOf({ x: -16, y: fy, w: 32, d: 2.2, r: 0.8 }, 3, 5, P, 2, 0.3), { tone: "mid" }));
  out.push(S(k.slabOf({ x: -12, y: fy + 2.2, w: 24, d: 1.4, r: 0.6 }, 3.6, 3.8, P, 2), { tone: "hi", crease: "lo" }));
  for (const x of [-30, 30]) out.push(S(k.slabOf({ x: x - 3, y: fy, w: 6, d: 3, r: 1 }, 2.5, 3, P, 2), { tone: "mid" }), S(k.slabOf({ x: x - 1.2, y: fy + 3, w: 2.4, d: 5, r: 1 }, 3, 2, P, 2), { tone: "mid" }));
  out.push(S(k.slabOf({ x: -32, y: fy + 7, w: 64, d: 2.4, r: 1.1 }, 3, 2, P, 3), { tone: "mid" }));
  return out.join("");
}

// sorts: one prism per expert, rebuilt when the mode changes
function sortSvg(i, kept, P) {
  const [x, y] = cellOf(i);
  const h = heightOf(i, kept);
  return `<g data-e="${i}">${solidSvg(slabOf({ x: x - SORT / 2, y: y - SORT / 2, w: SORT, d: SORT, r: 0.7 }, TOP, h, P, 2, h > 3 ? 0.3 : 0), { tone: kept ? "hi" : "lo", crease: kept ? "lo" : "none" })}</g>`;
}

function stickBase() {
  const out = [];
  out.push(S(k.slabOf({ x: STICK.x - 4, y: STICK.y - 2, w: STICK.w + 8, d: STICK.d + 4, r: 2 }, 0, 1, P, 3), { tone: "lo", crease: "none" }));
  out.push(S(k.slabOf(STICK, 1, 2, P, 3, 0.4), { tone: "mid" }));
  out.push(S(k.slabOf({ x: STICK.x, y: STICK.y, w: STICK.w, d: 2, r: 0.6 }, 3, 6, P, 2, 0.3), { tone: "mid" }));
  out.push(S(k.slabOf({ x: STICK.x, y: STICK.y, w: 3, d: STICK.d, r: 0.6 }, 3, 6, P, 2, 0.3), { tone: "mid" }));
  out.push(D([at([STICK.x + STICK.w - 4, STICK.y + 1, 9]), at([STICK.x + 8, STICK.y + 1, 9])], { size: 0.4 }));
  return out.join("");
}
function stickSorts(picks, keepSet, P) {
  return picks.map((e, j) => {
    const x = STICK.x + 5 + j * 11;
    const kept = keepSet.has(e);
    return solidSvg(slabOf({ x, y: STICK.y + 2.6, w: 9, d: 7, r: 0.6 }, 3, kept ? 9 : 4, P, 2, 0.3), { tone: kept ? "hi" : "lo", crease: kept ? "lo" : "none", lit: kept });
  }).join("");
}
function stickEnd() {
  // the knee: an adjustable stop on the right end of the stick
  return S(k.slabOf({ x: STICK.x + STICK.w - 3, y: STICK.y, w: 3, d: STICK.d, r: 0.6 }, 3, 7, P, 2, 0.3), { tone: "mid" }) +
    S(k.cylinder(STICK.x + STICK.w + 2, STICK.y + STICK.d / 2, 2.2, 1, 6, P, 16), { tone: "mid" }) +
    L(k.knurl(STICK.x + STICK.w + 2, STICK.y + STICK.d / 2, 2.2, 1, 7, 14, P), { tone: "lo" });
}
function solidSvg(p, s) { return k.solidSvg(p, s); }
function slabOf(...a) { return k.slabOf(...a); }

const start = 1;
const startSet = new Set(MODES[start].keep);
const svg = k.figureSvg({
  width: WIDTH, height: HEIGHT,
  label: "A printer's type case on a plinth, with 288 compartments, one per expert in layer 20 of GLM-5.3-Flash. The sorts in each compartment are as tall as the share of tokens routed to that expert during calibration. A composing stick in front holds the 8 experts picked for the current token.",
  body: [
    plinth(),
    caseBody(),
    `<g data-part="sorts">${ORDER.map((i) => sortSvg(i, startSet.has(i), P)).join("")}</g>`,
    stickBase(),
    `<g data-part="stick">${stickSorts([0, 1, 2, 3, 4, 5, 6, 7].map((j) => REAP[j * 9]), startSet, P)}</g>`,
    stickEnd(),
  ].join(""),
});

function readoutOf(mode, tokens, hits) {
  const m = MODES[mode];
  const live = tokens ? ` · live ${((hits / (tokens * 8)) * 100).toFixed(0)}%` : "";
  return `${m.keep.length} kept · ${(m.share * 100).toFixed(1)}% of routing${live}`;
}

export const html = k.plateHtml({
  fig: "Fig 2",
  title: "One layer's experts",
  hint: "Pick what to keep",
  readout: readoutOf(start, 0, 0),
  keys: [
    { mark: "raised", label: "Sort height: share of tokens routed there (measured)" },
    { mark: "lit", label: "Composing stick: the 8 experts this token used" },
    { mark: "dim", label: "Pruned compartment" },
  ],
  caption: `Layer 20 of 42 MoE layers. REAP ranks experts by how much they contribute when chosen, not by how often. On GLM those scores are almost flat, so it keeps some quiet experts and drops some busy ones: its 153 carry ${(MODES[1].share * 100).toFixed(1)}% of this layer's routing, where the busiest 153 would carry ${(MODES[2].share * 100).toFixed(1)}%. Across all layers the averages are 54% and 68%.`,
  body: `<div class="iso-stage" tabindex="0" role="radiogroup" aria-label="Which experts to keep: arrow keys switch">${svg}</div>
<div class="fmt-row" role="group" aria-label="Which experts to keep">${MODES.map((m, i) => `<button type="button" id="keep-${m.id}" class="fmt" data-i="${i}" aria-pressed="${i === start}"><span>${m.name}</span><b>${(m.share * 100).toFixed(0)}%</b></button>`).join("")}</div>`,
});

export const live = `
{
const root = document.getElementById("${ROOT}");
const P = ${JSON.stringify(P)};
const FREQ = ${JSON.stringify(FREQ)};
const MODES = ${JSON.stringify(MODES.map((m) => ({ name: m.name, keep: m.keep, share: m.share })))};
const ORDER = ${JSON.stringify(ORDER)};
const COLS = ${COLS}, PITCH = ${PITCH}, SORT = ${SORT}, CASE = ${JSON.stringify(CASE)}, TOP = ${TOP}, MAXF = ${MAXF}, MAXH = ${MAXH}, STICK = ${JSON.stringify(STICK)};
const cellOf = ${cellOf.toString()};
const heightOf = ${heightOf.toString()};
const sortSvg = ${sortSvg.toString()};
const stickSorts = ${stickSorts.toString()};
const readoutOf = ${readoutOf.toString()};
const stage = root.querySelector(".iso-stage");
const sorts = root.querySelector('[data-part="sorts"]');
const stick = root.querySelector('[data-part="stick"]');
const readout = root.querySelector("[data-readout]");
const buttons = [...root.querySelectorAll(".fmt")];
const still = matchMedia("(prefers-reduced-motion: reduce)");
const cum = []; FREQ.reduce((s, f, i) => (cum[i] = s + f), 0);
const total = cum[cum.length - 1];
// the router's draw: 8 distinct experts, each picked with probability proportional to its measured share
function route() {
  const picks = new Set();
  while (picks.size < 8) { const r = Math.random() * total; let lo = 0, hi = cum.length - 1; while (lo < hi) { const m = (lo + hi) >> 1; if (cum[m] < r) lo = m + 1; else hi = m; } picks.add(lo); }
  return [...picks];
}
let mode = 1, keepSet = new Set(MODES[mode].keep), tokens = 0, hits = 0, lit = [], visible = false, timer = 0;
function setMode(i) {
  mode = i; keepSet = new Set(MODES[i].keep); tokens = 0; hits = 0;
  buttons.forEach((b, j) => b.setAttribute("aria-pressed", String(j === i)));
  sorts.innerHTML = ORDER.map((e) => sortSvg(e, keepSet.has(e), P)).join("");
  lit = [];
  readout.textContent = readoutOf(mode, 0, 0);
}
function step() {
  lit.forEach((g) => g && g.firstElementChild && g.firstElementChild.removeAttribute("data-lit"));
  const picks = route();
  tokens = Math.min(tokens + 1, 500);
  const h = picks.filter((e) => keepSet.has(e)).length;
  hits = tokens === 500 ? hits * (499 / 500) + h : hits + h;
  lit = picks.map((e) => sorts.querySelector('[data-e="' + e + '"]'));
  lit.forEach((g, j) => { if (g && keepSet.has(picks[j])) g.firstElementChild.setAttribute("data-lit", ""); });
  stick.innerHTML = stickSorts(picks, keepSet, P);
  readout.textContent = readoutOf(mode, tokens, hits);
}
function loop() { clearInterval(timer); timer = 0; if (visible && !still.matches) timer = setInterval(step, 700); }
new IntersectionObserver((e) => { visible = e[e.length - 1].isIntersecting; loop(); }, { rootMargin: "60px 0px" }).observe(stage);
buttons.forEach((b, i) => b.addEventListener("click", () => setMode(i)));
stage.addEventListener("keydown", (e) => {
  if (e.key === "ArrowRight" || e.key === "ArrowDown") { e.preventDefault(); setMode((mode + 1) % MODES.length); }
  if (e.key === "ArrowLeft" || e.key === "ArrowUp") { e.preventDefault(); setMode((mode + MODES.length - 1) % MODES.length); }
  if (e.key === " " || e.key === "Enter") { e.preventDefault(); step(); }
});
step();
}
`;
