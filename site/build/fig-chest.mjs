// Fig 3: a card-index chest with one drawer per bit width. Each drawer is as deep as its width (14 units per bit),
// and holds one index card per GB of packed weights stored at that width in GLM 5.3 Flash-sauhard (measured from the
// safetensors headers; scales and biases not counted).
import * as k from "../vendor/anatomy/kit/iso-kit.mjs";

export const ROOT = "fig-chest";
export const DRAWERS = [
  { bits: 8, gb: 0.34, what: "attention and shared-expert tensors the solver found most fragile" },
  { bits: 6, gb: 2.81, what: "attention, embeddings, output head, shared experts" },
  { bits: 5, gb: 1.42, what: "attention projections" },
  { bits: 4, gb: 10.91, what: "routed experts in sensitive layers, plus attention and embeddings" },
  { bits: 3, gb: 53.42, what: "routed experts: 84% of all weights" },
];
const PER_BIT = 14;
const CHEST = { x: -52, y: -112, w: 104, d: 112, r: 2 };
const PLINTH_Z = 6;
const FRONT_H = 16, GAP = 2;
const zOf = (i) => PLINTH_Z + 2 + (DRAWERS.length - 1 - i) * (FRONT_H + GAP); // drawer 0 on top
const TOP_Z = zOf(0) + FRONT_H + 2;
const FT = 2.4; // front thickness
const IN = { x0: -44, x1: 44 };
const CARD = 0.4;

const WIDTH = 560, HEIGHT = 440;
const P = k.fitProjection(
  [...k.boxCorners({ x: CHEST.x - 6, y: CHEST.y - 6, w: CHEST.w + 12, d: CHEST.d + 12 + 110, r: 2 }, -4, TOP_Z + 4)],
  WIDTH, HEIGHT, { pad: 20, azimuth: 30, elevation: 30 },
);
const at = (p) => k.iso(p, P);
const S = (paths, style) => k.solidSvg(paths, style);
const L = (d, style) => k.lineSvg(d, style);
const D = (pts, style) => k.dotsSvg(pts, style);

function body() {
  const out = [];
  const base = { x: CHEST.x - 4, y: CHEST.y - 4, w: CHEST.w + 8, d: CHEST.d + 8, r: 3 };
  out.push(`<path class="iso-halo" d="${k.haloOf(base, -4, TOP_Z, P)}"/>`);
  for (const [x, y] of k.corners(base, 8).sort((a, b) => k.depthOf([...a, 0], P) - k.depthOf([...b, 0], P)))
    out.push(S(k.slabOf({ x: x - 4, y: y - 4, w: 8, d: 8, r: 1.5 }, -4, 4, P, 3), { tone: "lo" }));
  out.push(S(k.slabOf(base, 0, PLINTH_Z, P, 4, 0.8), { tone: "mid" }));
  out.push(S(k.slabOf(CHEST, PLINTH_Z, TOP_Z - PLINTH_Z, P, 3, 0.6), { tone: "mid", inner: L(k.sideSeam(CHEST, PLINTH_Z + 4, P, 3), { tone: "faint" }) }));
  // top cap with a moulded lip
  const cap = { x: CHEST.x - 3, y: CHEST.y - 3, w: CHEST.w + 6, d: CHEST.d + 6, r: 3 };
  out.push(S(k.slabOf(cap, TOP_Z, 3, P, 4, 0.8), { tone: "mid" }));
  out.push(L(k.planOutline(k.insetPlan(cap, 4), TOP_Z + 3, P, 4), { tone: "faint" }));
  out.push(D(k.corners(cap, 7).map(([x, y]) => at([x, y, TOP_Z + 3])), { size: 0.45 }));
  // recess lines between drawer openings on the front face (y = 0), edge to edge
  for (let i = 0; i <= DRAWERS.length; i++) {
    const z = zOf(Math.min(i, DRAWERS.length - 1)) + (i === DRAWERS.length ? -1 : FRONT_H + 1);
    out.push(L(k.segment([CHEST.x, 0, z], [CHEST.x + CHEST.w, 0, z], P), { tone: "faint" }));
  }
  return out.join("");
}

// one drawer pulled out by dy; only the part outside the chest is drawn (the chest hides the rest)
function drawerSvg(i, dy, P) {
  const d = DRAWERS[i], z = zOf(i), out = [];
  const front = dy;
  const lip = { x: -50, y: front - FT, w: 100, d: FT, r: 0.8 };
  if (dy > 0.6) {
    const y0 = 0, y1 = front - FT;
    out.push(solidSvg(slabOf({ x: IN.x0 - 2, y: y0, w: IN.x1 - IN.x0 + 4, d: y1 - y0, r: 0 }, z, 1.2, P, 2), { tone: "lo", crease: "none" }));
    out.push(solidSvg(slabOf({ x: IN.x0 - 2, y: y0, w: 2, d: y1 - y0, r: 0 }, z, FRONT_H - 3, P, 2), { tone: "mid" }));
    // cards from the front backwards, one per GB; the last card is as thick as the remainder
    const n = Math.ceil(d.gb);
    const cards = [];
    for (let j = 0; j < n; j++) {
      const yc = y1 - 2 - (j + 1) * CARD * 1.25;
      if (yc < y0) continue;
      const tabX = IN.x0 + 4 + ((j * 7) % 5) * 15;
      cards.push([yc, tabX, j === n - 1 ? Math.max(0.15, (d.gb - (n - 1))) : 1]);
    }
    cards.sort((a, b) => a[0] - b[0]);
    for (const [yc, tabX, frac] of cards) {
      out.push(solidSvg(slabOf({ x: IN.x0 + 1, y: yc, w: IN.x1 - IN.x0 - 2, d: CARD * frac, r: 0 }, z + 1.2, FRONT_H - 6, P, 2), { tone: "hi", crease: "none" }));
      out.push(solidSvg(slabOf({ x: tabX, y: yc, w: 11, d: CARD * frac, r: 0 }, z + FRONT_H - 4.8, 2, P, 2), { tone: "hi", crease: "none" }));
    }
    out.push(solidSvg(slabOf({ x: IN.x1, y: y0, w: 2, d: y1 - y0, r: 0 }, z, FRONT_H - 3, P, 2), { tone: "mid" }));
  }
  out.push(solidSvg(slabOf(lip, z, FRONT_H, P, 2, 0.5), { tone: "hi", crease: "lo", lit: dy > 0.6 }));
  // label holder and pull on the face
  const fy = front;
  out.push(solidSvg(slabOf({ x: -12, y: fy, w: 24, d: 0.8, r: 0.3 }, z + 8, 5.5, P, 2), { tone: "mid", crease: "none" }));
  out.push(solidSvg(slabOf({ x: -10, y: fy + 0.8, w: 20, d: 0.4, r: 0.2 }, z + 8.8, 3.9, P, 2), { tone: "hi", crease: "none" }));
  for (const x of [-9, 9]) out.push(solidSvg(slabOf({ x: x - 1.2, y: fy, w: 2.4, d: 3.2, r: 0.8 }, z + 3, 2, P, 2), { tone: "mid" }));
  out.push(solidSvg(slabOf({ x: -11, y: fy + 3.2, w: 22, d: 1.8, r: 0.8 }, z + 3, 2, P, 2), { tone: "mid" }));
  return out.join("");
}
function chestSvg(pulls, P) {
  const idx = DRAWERS.map((_, i) => i);
  const closed = idx.filter((i) => pulls[i] <= 0.6).sort((a, b) => zOf(a) - zOf(b));
  const open = idx.filter((i) => pulls[i] > 0.6).sort((a, b) => zOf(a) - zOf(b));
  return [...closed, ...open].map((i) => drawerSvg(i, pulls[i], P)).join("");
}
function solidSvg(p, s) { return k.solidSvg(p, s); }
function slabOf(...a) { return k.slabOf(...a); }

function readoutOf(i) {
  const d = DRAWERS[i];
  return `${d.bits}-bit · ${d.gb} GB · ${d.bits * PER_BIT}-unit drawer`;
}

const start = 4;
const pulls0 = DRAWERS.map((d, i) => (i === start ? d.bits * PER_BIT - 6 : 0));
const svg = k.figureSvg({
  width: WIDTH, height: HEIGHT,
  label: "A card-index chest with five drawers, for 8, 6, 5, 4 and 3 bits per weight. Each drawer is as deep as its bit width, and holds one index card per gigabyte of weights stored at that width: 53 cards in the 3-bit drawer, 11 in the 4-bit drawer, and a few in the others.",
  body: body() + `<g data-part="drawers">${chestSvg(pulls0, P)}</g>`,
});

export const html = k.plateHtml({
  fig: "Fig 3",
  title: "Where the bits went",
  hint: "Open a drawer, or press 1 to 5",
  readout: readoutOf(start),
  keys: [
    { mark: "lit", label: "Open drawer: depth is the bit width" },
    { mark: "raised", label: "One card per GB of packed weights" },
  ],
  caption: "GLM 5.3 Flash-sauhard, by bit width. A solver measured how much error every tensor takes at each width, then chose widths that minimise total error under a 76 GB budget. Almost all of the routed-expert weight landed in 3 bits; attention, embeddings and shared experts got 4 to 8.",
  body: `<div class="iso-stage" tabindex="0" role="radiogroup" aria-label="Bit-width drawers: keys 1 to 5">${svg}</div>
<div class="fmt-row" role="group" aria-label="Open a drawer">${DRAWERS.map((d, i) => `<button type="button" id="bits-${d.bits}" class="fmt" data-i="${i}" aria-pressed="${i === start}"><span>${d.bits}-bit</span><b>${d.gb} GB</b></button>`).join("")}</div>
<p class="fmt-note" data-what>${DRAWERS[start].what}</p>`,
});

export const live = `
{
const root = document.getElementById("${ROOT}");
const P = ${JSON.stringify(P)};
const DRAWERS = ${JSON.stringify(DRAWERS)};
const PER_BIT = ${PER_BIT}, PLINTH_Z = ${PLINTH_Z}, FRONT_H = ${FRONT_H}, GAP = ${GAP}, FT = ${FT}, IN = ${JSON.stringify(IN)}, CARD = ${CARD};
const zOf = ${zOf.toString()};
const drawerSvg = ${drawerSvg.toString()};
const chestSvg = ${chestSvg.toString()};
const readoutOf = ${readoutOf.toString()};
const stage = root.querySelector(".iso-stage");
const group = root.querySelector('[data-part="drawers"]');
const readout = root.querySelector("[data-readout]");
const what = root.querySelector("[data-what]");
const buttons = [...root.querySelectorAll(".fmt")];
const still = matchMedia("(prefers-reduced-motion: reduce)");
let open = ${start}, pulls = ${JSON.stringify(pulls0)}, frame = 0, last = 0, clock = 0, touring = !still.matches, visible = false, idle = 0;
const targetOf = (i) => (i === open ? DRAWERS[i].bits * PER_BIT - 6 : 0);
function pick(i, byUser) {
  open = i;
  buttons.forEach((b, j) => b.setAttribute("aria-pressed", String(j === i)));
  readout.textContent = readoutOf(i);
  what.textContent = DRAWERS[i].what;
  if (byUser) { touring = false; clearTimeout(idle); idle = setTimeout(() => { touring = !still.matches; clock = 0; run(); }, 6000); }
  run();
}
function tick(now) {
  const dt = last ? Math.min((now - last) / 1000, 1 / 30) : 1 / 60;
  last = now;
  if (touring) { clock += dt; const next = (${start} - Math.floor(clock / 3.2) % DRAWERS.length + DRAWERS.length) % DRAWERS.length; if (next !== open) pick(next, false); }
  let moving = false;
  pulls = pulls.map((p, i) => { const t = targetOf(i); const n = still.matches ? t : p + (t - p) * (1 - Math.exp(-dt / 0.3)); if (Math.abs(t - n) < 0.05) return t; moving = true; return n; });
  group.innerHTML = chestSvg(pulls, P);
  frame = visible && (moving || touring) ? requestAnimationFrame(tick) : 0;
}
const run = () => { if (!frame && visible) { last = 0; frame = requestAnimationFrame(tick); } };
new IntersectionObserver((e) => { visible = e[e.length - 1].isIntersecting; if (visible) run(); }, { rootMargin: "60px 0px" }).observe(stage);
buttons.forEach((b, i) => b.addEventListener("click", () => pick(i, true)));
stage.addEventListener("keydown", (e) => {
  if (e.key >= "1" && e.key <= "5") { e.preventDefault(); pick(Number(e.key) - 1, true); }
  if (e.key === "ArrowDown" || e.key === "ArrowRight") { e.preventDefault(); pick((open + 1) % DRAWERS.length, true); }
  if (e.key === "ArrowUp" || e.key === "ArrowLeft") { e.preventDefault(); pick((open + DRAWERS.length - 1) % DRAWERS.length, true); }
});
}
`;
