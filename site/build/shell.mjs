// Shared head, styles, top bar and footer for the secondary pages (report, run). Same tokens as the journey page.
import { readFileSync } from "node:fs";

export const SITE = "https://glm-flash-sauhard.vercel.app";
export const REPO = "https://github.com/Sauhard74/glm-5.3-flash-sauhard";
export const HF = "Sauhard74/GLM-5.3-Flash-sauhard-MLX";
const LOGO = readFileSync(new URL("../public/favicon.svg", import.meta.url), "utf8").replace("<svg ", '<svg class="logo" aria-hidden="true" ');

const CSS = `
:root{
  --bg:#f2f2ef; --card:#fbfbfa; --ink:#18181a; --text:#46464a; --muted:#8a8a8f; --line:#dcdcd9; --faint:#e8e8e5;
  --accent:#3e63dd; --accent-soft:#e3e9fb; --warn:#c2410c; --good:#30a46c; --open:#9a9aa0;
  --display:"Familjen Grotesk","Helvetica Neue",Arial,sans-serif;
  --body:"Newsreader",Georgia,"Times New Roman",serif;
  --mono:"JetBrains Mono",ui-monospace,Menlo,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#0b0b0b; --card:#131313; --ink:#ececec; --text:#b4b4b4; --muted:#6e6e6e; --line:#232323; --faint:#1a1a1a;
  --accent:#7b9dff; --accent-soft:#16203a; --warn:#fb923c; --good:#6be0a0; --open:#5c5c62; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#0b0b0b; --card:#131313; --ink:#ececec; --text:#b4b4b4; --muted:#6e6e6e; --line:#232323; --faint:#1a1a1a;
  --accent:#7b9dff; --accent-soft:#16203a; --warn:#fb923c; --good:#6be0a0; --open:#5c5c62; color-scheme:dark}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--text);font:18px/1.62 var(--body);-webkit-font-smoothing:antialiased}
a{color:var(--accent);text-underline-offset:3px}
:focus-visible{outline:2px solid var(--accent);outline-offset:3px}
h1,h2,h3{font-family:var(--display);color:var(--ink);text-wrap:balance;margin:0;letter-spacing:-.02em;line-height:1.05}
p{margin:0}
.wrap{max-width:1100px;margin:0 auto;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
.eyebrow{font:500 12px/1.4 var(--mono);letter-spacing:.14em;text-transform:uppercase;color:var(--muted)}
.top{border-bottom:1px solid var(--line);position:sticky;top:0;z-index:10;background:color-mix(in srgb,var(--bg) 88%,transparent);backdrop-filter:blur(10px)}
.top .wrap{display:flex;align-items:center;justify-content:space-between;gap:16px;height:56px}
.brand{display:flex;align-items:center;gap:10px;text-decoration:none;color:var(--ink);font:600 15px var(--display);white-space:nowrap}
.brand .logo{width:26px;height:26px}
.top nav{display:flex;gap:4px}
.top nav a{font:500 13px var(--display);color:var(--muted);text-decoration:none;padding:6px 10px;border-radius:999px;white-space:nowrap}
.top nav a:hover{color:var(--ink)} .top nav a.on{color:var(--ink);background:var(--card);border:1px solid var(--line)}
@media (max-width:560px){.brand span{display:none}}
.hero{padding-block:clamp(48px,8vw,96px) 32px}
.hero h1{font-size:clamp(40px,7vw,84px);font-weight:700;letter-spacing:-.045em;line-height:.95}
.hero h1 .q{color:var(--muted);font-weight:500}
.lede{max-width:64ch;font-size:clamp(18px,2vw,21px);margin-top:22px}
.lede b{color:var(--ink);font-weight:500}
.sec{padding-block:clamp(48px,7vw,88px);border-top:1px solid var(--line)}
.sec-head{display:flex;flex-direction:column;gap:12px;margin-bottom:28px;max-width:70ch}
.sec-head .num{font:500 13px var(--mono);color:var(--accent);letter-spacing:.06em}
.sec-head h2{font-size:clamp(28px,3.6vw,42px);font-weight:650}
.grid2{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}
@media (max-width:860px){.grid2{grid-template-columns:1fr}}
.panel{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:22px;min-width:0}
.panel h3{font-size:18px;font-weight:600;margin-bottom:4px}
.panel .sub{font:12px/1.5 var(--mono);color:var(--muted);margin-bottom:16px}
.tiles{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));border-top:1px solid var(--line);margin-top:40px}
.tiles div{padding:18px 18px 0 0;display:flex;flex-direction:column;gap:3px;min-width:0}
.tiles b{font:600 clamp(30px,4vw,44px)/1 var(--display);color:var(--ink);letter-spacing:-.03em;font-variant-numeric:tabular-nums}
.tiles b small{font-size:.42em;color:var(--muted);margin-left:3px;font-weight:500}
.tiles span{font:12px/1.45 var(--mono);color:var(--muted)}
@media (max-width:760px){.tiles{grid-template-columns:repeat(2,minmax(0,1fr));row-gap:18px}}
.bars{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:8px}
.bar{display:grid;grid-template-columns:minmax(0,220px) minmax(0,1fr) 64px;gap:12px;align-items:center;font:13px var(--mono);color:var(--text)}
.bar .t{height:12px;background:var(--faint);border-radius:0 4px 4px 0;overflow:hidden}
.bar .t i{display:block;height:100%;width:var(--w);background:var(--open);border-radius:0 4px 4px 0}
.bar.on .t i{background:var(--accent)} .bar.on span,.bar.on b{color:var(--ink)}
.bar b{font-weight:500;text-align:right;font-variant-numeric:tabular-nums;color:var(--ink)}
.bar span small{display:block;color:var(--muted);font-size:11px}
@media (max-width:560px){.bar{grid-template-columns:1fr 64px}.bar .t{grid-column:1/-1;order:3}}
.tbl{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px;font-family:var(--mono)}
th,td{padding:9px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{font-weight:500;font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
td.best{color:var(--ink);font-weight:600}
tr.us td{background:color-mix(in srgb,var(--accent-soft) 70%,transparent)} tr.us td:first-child{color:var(--accent)}
td small{color:var(--muted)}
.note{font:13px/1.6 var(--mono);color:var(--muted);margin-top:14px;max-width:90ch}
.prose{max-width:68ch;display:flex;flex-direction:column;gap:14px}
.callout{border-left:3px solid var(--warn);padding:4px 0 4px 16px;font-size:17px}
.callout b{color:var(--ink);font-weight:500}
pre{margin:0;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;overflow-x:auto;font:13px/1.7 var(--mono);color:var(--ink)}
pre .c{color:var(--muted)}
.code{position:relative}.code pre{padding-right:76px}
.copy{position:absolute;top:8px;right:8px;font:500 12px var(--display);border:1px solid var(--line);background:var(--bg);color:var(--ink);padding:4px 10px;border-radius:6px;cursor:pointer}
code{font:.86em var(--mono);background:var(--faint);padding:1px 5px;border-radius:4px;color:var(--ink)}
footer{border-top:1px solid var(--line);padding-block:40px 64px;font:13px/1.6 var(--mono);color:var(--muted)}
footer .wrap{display:flex;flex-wrap:wrap;gap:8px 24px;justify-content:space-between}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important;scroll-behavior:auto!important}}
`;

const COPY_JS = `document.querySelectorAll(".code").forEach((box) => {
  const b = document.createElement("button"); b.className = "copy"; b.type = "button"; b.textContent = "Copy"; box.appendChild(b);
  b.addEventListener("click", () => {
    const text = box.querySelector("pre").innerText.split("\\n").filter((l) => !l.trimStart().startsWith("#")).join("\\n").trim();
    navigator.clipboard.writeText(text).then(() => { b.textContent = "Copied"; setTimeout(() => (b.textContent = "Copy"), 1500); }, () => {});
  });
});`;

const NAV = [["/", "Journey"], ["/report/", "Benchmarks"], ["/run/", "Run it"]];

export function page({ path, title, desc, body, css = "", js = "" }) {
  const nav = NAV.map(([h, t]) => `<a href="${h}"${h === path ? ' class="on" aria-current="page"' : ""}>${t}</a>`).join("");
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>${title}</title>
<meta name="description" content="${desc}"><meta property="og:type" content="website"><meta property="og:url" content="${SITE}${path}"><meta property="og:site_name" content="GLM 5.3 Flash-sauhard"><meta property="og:title" content="${title}"><meta property="og:description" content="${desc}"><meta property="og:image" content="${SITE}/og.png"><meta property="og:image:width" content="1200"><meta property="og:image:height" content="630"><meta name="twitter:card" content="summary_large_image"><meta name="twitter:title" content="${title}"><meta name="twitter:description" content="${desc}"><meta name="twitter:image" content="${SITE}/og.png"><link rel="icon" href="/favicon.ico" sizes="any"><link rel="icon" href="/favicon.svg" type="image/svg+xml"><link rel="apple-touch-icon" href="/apple-touch-icon.png"><meta name="theme-color" content="#0b0b0b">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Familjen+Grotesk:wght@400;500;600;700&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&family=JetBrains+Mono:wght@400;500&display=swap">
<style>${CSS}${css}</style></head><body>
<header class="top"><div class="wrap"><a class="brand" href="/">${LOGO}<span>GLM 5.3 Flash-sauhard</span></a><nav aria-label="Pages">${nav}</nav></div></header>
${body}
<footer><div class="wrap"><span>GLM 5.3 Flash-sauhard · built from zai-org/GLM-5.3-Flash (MIT)</span><span><a href="${REPO}">Code</a> · <a href="https://huggingface.co/${HF}">Weights</a> · <a href="/bench.json">Raw numbers</a></span></div></footer>
<script>${COPY_JS}${js}</script>
</body></html>`;
}
