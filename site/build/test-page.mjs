import { writeFileSync } from "node:fs";
import * as k from "../vendor/anatomy/kit/iso-kit.mjs";
import { kitScript } from "../vendor/anatomy/scripts/inline-kit.mjs";
const which = process.argv[2];
const theme = process.argv.includes("--light") ? "light" : "dark";
const m = await import(`./${which}.mjs`);
const extra = `<style>.fmt-row{display:flex;flex-wrap:wrap;gap:6px;padding:10px 14px 14px}.fmt{font:12px ui-monospace;background:transparent;border:1px solid var(--anatomy-faint);color:var(--anatomy-text);border-radius:6px;padding:5px 8px;cursor:pointer}.fmt[aria-pressed=true]{color:var(--anatomy-strong);border-color:var(--anatomy-mid)}.fmt b{margin-left:6px}</style>`;
writeFileSync(`out-${which}-${theme}.html`, k.pageHtml({ title: which, theme, body: extra + `<div id="${m.ROOT || "fig-scale"}">${m.html}</div>`, script: kitScript() + m.live, width: 760 }));
console.log("ok");
