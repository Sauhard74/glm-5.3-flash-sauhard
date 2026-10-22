// Writes public/report/index.html (every benchmark we ran) and public/run/index.html (the setup recipe).
// Numbers come from build/bench.json (export_bench.py) and build/speed.json (server measurements, measure_speed.py).
import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { HF, page, REPO } from "./shell.mjs";

const B = JSON.parse(readFileSync(new URL("./bench.json", import.meta.url)));
const S = JSON.parse(readFileSync(new URL("./speed.json", import.meta.url)));
const T = B.tasks[B.ship];
const D = B.tasks["glm-reap153-76-lsq-dwqmse"];

const pct = (v, d = 1) => (v * 100).toFixed(d);
// 95% Wilson interval: these are small subsets, so every score carries its range
function wilson(p, n) {
  const z = 1.96, d = 1 + (z * z) / n, c = (p + (z * z) / (2 * n)) / d, h = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / d;
  return `${Math.round((c - h) * 100)}–${Math.round((c + h) * 100)}%`;
}
const bar = (label, sub, v, max, txt, on, tip) =>
  `<li class="bar${on ? " on" : ""}" title="${tip ?? `${label}: ${txt}`}"><span>${label}${sub ? `<small>${sub}</small>` : ""}</span><span class="t"><i style="--w:${Math.min(100, (v / max) * 100)}%"></i></span><b>${txt}</b></li>`;

// ---------- report ----------
const code = T.code.summary, math = T.math.summary, tools = T.tools.summary;
const ship = B.kl.find((k) => k.build === B.ship);
const dec = S.server.map((r) => r.decode_tps);
const decRange = `${Math.min(...dec).toFixed(1)}–${Math.max(...dec).toFixed(1)}`;

const taskRows = [
  ["HumanEval+", "164-problem set, first 40; extra EvalPlus tests", T.code.breakdown["humaneval+"].n, code["humaneval+_pass@1"], D.code.summary["humaneval+_pass@1"]],
  ["MBPP+", "first 40; EvalPlus tests", T.code.breakdown["mbpp+"].n, code["mbpp+_pass@1"], D.code.summary["mbpp+_pass@1"]],
  ["MBPP (base tests)", "same 40, original tests", T.code.breakdown["mbpp+"].n, T.code.breakdown["mbpp+"]["pass@1_base"], D.code.breakdown["mbpp+"]["pass@1_base"]],
  ["MATH-500", "first 40, exact answer match", math.n, math.accuracy, D.math.summary.accuracy],
  ["BFCL function calling", "7 categories × 15", tools.n, tools.accuracy, D.tools.summary.accuracy],
].map(([n, how, k, v, d]) => `<tr${n === "MBPP (base tests)" ? "" : ""}><td>${n}<br><small>${how}</small></td><td class="n">${k}</td><td class="n best">${pct(v)}%</td><td class="n"><small>${wilson(v, k)}</small></td><td class="n">${pct(d)}%</td></tr>`).join("");

const TOOLCAT = { simple: "Simple", multiple: "Pick one of several", parallel: "Parallel calls", parallel_multiple: "Parallel + pick", irrelevance: "Decline when no tool fits", live_simple: "Live, simple", live_multiple: "Live, pick one" };
const toolBars = Object.entries(T.tools.breakdown).map(([k, v]) => bar(TOOLCAT[k], `n=${v.n}`, v.accuracy, 1, `${pct(v.accuracy, 0)}%`, true)).join("");

const klBars = B.kl.map((k) => bar(k.label, `${k.budget_gb} GB budget`, k.assistant_kl, 0.13, k.assistant_kl.toFixed(3), k.build === B.ship,
  `${k.label}: assistant KL ${k.assistant_kl.toFixed(3)}, top-1 ${pct(k.assistant_top1)}%`)).join("");
const klRows = B.kl.map((k) => `<tr${k.build === B.ship ? ' class="us"' : ""}><td>${k.label}</td><td class="n">${k.budget_gb}</td><td class="n">${k.assistant_kl.toFixed(3)}</td><td class="n">${pct(k.assistant_top1)}%</td><td class="n">${k.reasoning_kl.toFixed(3)}</td><td class="n">${k.tool_kl.toFixed(2)}</td><td class="n">${k.user_kl.toFixed(2)}</td></tr>`).join("");
const DOM = { reasoning: "Reasoning / math", code: "Code", agentic: "Agentic, multi-turn tools", longctx: "Long documents, 16K–64K" };
const domRows = Object.entries(B.domains).map(([d, v]) => `<tr><td>${DOM[d]}</td><td class="n">${v.samples}</td><td class="n">${v.tokens.toLocaleString("en-US")}</td><td class="n">${v.median_kl.toFixed(3)}</td><td class="n">${v.mean_kl.toFixed(3)}</td><td class="n">${pct(v.top1_agree)}%</td></tr>`).join("");

const ctxRows = S.harness.map((r) => `<tr><td class="n">${r.ctx.toLocaleString("en-US")}</td><td class="n">${r.prefill_tps.toFixed(0)}</td><td class="n">${r.ttft_s.toFixed(1)} s</td><td class="n">${r.decode_tps.toFixed(1)}</td><td class="n">${r.peak_gb.toFixed(1)} GB</td></tr>`).join("");
const srvRows = [{ ...S.peak, peak: true }, ...S.server].map((r) => `<tr><td class="n">${r.prompt_tokens.toLocaleString("en-US")}</td><td class="n">${r.generated_tokens}</td><td class="n">${r.prefill_tps.toFixed(0)}</td><td class="n best">${r.decode_tps.toFixed(1)}${r.peak ? " <small>peak</small>" : ""}</td></tr>`).join("");

const report = page({
  path: "/report/",
  title: "GLM 5.3 Flash-sauhard · Benchmarks",
  desc: "Every benchmark run on GLM 5.3 Flash-sauhard: fidelity to the original GLM-5.3-Flash, HumanEval+, MBPP+, MATH-500, BFCL function calling, speed and memory on a 128 GB MacBook Pro.",
  css: `.mix{display:flex;height:14px;border-radius:4px;overflow:hidden;gap:2px;margin:6px 0 10px}.mix i{display:block;height:100%}
.key{display:flex;flex-wrap:wrap;gap:4px 14px;font:12px var(--mono);color:var(--muted)}.key i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:6px}`,
  body: `<section class="hero"><div class="wrap">
  <p class="eyebrow">Benchmark report · ${B.ship} · October 2026</p>
  <h1 style="margin-top:18px">What GLM 5.3 Flash-sauhard scores.<br><span class="q">Everything we ran, with the error bars.</span></h1>
  <p class="lede">All numbers on this page were measured on one MacBook Pro M5 Max with 128 GB, on the exact weights people download. Two kinds of test: <b>fidelity</b>, which compares every next-token prediction against the full-precision GLM-5.3-Flash, and <b>tasks</b>, which grade answers to coding, math and tool-calling problems.</p>
  <div class="tiles">
    <div><b>${pct(code["humaneval+_pass@1"])}<small>%</small></b><span>HumanEval+ pass@1</span></div>
    <div><b>${pct(math.accuracy)}<small>%</small></b><span>MATH-500 accuracy</span></div>
    <div><b>${pct(tools.accuracy)}<small>%</small></b><span>BFCL function calling</span></div>
    <div><b>${pct(ship.assistant_top1)}<small>%</small></b><span>next-token agreement with the original</span></div>
  </div>
</div></section>

<section class="sec" id="tasks"><div class="wrap">
  <div class="sec-head"><span class="num">01 · Tasks</span><h2>Coding, math and tool calls</h2>
  <p>Graded by running the code against the tests, matching the final answer, or checking the function call. Thinking on, temperature 1.0, top-p 0.95 (GLM's recommended sampling), one sample per problem. Subsets are small, so read the range, not the point.</p></div>
  <div class="panel tbl"><table>
    <thead><tr><th>Benchmark</th><th class="n">Problems</th><th class="n">Flash-sauhard</th><th class="n">95% range</th><th class="n">DWQ variant</th></tr></thead>
    <tbody>${taskRows}</tbody></table>
    <p class="note">The DWQ variant is the same build after an extra distillation pass. It matched the original slightly better on fidelity but lost 3.75 points on code, so it was not shipped. Average answer length: ${Math.round(T.code.breakdown["humaneval+"].mean_gen_tokens).toLocaleString("en-US")} tokens on HumanEval+, ${Math.round(T.math.breakdown.math500.mean_gen_tokens).toLocaleString("en-US")} on MATH-500; no answer hit the thinking budget.</p>
  </div>
  <div class="grid2" style="margin-top:16px">
    <div class="panel"><h3>Function calling by category</h3><p class="sub">BFCL, 15 cases each, parsed with GLM's native tool format</p><ul class="bars">${toolBars}</ul>
      <p class="note">Weakest: declining when no tool fits, and the "live" user-written cases, where ${pct(T.tools.breakdown.live_multiple.no_call, 0)}% of answers made no call at all.</p></div>
    <div class="panel"><h3>What these numbers are not</h3><p class="sub">Read before quoting</p>
      <div class="prose" style="font-size:16px">
        <p>These are subsets, not full leaderboard runs: 40 problems per dataset and 105 tool cases. A 95% score on 40 problems is anything from about ${wilson(code["humaneval+_pass@1"], 40)}.</p>
        <p>We could not run the original GLM-5.3-Flash on the same harness: it does not fit in 128 GB. That is the job of the fidelity tests below.</p>
        <p>HumanEval and MATH are old and likely seen in training. They show the build still works, not how it ranks against frontier models.</p>
      </div></div>
  </div>
</div></section>

<section class="sec" id="fidelity"><div class="wrap">
  <div class="sec-head"><span class="num">02 · Fidelity</span><h2>How close it stays to the original</h2>
  <p>${B.kl_eval.samples} held-out conversations (${B.kl_eval.tokens.toLocaleString("en-US")} tokens) were run through full-precision GLM-5.3-Flash once, keeping its top-${B.kl_eval.ref_topk} predictions at every position. Each build is scored against that. Lower KL divergence is closer; top-1 is how often both pick the same next token.</p></div>
  <div class="grid2">
    <div class="panel"><h3>Assistant KL, every candidate build</h3><p class="sub">On the tokens the model writes itself. Lower is better.</p><ul class="bars">${klBars}</ul></div>
    <div class="panel"><h3>Why "assistant" tokens</h3><p class="sub">Splitting by who wrote the text</p>
      <div class="prose" style="font-size:16px">
        <p>On positions where the model writes its own answer, GLM 5.3 Flash-sauhard diverges by <b style="color:var(--ink)">${ship.assistant_kl.toFixed(3)}</b> and agrees on the top token ${pct(ship.assistant_top1)}% of the time.</p>
        <p>On text a user or a tool wrote, KL is large (${ship.user_kl.toFixed(1)} and ${ship.tool_kl.toFixed(1)}) for every build, including near-lossless ones: neither model can predict a user's next word. It still moves with pruning, so we track it as a comprehension signal rather than a quality score.</p>
      </div></div>
  </div>
  <div class="panel tbl" style="margin-top:16px"><table>
    <thead><tr><th>Build</th><th class="n">GB</th><th class="n">Assistant KL</th><th class="n">Top-1</th><th class="n">Reasoning KL</th><th class="n">Tool-text KL</th><th class="n">User-text KL</th></tr></thead>
    <tbody>${klRows}</tbody></table>
    <p class="note">LSQ rounding (refitting each quantization scale against calibration data) cut assistant KL by ${pct(1 - ship.assistant_kl / B.kl[0].assistant_kl, 0)}% at the same size. The two DWQ rows are slightly lower on assistant KL but higher on tool and user text, and worse on code.</p>
  </div>
  <div class="panel tbl" style="margin-top:16px"><h3>The shipped build, by workload</h3><p class="sub">All positions. Medians show the typical token; means are pulled up by user and tool text.</p><table>
    <thead><tr><th>Workload</th><th class="n">Samples</th><th class="n">Tokens</th><th class="n">Median KL</th><th class="n">Mean KL</th><th class="n">Top-1</th></tr></thead>
    <tbody>${domRows}</tbody></table></div>
</div></section>

<section class="sec" id="speed"><div class="wrap">
  <div class="sec-head"><span class="num">03 · Speed and memory</span><h2>How fast, and how much room it leaves</h2>
  <p>Measured on the MLX server people run (one request at a time, 2048-token prefill steps), and on the bare model at fixed prompt lengths. Speculative decoding is off: GLM's draft head ships in the weights but has not been verified yet.</p></div>
  <div class="grid2">
    <div class="panel tbl"><h3>Through the server</h3><p class="sub">Real requests to the OpenAI-compatible API</p><table>
      <thead><tr><th class="n">Prompt tokens</th><th class="n">Generated</th><th class="n">Prefill tok/s</th><th class="n">Decode tok/s</th></tr></thead><tbody>${srvRows}</tbody></table></div>
    <div class="panel tbl"><h3>Bare model, by prompt length</h3><p class="sub">${S.harness_decode} tokens decoded after each prompt</p><table>
      <thead><tr><th class="n">Prompt</th><th class="n">Prefill tok/s</th><th class="n">First token</th><th class="n">Decode tok/s</th><th class="n">Peak</th></tr></thead><tbody>${ctxRows}</tbody></table></div>
  </div>
  <div class="panel" style="margin-top:16px"><h3>Memory on a 128 GB Mac</h3><p class="sub">The server is capped at 84 GB; the rest stays with the system and your apps</p>
    <div class="mix" role="img" aria-label="Weights ${B.build.written_gib.main} GB, draft head ${B.build.written_gib.mtp} GB, cache and runtime up to the 84 GB cap, then 44 GB left for apps">
      <i style="flex:${B.build.written_gib.main};background:var(--accent)"></i><i style="flex:${B.build.written_gib.mtp};background:color-mix(in srgb,var(--accent) 50%,var(--card))"></i><i style="flex:${84 - B.build.written_gib.main - B.build.written_gib.mtp};background:var(--open)"></i><i style="flex:44;background:var(--faint)"></i></div>
    <div class="key"><span><i style="background:var(--accent)"></i>Weights ${B.build.written_gib.main} GB</span><span><i style="background:color-mix(in srgb,var(--accent) 50%,var(--card))"></i>Draft head ${B.build.written_gib.mtp} GB</span><span><i style="background:var(--open)"></i>KV cache and runtime, to the 84 GB cap</span><span><i style="background:var(--faint);outline:1px solid var(--line)"></i>44 GB for everything else</span></div>
    <p class="note">${B.build.avg_bits_per_weight} bits per weight on average. Resident size while serving: 71.6 GB.</p>
  </div>
</div></section>

<section class="sec" id="open"><div class="wrap">
  <div class="sec-head"><span class="num">04 · Not measured yet</span><h2>Open items</h2></div>
  <div class="prose">
    <p class="callout"><b>Public leaderboard scores.</b> The original GLM-5.3-Flash scores 42 on the Artificial Analysis index at medium effort, one point above Claude Sonnet 5.5. GLM 5.3 Flash-sauhard has not been run on that suite; from its fidelity we expect 40–41, but that is an estimate.</p>
    <p class="callout"><b>Long context at 64K to 128K.</b> Fidelity was measured on documents up to 64K tokens. Retrieval at 128K has not been tested.</p>
    <p class="callout"><b>Speculative decoding.</b> The draft head is in the weights; until it is verified, all speeds above are plain decoding.</p>
  </div>
  <p class="note">Software: MLX ${B.runtime.mlx}, mlx-lm ${B.runtime["mlx-lm"]}, mlx-vlm ${B.runtime["mlx-vlm"]}, Python ${B.runtime.python}. Raw summaries: <a href="/bench.json">bench.json</a>. Harness code: <a href="${REPO}/tree/main/evals">evals/</a>.</p>
</div></section>`,
});

// ---------- run ----------
const MODEL = "models/glm-5.3-flash-sauhard";
const step = (n, title, body, cmd) => `<li class="step"><div class="sn">${n}</div><div class="sb"><h3>${title}</h3>${body}${cmd ? `<div class="code"><pre>${cmd}</pre></div>` : ""}</div></li>`;

const run = page({
  path: "/run/",
  title: "Run GLM 5.3 Flash-sauhard on your Mac",
  desc: "Set up GLM 5.3 Flash-sauhard on a 128 GB Apple Silicon Mac in five steps: download the 71 GB MLX weights, start the server, and connect Claude Code or any OpenAI-compatible app.",
  css: `.reqs{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1px;background:var(--line);border:1px solid var(--line);border-radius:12px;overflow:hidden;margin-top:36px}
.reqs div{background:var(--card);padding:14px 16px;display:flex;flex-direction:column;gap:2px;min-width:0}
.reqs b{font:600 20px var(--display);color:var(--ink)} .reqs span{font:12px/1.45 var(--mono);color:var(--muted)}
@media (max-width:760px){.reqs{grid-template-columns:repeat(2,minmax(0,1fr))}}
.steps{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:28px;max-width:860px}
.step{display:grid;grid-template-columns:40px minmax(0,1fr);gap:16px}
.sn{width:32px;height:32px;border-radius:50%;border:1px solid var(--accent);color:var(--accent);display:grid;place-items:center;font:600 14px var(--mono)}
.sb{display:flex;flex-direction:column;gap:10px;min-width:0}
.sb h3{font-size:21px;font-weight:600;margin-top:3px}
.sb p{font-size:17px;max-width:68ch}
.apps{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}
@media (max-width:760px){.apps{grid-template-columns:1fr}}
.apps .panel{padding:16px} .apps h3{font-size:16px}
dl{margin:0;display:grid;grid-template-columns:minmax(0,1fr);gap:14px;max-width:860px}
dt{font:600 17px var(--display);color:var(--ink)} dd{margin:4px 0 0;font-size:17px}`,
  body: `<section class="hero"><div class="wrap">
  <p class="eyebrow">Recipe · about 30 minutes, most of it downloading</p>
  <h1 style="margin-top:18px">Run it on your Mac.<br><span class="q">Five steps, no cloud.</span></h1>
  <p class="lede">GLM 5.3 Flash-sauhard is published as ready-to-run MLX weights. Download them, start the server, and point <b>Claude Code</b> or any OpenAI-compatible app at <code>localhost</code>. Nothing leaves your machine.</p>
  <div class="reqs">
    <div><b>Apple Silicon</b><span>M-series Mac; tested on M5 Max</span></div>
    <div><b>128 GB</b><span>unified memory; 96 GB is too tight</span></div>
    <div><b>75 GB</b><span>free disk for the weights</span></div>
    <div><b>macOS 14+</b><span>with Xcode command-line tools</span></div>
  </div>
</div></section>

<section class="sec"><div class="wrap">
  <ol class="steps">
  ${step(1, "Install uv and get the code", `<p>uv manages Python and pins every package to the versions this build was tested with (MLX ${B.runtime.mlx}, mlx-vlm ${B.runtime["mlx-vlm"]}). The repo holds the server, the memory cap and the dashboard.</p>`,
    `curl -LsSf https://astral.sh/uv/install.sh | sh
git clone ${REPO}.git
cd glm-5.3-flash-sauhard
uv sync`)}
  ${step(2, "Download the weights", `<p>71 GB from Hugging Face. The download resumes if it is interrupted; run the same command again.</p>`,
    `uv run hf download ${HF} --local-dir ${MODEL}`)}
  ${step(3, "Start the server", `<p>Serves the OpenAI API (<code>/v1/chat/completions</code>) and the Anthropic API (<code>/v1/messages</code>) on port 8080, capped at 84 GB. Loading takes a minute or two; the vision warning in the log is expected, since this build is text-only.</p>`,
    `<span class="c"># a key so only your apps can use the server</span>
openssl rand -hex 24 > control/api_key
CAP_GB=84 MTP=0 scripts/quickserve.sh ${MODEL}`)}
  ${step(4, "Say hello", `<p>In a second terminal. The <code>model</code> field must be the same path the server was started with.</p>`,
    `curl -s http://127.0.0.1:8080/v1/chat/completions \\
  -H "Authorization: Bearer $(cat control/api_key)" -H "Content-Type: application/json" \\
  -d '{"model":"${MODEL}","messages":[{"role":"user","content":"Write a haiku about unified memory."}],"max_tokens":800}'`)}
  ${step(5, "Connect Claude Code", `<p>Runs Claude Code against the local model, with a 128K context window. Subagents use it too.</p>`,
    `MODEL_ID=${MODEL} scripts/claude-local.sh`)}
  </ol>
</div></section>

<section class="sec"><div class="wrap">
  <div class="sec-head"><span class="num">Other apps</span><h2>Any OpenAI-compatible client works</h2><p>Use these settings in Open WebUI, Cline, Continue, Zed, Jan, LM Studio's client mode or your own code.</p></div>
  <div class="apps">
    <div class="panel"><h3>Base URL</h3><p class="sub" style="margin:0">http://127.0.0.1:8080/v1</p></div>
    <div class="panel"><h3>Model</h3><p class="sub" style="margin:0">${MODEL}</p></div>
    <div class="panel"><h3>API key</h3><p class="sub" style="margin:0">contents of control/api_key</p></div>
  </div>
  <p class="note">Recommended sampling: temperature 1.0, top-p 0.95. The model thinks before it answers, so allow at least 2,000 output tokens.</p>
</div></section>

<section class="sec"><div class="wrap">
  <div class="sec-head"><span class="num">Optional</span><h2>The dashboard, always on</h2><p>Installs a small launchd service with the control panel at <code>http://127.0.0.1:8090</code>: start, stop and restart with one click, live memory and speed, and it comes back after a reboot. It creates the API key for you.</p></div>
  <div class="code" style="max-width:860px"><pre>scripts/install-control.sh             <span class="c"># open http://127.0.0.1:8090, pick GLM 5.3 Flash-sauhard, Start</span>
scripts/install-control.sh --uninstall <span class="c"># remove it</span></pre></div>
  <p class="note">The dashboard's second profile, Qwen3.8-Flash-Next, needs its own download; see <code>control/profiles.json</code>.</p>
</div></section>

<section class="sec"><div class="wrap">
  <div class="sec-head"><span class="num">If something goes wrong</span><h2>Troubleshooting</h2></div>
  <dl>
    <div><dt>The server is killed while loading, or the Mac swaps heavily</dt><dd>Close large apps (browsers with many tabs, Docker, other models) before starting. The model needs about 72 GB resident; macOS kills a process that pushes memory pressure too high.</dd></div>
    <div><dt>"model not found" or the model loads twice</dt><dd>The <code>model</code> field in requests must match the path passed to <code>quickserve.sh</code> exactly.</dd></div>
    <div><dt>Slow first reply</dt><dd>The first request after loading warms the Metal kernels. Long prompts prefill at about ${Math.round(S.harness[0].prefill_tps / 10) * 10} tokens per second, so a 30K-token codebase takes about a minute; repeated prefixes are cached.</dd></div>
    <div><dt>401 from the server</dt><dd>Send the key as <code>Authorization: Bearer …</code> (OpenAI style) or <code>x-api-key</code> (Anthropic style).</dd></div>
  </dl>
</div></section>

<section class="sec"><div class="wrap">
  <div class="sec-head"><span class="num">Advanced</span><h2>Rebuild it from the original</h2><p>Every step that made these weights is in the repo, so you can calibrate on your own workload or pick a different size. You need the full GLM-5.3-Flash checkpoint (about 600 GB of temporary disk) and an evening: calibration took 5.5 hours on the M5 Max and the build about 10 minutes.</p></div>
  <div class="code" style="max-width:860px"><pre><span class="c"># 1. the original checkpoint</span>
uv run hf download zai-org/GLM-5.3-Flash --local-dir models/src-glm-5.3-flash
<span class="c"># 2. calibration set: public code, agent, math and long-document data</span>
uv run python -m calib_data.build --model glm --out artifacts/glm/data
<span class="c"># 3. stream the model layer by layer, recording routing and sensitivity (~5.5 h)</span>
uv run python -m calib.run --model glm --src models/src-glm-5.3-flash --data artifacts/glm/data --out artifacts/glm/calib
<span class="c"># 4. prune to 153 experts, allocate bits to a 76 GB budget, LSQ rounding (~10 min)</span>
uv run python -m build.make --model glm --src models/src-glm-5.3-flash --calib artifacts/glm/calib \\
  --prune 153 --score reap --budget-gb 76 --drop-vision --rounding lsq --out ${MODEL}
<span class="c"># 5. check it against the original</span>
uv run python -m evals.kl --model ${MODEL}</pre></div>
  <p class="note">Change <code>--budget-gb</code> for a smaller or larger build, or edit <code>calib_data/recipes.py</code> to calibrate on a different mix of work.</p>
</div></section>`,
});

mkdirSync(new URL("../public/report/", import.meta.url), { recursive: true });
mkdirSync(new URL("../public/run/", import.meta.url), { recursive: true });
writeFileSync(new URL("../public/report/index.html", import.meta.url), report);
writeFileSync(new URL("../public/run/index.html", import.meta.url), run);
copyFileSync(new URL("./bench.json", import.meta.url), new URL("../public/bench.json", import.meta.url));
console.log("wrote public/report/index.html", report.length, "public/run/index.html", run.length);
