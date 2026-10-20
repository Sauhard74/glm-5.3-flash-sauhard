"""Offline synthetic rows with the same schema as each real source (for --smoke and tests)."""

import json
import random

WORDS = ("the model layer token expert router cache memory budget signal paper result method value system "
         "graph kernel buffer stream request server client update record index query vector matrix proof "
         "theorem river captain village letter night garden window station engine harbor winter").split()


def words(rng, n):
    return " ".join(rng.choice(WORDS) for _ in range(n))


def prose(rng, n_sent):
    return " ".join(words(rng, rng.randint(8, 20)).capitalize() + "." for _ in range(n_sent))


def py_code(rng, n_funcs):
    out = ["import os", "import sys", ""]
    for i in range(n_funcs):
        name = f"{rng.choice(WORDS)}_{rng.choice(WORDS)}_{i}"
        out += [f"def {name}(x, y=None):", f'    """{words(rng, 8)}."""', "    total = 0",
                "    for i in range(x):", f"        total += i * {rng.randint(2, 9)}",
                "    if y is not None:", "        total -= len(y)", "    return total", ""]
    return "\n".join(out)


def go_code(rng, n_funcs):
    out = ["package main", "", 'import "fmt"', ""]
    for i in range(n_funcs):
        name = f"{rng.choice(WORDS).capitalize()}{rng.choice(WORDS).capitalize()}{i}"
        out += [f"func {name}(n int) int {{", "\tsum := 0", "\tfor i := 0; i < n; i++ {",
                f"\t\tsum += i * {rng.randint(2, 9)}", "\t}", '\tfmt.Println("done")', "\treturn sum", "}", ""]
    return "\n".join(out)


CODERS = {"Python": py_code, "Go": go_code}


def github_code(rng, n=60):
    rows = []
    for i in range(n):
        lang = rng.choice(list(CODERS))
        rows.append({"code": CODERS[lang](rng, rng.randint(6, 30)), "repo_name": f"user{i % 7}/proj{i % 5}",
                     "path": f"src/mod_{i}.{'py' if lang == 'Python' else 'go'}", "language": lang})
    return rows


def commitpack(rng, n=40):
    rows = []
    for i in range(n):
        lang = rng.choice(list(CODERS))
        old = CODERS[lang](rng, rng.randint(6, 14))
        new = old.replace("total = 0", "total = 1", 1).replace("sum := 0", "sum := 1", 1) + "\n# updated\n"
        ext = "py" if lang == "Python" else "go"
        rows.append({"commit": f"{i:040x}", "old_file": f"pkg/f{i}.{ext}", "new_file": f"pkg/f{i}.{ext}",
                     "old_contents": old, "new_contents": new, "subject": f"Fix {words(rng, 4)}",
                     "message": f"Fix {words(rng, 4)}\n\n{prose(rng, 2)}", "lang": lang, "license": "mit",
                     "repos": f"user{i}/repo{i},fork{i}/repo{i}"})
    return rows


OH_SYSTEM = """You are a helpful assistant that can interact with a computer to solve tasks.

You have access to the following functions:

---- BEGIN FUNCTION #1: execute_bash ----
Description: Execute a bash command in the terminal.
Parameters:
  (1) command (string, required): The bash command to execute.
---- END FUNCTION #1 ----

---- BEGIN FUNCTION #2: str_replace_editor ----
Description: Custom editing tool for viewing, creating and editing files
Parameters:
  (1) command (string, required): The commands to run.
Allowed values: [`view`, `create`, `str_replace`]
  (2) path (string, required): Absolute path to file or directory.
  (3) view_range (array, optional): Line range to show.
---- END FUNCTION #2 ----

If you choose to call a function ONLY reply in the following format with NO suffix:
<function=example_function_name>
<parameter=example_parameter_1>value_1</parameter>
</function>"""


def openhands(rng, n=12):
    rows = []
    for i in range(n):
        msgs = [{"role": "system", "content": OH_SYSTEM},
                {"role": "user", "content": f"<uploaded_files>\n/workspace/repo{i}\n</uploaded_files>\n{prose(rng, 4)}"}]
        for k in range(rng.randint(2, 4)):
            msgs.append({"role": "assistant", "content": f"{prose(rng, 1)}\n<function=execute_bash>\n"
                         f"<parameter=command>grep -rn {rng.choice(WORDS)} /workspace/repo{i}</parameter>\n</function>"})
            msgs.append({"role": "user", "content": f"EXECUTION RESULT of [execute_bash]:\n{prose(rng, 3)}"})
        msgs.append({"role": "assistant", "content": "<function=str_replace_editor>\n<parameter=command>view</parameter>\n"
                     f"<parameter=path>/workspace/repo{i}/a.py</parameter>\n<parameter=view_range>[1, 20]</parameter>\n</function>"})
        msgs.append({"role": "user", "content": f"EXECUTION RESULT of [str_replace_editor]:\n{py_code(rng, 2)}"})
        msgs.append({"role": "assistant", "content": prose(rng, 3)})
        rows.append({"messages": msgs})
    return rows


def toucan(rng, subset, n=12):
    rows = []
    for i in range(n):
        name = f"svc-{rng.choice(WORDS)}-{i}"
        tools = [{"type": "function", "function": {"name": name, "description": prose(rng, 1), "parameters": {
            "type": "object", "properties": {"query": {"type": "string", "description": words(rng, 4)},
                                             "limit": {"type": "integer"}}, "required": ["query"]}}}]
        msgs = [{"role": "system", "content": "<|im_system|>tool_declare<|im_middle|>" + json.dumps(tools)}]
        for t in range(2 if subset == "multi-turn" else 1):
            q = words(rng, 3)
            msgs += [{"role": "user", "content": prose(rng, 2)},
                     {"role": "assistant", "content": prose(rng, 1)},
                     {"role": "assistant", "content": "", "function_call": {
                         "name": name, "arguments": json.dumps({"query": q, "limit": rng.randint(1, 9)})}},
                     {"role": "function", "name": name, "content": json.dumps({"results": [words(rng, 5)]})},
                     {"role": "assistant", "content": prose(rng, 3)}]
        rows.append({"uuid": f"{subset}-{i}", "subset_name": subset, "messages": json.dumps(msgs),
                     "available_tools": json.dumps(tools)})
    return rows


def openr1(rng, n=12):
    return [{"uuid": f"m{i}", "problem": f"Compute {rng.randint(2, 99)} * {rng.randint(2, 99)}. {prose(rng, 1)}",
             "messages": [{"role": "user", "content": "q"},
                          {"role": "assistant", "content": f"<think>\n{prose(rng, 12)}\n</think>\n\n{prose(rng, 2)} \\boxed{{{i}}}"}],
             "correctness_math_verify": [True]} for i in range(n)]


def openthoughts(rng, domain, n=12):
    return [{"problem": prose(rng, 3), "deepseek_reasoning": prose(rng, 14), "deepseek_solution": prose(rng, 3),
             "domain": domain} for _ in range(n)]


def narrativeqa(rng, n_docs=8, n_q=3):
    rows = []
    for d in range(n_docs):
        text = "\n\n".join(prose(rng, 6) for _ in range(rng.randint(45, 90)))
        for q in range(n_q):
            rows.append({"document": {"id": f"doc{d}", "kind": "gutenberg", "text": text},
                         "question": {"text": f"Who is the {rng.choice(WORDS)}?"},
                         "answers": [{"text": words(rng, 4)}, {"text": words(rng, 3)}]})
    return rows


def arxiv(rng, n=24):
    return [{"article": "\n".join(prose(rng, 8) for _ in range(rng.randint(5, 12))) + " @xmath0 [@xcite].",
             "abstract": prose(rng, 5)} for _ in range(n)]


BUILDERS = {
    "github_code": github_code,
    "commitpack": commitpack,
    "openhands": openhands,
    "toucan_multi": lambda rng: toucan(rng, "multi-turn"),
    "toucan_single": lambda rng: toucan(rng, "single-turn-original"),
    "openr1_math": openr1,
    "openthoughts_code": lambda rng: openthoughts(rng, "code"),
    "openthoughts_science": lambda rng: openthoughts(rng, "physics"),
    "narrativeqa": narrativeqa,
    "arxiv": arxiv,
}


def fixture_rows(name):
    rows = BUILDERS[name](random.Random(f"fixture:{name}"))
    for i, r in enumerate(rows):
        r["_src"] = f"fixture:{name}:{i}"
    return rows
