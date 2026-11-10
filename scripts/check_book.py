"""Sanity checks for the book. Run from the repo root:  python3 scripts/check_book.py

Checks, for every page in docs/:
  - every ```python block runs without error
  - every figure referenced as figures/... exists
  - <details> and </details> tags are balanced
  - no stray "$" in prose (a lone $ switches on inline math in KaTeX and GitHub)

Optional deeper checks (need Node): pass --render to also parse every Mermaid
diagram and render every $$ math block, using mermaid + katex installed in a
scratch directory given by --node-modules (default /tmp/mmcheck/node_modules).
"""
import glob
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
failures = 0


def fail(msg):
    global failures
    failures += 1
    print("FAIL:", msg)


pages = sorted(glob.glob(os.path.join(DOCS, "*.md")))
n_code = 0
for path in pages:
    name = os.path.basename(path)
    src = open(path, encoding="utf-8").read()

    for block in re.findall(r"```python\n(.*?)```", src, re.S):
        n_code += 1
        r = subprocess.run([sys.executable, "-W", "ignore", "-c", block],
                           capture_output=True, text=True, timeout=120)
        if r.returncode:
            fail(f"{name}: code block {n_code} errors:\n{r.stderr[-400:]}")

    for img in re.findall(r"\]\((figures/[^)]+)\)", src):
        if not os.path.exists(os.path.join(DOCS, img)):
            fail(f"{name}: missing image {img}")

    if src.count("<details") != src.count("</details>"):
        fail(f"{name}: unbalanced <details> ({src.count('<details')} open, {src.count('</details>')} close)")

    prose = re.sub(r"```.*?```", "", src, flags=re.S)
    prose = re.sub(r"^\s*\$\$\n.*?\n\s*\$\$\s*$", "", prose, flags=re.S | re.M)
    prose = re.sub(r"`[^`]*`", "", prose)
    if "$" in prose:
        i = prose.index("$")
        fail(f"{name}: stray '$' near: {prose[max(0, i - 50):i + 30]!r}")

print(f"{len(pages)} pages, {n_code} code blocks checked")

if "--render" in sys.argv:
    nm = "/tmp/mmcheck/node_modules"
    if "--node-modules" in sys.argv:
        nm = sys.argv[sys.argv.index("--node-modules") + 1]
    js = f"""
import {{ JSDOM }} from "{nm}/jsdom/lib/api.js";
import fs from "fs";
const dom = new JSDOM("<!doctype html><body></body>");
globalThis.window = dom.window; globalThis.document = dom.window.document;
globalThis.DOMPurify = (await import("{nm}/dompurify/dist/purify.es.mjs")).default(dom.window);
const mermaid = (await import("{nm}/mermaid/dist/mermaid.core.mjs")).default;
const katex = (await import("{nm}/katex/dist/katex.mjs")).default;
mermaid.initialize({{ startOnLoad: false }});
let bad = 0, nm_ = 0, nk = 0;
for (const f of process.argv.slice(2)) {{
  const src = fs.readFileSync(f, "utf8");
  for (const m of src.matchAll(/```mermaid\\n([\\s\\S]*?)```/g)) {{
    nm_++; try {{ await mermaid.parse(m[1]); }} catch (e) {{ bad++; console.log("FAIL mermaid", f, String(e.message || e).slice(0, 200)); }}
  }}
  for (const m of src.matchAll(/^\\s*\\$\\$\\n([\\s\\S]*?)\\n\\s*\\$\\$\\s*$/gm)) {{
    nk++; try {{ katex.renderToString(m[1], {{ displayMode: true, throwOnError: true }}); }} catch (e) {{ bad++; console.log("FAIL katex", f, e.message); }}
  }}
}}
console.log(nm_ + " mermaid diagrams, " + nk + " math blocks, " + bad + " failures");
process.exit(bad ? 1 : 0);
"""
    script = os.path.join("/tmp", "check_book_render.mjs")
    open(script, "w").write(js)
    r = subprocess.run(["node", script, *pages])
    if r.returncode:
        failures += 1

print("OK" if not failures else f"{failures} problem(s) found")
sys.exit(1 if failures else 0)
