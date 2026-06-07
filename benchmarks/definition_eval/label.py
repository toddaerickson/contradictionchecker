"""Keyboard-driven labeler for definition-eval candidate pairs.

Mined candidate pairs (``candidates_*.jsonl``, ``label == ""``) need a human
verdict — ``consistent`` or ``divergent`` — before the harness can report real
precision/recall. This serves a tiny local page that shows one pair at a time
side-by-side (with a word-diff highlight) and labels it with a single keypress,
autosaving to ``--out``. Resumable: rows already in ``--out`` come back labelled.

    uv run python -m benchmarks.definition_eval.label \
        --in benchmarks/definition_eval/candidates_atkins.jsonl \
             benchmarks/definition_eval/candidates_fcs.jsonl \
        --out benchmarks/definition_eval/pairs_labeled.jsonl

Then score the detector against what you labelled:

    uv run python -m benchmarks.definition_eval.harness \
        --pairs benchmarks/definition_eval/pairs_labeled.jsonl

Keys:  c = consistent   d = divergent   s = skip   ← / → = move   (relabel anytime)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

_FIELDS = ("pair_id", "category", "term", "def_a", "def_b", "doc_a", "doc_b")
_LABELS = {"consistent", "divergent", ""}


class LabelIn(BaseModel):
    pair_id: str
    label: str


def _load_candidates(paths: list[Path]) -> list[dict[str, Any]]:
    """Load candidate pairs from one or more JSONL files, de-duped by pair_id."""
    seen: dict[str, dict[str, Any]] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pid = row["pair_id"]
            if pid not in seen:
                seen[pid] = {k: row.get(k, "") for k in _FIELDS}
    return list(seen.values())


def _load_labels(out_path: Path) -> dict[str, str]:
    """Read existing labels from the output file (resume support)."""
    labels: dict[str, str] = {}
    if not out_path.exists():
        return labels
    for line in out_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("label"):
            labels[row["pair_id"]] = row["label"]
    return labels


def create_app(candidates: list[dict[str, Any]], out_path: Path) -> FastAPI:
    labels = _load_labels(out_path)
    by_id = {c["pair_id"]: c for c in candidates}

    # _flush() rewrites the whole file from the candidate set, so any label in
    # --out for a pair_id outside that set would be silently destroyed on the
    # first save. Refuse rather than clobber (e.g. --out aimed at the curated
    # pairs.jsonl by mistake — its harness default name is also pairs.jsonl).
    orphaned = sorted(pid for pid in labels if pid not in by_id)
    if orphaned:
        raise ValueError(
            f"--out {out_path} has {len(orphaned)} label(s) for pair_ids not in the "
            f"candidate set (e.g. {orphaned[0]!r}). Point --out at a fresh file so "
            f"existing labels aren't overwritten."
        )

    def _flush() -> None:
        """Atomically rewrite the labelled set (only pairs with a verdict)."""
        lines = [
            json.dumps({**c, "label": labels[c["pair_id"]]}, ensure_ascii=False)
            for c in candidates
            if labels.get(c["pair_id"])
        ]
        tmp = out_path.with_suffix(".tmp")
        tmp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        tmp.replace(out_path)  # atomic on POSIX; no empty-file window mid-write

    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        data = [{**c, "label": labels.get(c["pair_id"], "")} for c in candidates]
        # Escape `</` so corpus text containing "</script>" can't close the
        # inline <script> block and break (or inject into) the page.
        embedded = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
        return HTMLResponse(_PAGE.replace("__DATA__", embedded))

    @app.post("/label")
    def set_label(body: LabelIn) -> JSONResponse:
        if body.label not in _LABELS:
            raise HTTPException(status_code=400, detail=f"bad label {body.label!r}")
        if body.pair_id not in by_id:
            raise HTTPException(status_code=404, detail="unknown pair_id")
        if body.label:
            labels[body.pair_id] = body.label
        else:
            labels.pop(body.pair_id, None)
        _flush()
        done = sum(1 for c in candidates if labels.get(c["pair_id"]))
        return JSONResponse({"ok": True, "labeled": done, "total": len(candidates)})

    return app


_PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>definition-eval labeler</title>
<style>
:root{--bg:#f4f3f0;--fg:#1a1a1a;--muted:#767672;--accent:#c9450a;--blue:#1a52c9;
--border:#e2e0db;--surface:#fff;--diff:#fde8b0}
*{box-sizing:border-box}body{margin:0;font-family:'IBM Plex Sans',system-ui,sans-serif;
background:var(--bg);color:var(--fg)}
header{display:flex;align-items:center;gap:1rem;padding:.6rem 1rem;background:#111;color:#e8e8e4}
header h1{font-size:.95rem;margin:0;font-family:'IBM Plex Mono',monospace}
.bar{flex:1;height:8px;background:#333;border-radius:4px;overflow:hidden}
.bar>span{display:block;height:100%;background:var(--accent);width:0}
.count{font-variant-numeric:tabular-nums;font-size:.85rem}
main{max-width:920px;margin:1.2rem auto;padding:0 1rem}
.term{font-size:1.3rem;font-weight:600;margin:.2rem 0 .1rem}
.meta{color:var(--muted);font-size:.8rem;font-family:'IBM Plex Mono',monospace;margin-bottom:1rem}
.defs{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
.def{background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:.9rem 1rem;
line-height:1.5;font-size:1rem}
.def h3{margin:0 0 .4rem;font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;
color:var(--muted);font-family:'IBM Plex Mono',monospace}
.diff{background:var(--diff);border-radius:2px;padding:0 1px}
.actions{display:flex;gap:.5rem;align-items:center;margin-top:1.2rem;flex-wrap:wrap}
button{font:inherit;border:1px solid var(--border);background:var(--surface);color:var(--fg);
border-radius:5px;padding:.5rem .9rem;cursor:pointer}
button.pos{border-color:var(--accent);color:var(--accent)}
button.pos.on{background:var(--accent);color:#fff}
button.neg{border-color:var(--blue);color:var(--blue)}
button.neg.on{background:var(--blue);color:#fff}
button:hover{filter:brightness(.97)}
kbd{font-family:'IBM Plex Mono',monospace;font-size:.75rem;background:#0001;border:1px solid var(--border);
border-radius:3px;padding:0 4px}
.legend{color:var(--muted);font-size:.8rem;margin-left:auto}
.done{text-align:center;padding:3rem 1rem;color:var(--muted)}
.cur{font-family:'IBM Plex Mono',monospace;font-size:.8rem;color:var(--muted)}
</style></head><body>
<header><h1>definition-eval labeler</h1>
<div class="bar"><span id="prog"></span></div>
<div class="count" id="count">0 / 0</div></header>
<main>
<div id="card"></div>
<div class="actions" id="actions" hidden>
<button class="neg" id="bc" onclick="setLabel('consistent')">Consistent <kbd>c</kbd></button>
<button class="pos" id="bd" onclick="setLabel('divergent')">Divergent <kbd>d</kbd></button>
<button onclick="skip(1)">Skip <kbd>s</kbd></button>
<button onclick="move(-1)"><kbd>&larr;</kbd></button>
<button onclick="move(1)"><kbd>&rarr;</kbd></button>
<span class="legend">divergent = the two definitions genuinely conflict for the same term</span>
</div>
<div class="cur" id="cur"></div>
</main>
<script>
const DATA = __DATA__;
let i = 0;
function esc(s){return (s||'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function tokens(s){return (s||'').split(/(\s+)/);}
function diffHtml(a,b){
  // highlight tokens of `a` whose lowercased form is absent from `b`
  const setB = new Set(tokens(b).map(t=>t.toLowerCase().replace(/[^a-z0-9]/g,'')).filter(Boolean));
  return tokens(a).map(t=>{
    const norm=t.toLowerCase().replace(/[^a-z0-9]/g,'');
    const esc=t.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
    return (norm && !setB.has(norm)) ? '<span class="diff">'+esc+'</span>' : esc;
  }).join('');
}
function labeledCount(){return DATA.filter(d=>d.label).length;}
function render(){
  const total=DATA.length, done=labeledCount();
  document.getElementById('count').textContent=done+' / '+total;
  document.getElementById('prog').style.width=(total?100*done/total:0)+'%';
  if(i>=total){
    document.getElementById('card').innerHTML='<div class="done">All '+total+' pairs reviewed. '
      +done+' labelled. Close this tab and run the harness on the --out file.</div>';
    document.getElementById('actions').hidden=true;
    document.getElementById('cur').textContent='';
    return;
  }
  document.getElementById('actions').hidden=false;
  const d=DATA[i];
  document.getElementById('card').innerHTML=
    '<div class="term">'+esc(d.term||'(no term)')+'</div>'
    +'<div class="meta">'+esc(d.category)+' · '+esc((d.doc_a||'?').slice(0,8))+' vs '+esc((d.doc_b||'?').slice(0,8))+'</div>'
    +'<div class="defs"><div class="def"><h3>Definition A</h3>'+diffHtml(d.def_a,d.def_b)+'</div>'
    +'<div class="def"><h3>Definition B</h3>'+diffHtml(d.def_b,d.def_a)+'</div></div>';
  document.getElementById('bc').classList.toggle('on',d.label==='consistent');
  document.getElementById('bd').classList.toggle('on',d.label==='divergent');
  document.getElementById('cur').textContent='pair '+(i+1)+' of '+total+' — '+(d.label||'unlabelled');
}
async function post(pair_id,label){
  await fetch('/label',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({pair_id,label})});
}
async function setLabel(label){
  const d=DATA[i]; if(!d)return;
  d.label=label; await post(d.pair_id,label);
  move(1);
}
function skip(){move(1);}
function move(step){
  i=Math.max(0,Math.min(DATA.length,i+step)); render();
}
// start at the first unlabelled pair (or the done screen if all are labelled)
i=DATA.findIndex(d=>!d.label); if(i<0)i=DATA.length;
document.addEventListener('keydown',e=>{
  if(e.key==='c')setLabel('consistent');
  else if(e.key==='d')setLabel('divergent');
  else if(e.key==='s')skip();
  else if(e.key==='ArrowLeft')move(-1);
  else if(e.key==='ArrowRight')move(1);
});
render();
</script></body></html>"""


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Label definition-eval candidate pairs.")
    ap.add_argument(
        "--in",
        dest="inputs",
        type=Path,
        nargs="+",
        default=[here / "candidates_atkins.jsonl", here / "candidates_fcs.jsonl"],
        help="Candidate JSONL file(s) to label.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=here / "pairs_labeled.jsonl",
        help="Where labelled pairs are written (harness --pairs target).",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8011)
    args = ap.parse_args()

    inputs = [p for p in args.inputs if p.exists()]
    if not inputs:
        ap.error(f"no candidate files found: {args.inputs}")
    candidates = _load_candidates(inputs)
    app = create_app(candidates, args.out)
    already = len(_load_labels(args.out))
    print(f"Loaded {len(candidates)} candidate pairs ({already} already labelled).")
    print(f"Labelling to {args.out}")
    print(f"Open http://{args.host}:{args.port}  —  keys: c=consistent d=divergent s=skip")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
