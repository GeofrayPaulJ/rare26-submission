"""
15_review_tool.py -- local, throwaway browser-based review tool. Run once, click
through images, close it. No build step, no JS framework -- Flask serves a handful
of self-contained HTML pages, each with a small inline <script> driving fetch() calls
against a few JSON/image endpoints below.

MODE A (/evc)     -- EVC self-test: the 50 ACHD (cancer) images from 02_evc/, in a
                     fixed-seed shuffled order. Masks are hidden by default (look
                     first); toggle buttons reveal any of the 5 expert masks or a
                     majority-vote (>=3/5) consensus mask, with an opacity slider.
                     Nothing is saved here -- it's a self-test, not a data product.

MODE B (/rare25)  -- the 158 RARE25 neoplasia images from manifests/rare25_canonical.csv,
                     one at a time, arrow-key navigation. Keys 1/2/3 set a visibility
                     rating, click on the image drops a lesion-location marker, a
                     textarea takes free-text notes, checkboxes flag anything unusual.
                     Every change is POSTed immediately to manifests/positive_review.csv
                     (upsert by filepath) -- closing the browser tab loses nothing.
                     On load, the tool jumps straight to the first not-yet-reviewed
                     image. A toggle strip shows 50 same-hospital non-dysplastic images
                     for side-by-side comparison (view-only, not reviewed/saved).

MODE C (/montage) -- a plain viewer for the review_*.png sheets already produced by
                     14_consolidate.py (review_duplicates.png, review_mixed/group_*.png,
                     review_fov_lowfit_*.png), so those can be worked through in the
                     same tool instead of a separate image viewer.

Read-only with respect to 00_source/, 02_evc/, and every existing manifest except the
one new file this tool owns: manifests/positive_review.csv.

USAGE:
    python 15_review_tool.py [--port 5000]
    Then open http://127.0.0.1:5000/ in a browser.
"""

from __future__ import annotations

import argparse
import csv
import io
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, request
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MANIFESTS = ROOT / "manifests"
SRC = ROOT / "00_source"
EVC_DIR = ROOT / "02_evc"
REVIEW_CSV = MANIFESTS / "positive_review.csv"

EVC_SHUFFLE_SEED = 42
N_NEGATIVE_SAMPLE = 50
REVIEW_FIELDS = [
    "filepath", "centre", "group_id", "visibility", "location_x", "location_y", "note",
    "flag_instrument", "flag_glare", "flag_lesion_fills_frame", "flag_redaction_over_tissue",
    "reviewed_at",
]

app = Flask(__name__)


# ============================= safe path helpers =============================

def _safe_join(base: Path, relpath: str) -> Path:
    p = (base / relpath).resolve()
    base_resolved = base.resolve()
    if not str(p).startswith(str(base_resolved)):
        raise FileNotFoundError("path escapes base directory")
    if not p.is_file():
        raise FileNotFoundError(relpath)
    return p


def _serve_image_bytes(data: bytes, mimetype: str) -> Response:
    resp = Response(data, mimetype=mimetype)
    resp.headers["Cache-Control"] = "public, max-age=31536000"
    return resp


# ============================= EVC (Mode A) =============================

def _load_evc_order() -> list[str]:
    stems = sorted(p.stem for p in (EVC_DIR / "images").glob("*_ACHD.png"))
    rng = random.Random(EVC_SHUFFLE_SEED)
    rng.shuffle(stems)
    return stems


EVC_ORDER = _load_evc_order()


def _colorize_mask(bmp_path: Path, color: tuple[int, int, int], alpha: int = 190) -> bytes:
    mask = np.asarray(Image.open(bmp_path).convert("L")) > 0
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = color
    rgba[..., 3] = np.where(mask, alpha, 0).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return buf.getvalue()


def _consensus_mask_bytes(stem: str, color=(255, 200, 0), alpha: int = 190) -> tuple[bytes, float]:
    masks = [np.asarray(Image.open(EVC_DIR / "annotations_bmp" / f"{stem}_exp{e}.bmp").convert("L")) > 0
              for e in range(1, 6)]
    votes = np.sum(masks, axis=0)
    consensus = votes >= 3
    h, w = consensus.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = color
    rgba[..., 3] = np.where(consensus, alpha, 0).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    area_pct = 100.0 * consensus.sum() / consensus.size
    return buf.getvalue(), area_pct


print(f"Pre-rendering EVC mask overlays for {len(EVC_ORDER)} cancer images...")
_EVC_MASK_CACHE: dict[str, bytes] = {}
_EVC_AREA_PCT: dict[str, float] = {}
for _stem in EVC_ORDER:
    for _e in range(1, 6):
        bmp = EVC_DIR / "annotations_bmp" / f"{_stem}_exp{_e}.bmp"
        _EVC_MASK_CACHE[f"{_stem}/{_e}"] = _colorize_mask(bmp, color=(255, 40, 40))
    cons_bytes, cons_pct = _consensus_mask_bytes(_stem)
    _EVC_MASK_CACHE[f"{_stem}/consensus"] = cons_bytes
    _EVC_AREA_PCT[_stem] = cons_pct
print("Done.")


@app.route("/evc/data")
def evc_data():
    items = []
    for stem in EVC_ORDER:
        items.append({
            "stem": stem,
            "img_url": f"/evc_img/{stem}.png",
            "mask_urls": {str(e): f"/evc_mask/{stem}/{e}" for e in range(1, 6)},
            "consensus_url": f"/evc_mask/{stem}/consensus",
            "consensus_area_pct": round(_EVC_AREA_PCT[stem], 3),
        })
    return jsonify(items)


@app.route("/evc_img/<path:relpath>")
def evc_img(relpath):
    p = _safe_join(EVC_DIR / "images", relpath)
    return _serve_image_bytes(p.read_bytes(), "image/png")


@app.route("/evc_mask/<stem>/<which>")
def evc_mask(stem, which):
    key = f"{stem}/{which}"
    if key not in _EVC_MASK_CACHE:
        return Response("not found", status=404)
    return _serve_image_bytes(_EVC_MASK_CACHE[key], "image/png")


# ============================= RARE25 (Mode B) =============================

_CANONICAL = pd.read_csv(MANIFESTS / "rare25_canonical.csv")
_POSITIVES = _CANONICAL[_CANONICAL["class_label"] == "neoplasia"].sort_values("filepath").reset_index(drop=True)


def _read_reviews() -> dict[str, dict]:
    if not REVIEW_CSV.exists():
        return {}
    df = pd.read_csv(REVIEW_CSV)
    return {row["filepath"]: row.to_dict() for _, row in df.iterrows()}


def _write_review_row(row: dict) -> None:
    reviews = _read_reviews()
    reviews[row["filepath"]] = row
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    with open(REVIEW_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        for fp in _POSITIVES["filepath"]:
            if fp in reviews:
                writer.writerow({k: reviews[fp].get(k, "") for k in REVIEW_FIELDS})


@app.route("/rare25/data")
def rare25_data():
    reviews = _read_reviews()
    items = []
    first_unreviewed = None
    for i, row in _POSITIVES.iterrows():
        fp = row["filepath"]
        rv = reviews.get(fp)
        reviewed = rv is not None and str(rv.get("visibility", "")) not in ("", "nan")
        if reviewed is False and first_unreviewed is None:
            first_unreviewed = i
        items.append({
            "index": int(i),
            "filepath": fp,
            "centre": row["centre"],
            "group_id": int(row["group_id"]),
            "img_url": f"/src_img/{fp}",
            "reviewed": bool(reviewed),
            "review": {k: (None if pd.isna(rv.get(k)) else rv.get(k)) for k in REVIEW_FIELDS} if rv is not None else None,
        })
    if first_unreviewed is None:
        first_unreviewed = 0
    return jsonify({"items": items, "first_unreviewed": int(first_unreviewed), "total": len(items),
                     "reviewed_count": sum(1 for it in items if it["reviewed"])})


@app.route("/rare25/negatives")
def rare25_negatives():
    hospital = request.args.get("hospital", "")
    neg = _CANONICAL[(_CANONICAL["class_label"] == "non-dysplastic") & (_CANONICAL["centre"] == hospital)]
    if len(neg) == 0:
        return jsonify([])
    n = min(N_NEGATIVE_SAMPLE, len(neg))
    sample = neg.sample(n=n, random_state=abs(hash(hospital)) % (2 ** 31)).sort_values("filepath")
    return jsonify([
        {"filepath": r["filepath"], "img_url": f"/src_img/{r['filepath']}"}
        for _, r in sample.iterrows()
    ])


@app.route("/src_img/<path:relpath>")
def src_img(relpath):
    p = _safe_join(SRC, relpath)
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    return _serve_image_bytes(p.read_bytes(), mime)


@app.route("/save", methods=["POST"])
def save_review():
    from datetime import datetime, timezone
    payload = request.get_json(force=True)
    if "filepath" not in payload:
        return jsonify({"ok": False, "error": "missing filepath"}), 400
    row = {k: payload.get(k, "") for k in REVIEW_FIELDS}
    row["reviewed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _write_review_row(row)
    reviews = _read_reviews()
    return jsonify({"ok": True, "reviewed_count": len(reviews)})


# ============================= Montage (Mode C) =============================

def _list_montage_files() -> list[str]:
    files = []
    if (MANIFESTS / "review_duplicates.png").exists():
        files.append("review_duplicates.png")
    fov = sorted(p.name for p in MANIFESTS.glob("review_fov_lowfit_*.png"))
    files.extend(fov)
    mixed_dir = MANIFESTS / "review_mixed"
    if mixed_dir.exists():
        def _gid(name: str) -> int:
            m = re.search(r"group_(\d+)\.png$", name)
            return int(m.group(1)) if m else 0
        mixed = sorted((f"review_mixed/{p.name}" for p in mixed_dir.glob("group_*.png")), key=_gid)
        files.extend(mixed)
    return files


@app.route("/montage/data")
def montage_data():
    return jsonify(_list_montage_files())


@app.route("/montage_img/<path:relpath>")
def montage_img(relpath):
    if relpath not in _list_montage_files():
        return Response("not found", status=404)
    p = _safe_join(MANIFESTS, relpath)
    return _serve_image_bytes(p.read_bytes(), "image/png")


# ============================= HTML pages =============================

BASE_STYLE = """
<style>
  * { box-sizing: border-box; }
  body { background:#161616; color:#eee; font-family: -apple-system, Segoe UI, Arial, sans-serif; margin:0; }
  header { padding:10px 16px; background:#1f1f1f; border-bottom:1px solid #333; display:flex; gap:16px; align-items:center; }
  header a { color:#8ab4ff; text-decoration:none; font-size:14px; }
  header a.active { color:#fff; font-weight:bold; }
  .wrap { padding:16px; }
  button { background:#2a2a2a; color:#eee; border:1px solid #444; padding:6px 12px; border-radius:4px; cursor:pointer; font-size:13px; }
  button:hover { background:#3a3a3a; }
  button.active { background:#3a6bd8; border-color:#3a6bd8; }
  input[type=text], textarea { background:#111; color:#eee; border:1px solid #444; border-radius:4px; padding:6px; font-family:inherit; }
  .row { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
  .muted { color:#999; font-size:12px; }
  .imgwrap { position:relative; display:inline-block; max-width:100%; }
  .imgwrap img.base { display:block; max-width:100%; max-height:75vh; }
  .imgwrap img.overlay { position:absolute; top:0; left:0; width:100%; height:100%; pointer-events:none; }
  .marker { position:absolute; width:14px; height:14px; margin:-7px 0 0 -7px; border-radius:50%;
            background:rgba(255,60,60,0.85); border:2px solid #fff; pointer-events:none; }
</style>
"""

INDEX_HTML = f"""<!doctype html><html><head><meta charset="utf-8"><title>RARE25 review tool</title>{BASE_STYLE}</head>
<body><div class="wrap">
<h2>RARE25 / EVC review tool</h2>
<p class="muted">Local, throwaway. Nothing here is served outside this machine.</p>
<ul style="line-height:2.2">
  <li><a href="/evc">Mode A -- EVC self-test (50 cancer images, expert masks)</a></li>
  <li><a href="/rare25">Mode B -- RARE25 positive review (158 neoplasia images, autosaved)</a></li>
  <li><a href="/montage">Mode C -- review_*.png montage viewer</a></li>
</ul>
</div></body></html>"""


EVC_HTML = f"""<!doctype html><html><head><meta charset="utf-8"><title>Mode A -- EVC self-test</title>{BASE_STYLE}</head>
<body>
<header><a href="/">&larr; home</a><span class="active">Mode A: EVC self-test</span>
  <span id="counter" class="muted"></span></header>
<div class="wrap">
  <div class="row" style="margin-bottom:10px;">
    <button id="btnNone">Hide masks</button>
    <button id="btn1">Expert 1</button><button id="btn2">Expert 2</button><button id="btn3">Expert 3</button>
    <button id="btn4">Expert 4</button><button id="btn5">Expert 5</button>
    <button id="btnCons">Consensus (&gt;=3/5)</button>
    <span class="muted">opacity</span>
    <input type="range" id="opacity" min="0" max="100" value="70">
    <span id="areaPct" class="muted"></span>
  </div>
  <div class="imgwrap">
    <img class="base" id="baseImg">
    <img class="overlay" id="overlayImg" style="opacity:0.7; display:none;">
  </div>
  <div class="row" style="margin-top:12px;">
    <button id="prev">&larr; Prev</button>
    <button id="next">Next &rarr;</button>
    <span class="muted">Left/Right arrow keys also work. Masks hide automatically on each new image -- look first, then reveal.</span>
  </div>
</div>
<script>
let items = [], idx = 0, currentOverlay = null;
async function load() {{
  items = await (await fetch('/evc/data')).json();
  render();
}}
function setOverlay(url) {{
  const ov = document.getElementById('overlayImg');
  if (!url) {{ ov.style.display = 'none'; currentOverlay = null; return; }}
  ov.src = url; ov.style.display = 'block'; currentOverlay = url;
}}
function render() {{
  const it = items[idx];
  document.getElementById('baseImg').src = it.img_url;
  document.getElementById('counter').textContent = `${{idx+1}} / ${{items.length}}  (${{it.stem}})`;
  document.getElementById('areaPct').textContent = `consensus area: ${{it.consensus_area_pct}}% of FOV`;
  setOverlay(null);
}}
document.getElementById('btnNone').onclick = () => setOverlay(null);
for (const n of [1,2,3,4,5]) {{
  document.getElementById('btn'+n).onclick = () => setOverlay(items[idx].mask_urls[n]);
}}
document.getElementById('btnCons').onclick = () => setOverlay(items[idx].consensus_url);
document.getElementById('opacity').oninput = (e) => {{
  document.getElementById('overlayImg').style.opacity = e.target.value / 100;
}};
document.getElementById('prev').onclick = () => {{ idx = Math.max(0, idx-1); render(); }};
document.getElementById('next').onclick = () => {{ idx = Math.min(items.length-1, idx+1); render(); }};
document.addEventListener('keydown', (e) => {{
  if (e.key === 'ArrowLeft') {{ idx = Math.max(0, idx-1); render(); }}
  else if (e.key === 'ArrowRight') {{ idx = Math.min(items.length-1, idx+1); render(); }}
}});
load();
</script>
</body></html>"""


RARE25_HTML = f"""<!doctype html><html><head><meta charset="utf-8"><title>Mode B -- RARE25 review</title>{BASE_STYLE}</head>
<body>
<header><a href="/">&larr; home</a><span class="active">Mode B: RARE25 positive review</span>
  <span id="progress" class="muted"></span></header>
<div class="wrap">
  <div class="row" style="margin-bottom:8px;">
    <span id="meta" class="muted"></span>
    <button id="toggleNeg">Show comparison strip (same-hospital negatives)</button>
  </div>
  <div class="imgwrap" id="imgwrap">
    <img class="base" id="baseImg">
    <div id="markerHolder"></div>
  </div>
  <div class="row" style="margin-top:10px;">
    <button id="prev">&larr; Prev</button>
    <button id="next">Next &rarr;</button>
    <span class="muted">Keys: 1=obvious 2=moderate 3=would-have-missed. Click image = lesion marker.</span>
  </div>
  <div class="row" style="margin-top:10px;">
    <button id="visObvious">1 Obvious</button>
    <button id="visModerate">2 Moderate</button>
    <button id="visMissed">3 Would-have-missed</button>
  </div>
  <div class="row" style="margin-top:10px;">
    <label><input type="checkbox" id="flagInstrument"> instrument in frame</label>
    <label><input type="checkbox" id="flagGlare"> heavy glare</label>
    <label><input type="checkbox" id="flagFills"> lesion fills frame</label>
    <label><input type="checkbox" id="flagRedaction"> redaction box over tissue</label>
  </div>
  <div class="row" style="margin-top:10px;">
    <textarea id="note" placeholder="free-text note" rows="3" style="width:600px;"></textarea>
  </div>
  <div id="negStrip" style="display:none; margin-top:16px;">
    <div class="muted" style="margin-bottom:6px;">Same-hospital non-dysplastic comparison (view only, not saved):</div>
    <div id="negThumbs" style="display:flex; gap:6px; overflow-x:auto;"></div>
    <img id="negLarge" style="max-width:100%; max-height:60vh; margin-top:8px; display:none;">
  </div>
</div>
<script>
let items = [], idx = 0, total = 0, reviewedCount = 0;

async function load() {{
  const data = await (await fetch('/rare25/data')).json();
  items = data.items; total = data.total; reviewedCount = data.reviewed_count;
  idx = data.first_unreviewed;
  render();
}}

function currentReview() {{
  const it = items[idx];
  return it.review || {{filepath: it.filepath, centre: it.centre, group_id: it.group_id,
    visibility:'', location_x:'', location_y:'', note:'',
    flag_instrument:false, flag_glare:false, flag_lesion_fills_frame:false, flag_redaction_over_tissue:false}};
}}

function render() {{
  const it = items[idx];
  document.getElementById('baseImg').src = it.img_url;
  document.getElementById('meta').textContent =
    `${{idx+1}}/${{items.length}}  ${{it.filepath}}  hospital=${{it.centre}}  group_id=${{it.group_id}}`;
  document.getElementById('progress').textContent = `${{reviewedCount}}/${{total}} reviewed`;
  const rv = currentReview();
  document.getElementById('note').value = rv.note || '';
  document.getElementById('flagInstrument').checked = !!rv.flag_instrument && rv.flag_instrument !== 'False';
  document.getElementById('flagGlare').checked = !!rv.flag_glare && rv.flag_glare !== 'False';
  document.getElementById('flagFills').checked = !!rv.flag_lesion_fills_frame && rv.flag_lesion_fills_frame !== 'False';
  document.getElementById('flagRedaction').checked = !!rv.flag_redaction_over_tissue && rv.flag_redaction_over_tissue !== 'False';
  setVisButtons(rv.visibility);
  const mh = document.getElementById('markerHolder');
  mh.innerHTML = '';
  if (rv.location_x !== '' && rv.location_x !== null && rv.location_x !== undefined) {{
    drawMarker(rv.location_x, rv.location_y);
  }}
  document.getElementById('negStrip').style.display = 'none';
}}

function setVisButtons(v) {{
  for (const [key, id] of [['obvious','visObvious'], ['moderate','visModerate'], ['would_have_missed','visMissed']]) {{
    document.getElementById(id).classList.toggle('active', v === key);
  }}
}}

function drawMarker(xFrac, yFrac) {{
  const img = document.getElementById('baseImg');
  const mh = document.getElementById('markerHolder');
  mh.innerHTML = '';
  const m = document.createElement('div');
  m.className = 'marker';
  m.style.left = (xFrac * img.clientWidth) + 'px';
  m.style.top = (yFrac * img.clientHeight) + 'px';
  mh.appendChild(m);
}}

let pending = {{}};
function stagedField(key, val) {{ pending[key] = val; }}

async function save() {{
  const it = items[idx];
  const rv = currentReview();
  const payload = {{
    filepath: it.filepath, centre: it.centre, group_id: it.group_id,
    visibility: pending.visibility !== undefined ? pending.visibility : (rv.visibility || ''),
    location_x: pending.location_x !== undefined ? pending.location_x : (rv.location_x ?? ''),
    location_y: pending.location_y !== undefined ? pending.location_y : (rv.location_y ?? ''),
    note: document.getElementById('note').value,
    flag_instrument: document.getElementById('flagInstrument').checked,
    flag_glare: document.getElementById('flagGlare').checked,
    flag_lesion_fills_frame: document.getElementById('flagFills').checked,
    flag_redaction_over_tissue: document.getElementById('flagRedaction').checked,
  }};
  const res = await fetch('/save', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body: JSON.stringify(payload)}});
  const j = await res.json();
  reviewedCount = j.reviewed_count;
  items[idx].review = payload;
  items[idx].reviewed = true;
  pending = {{}};
  document.getElementById('progress').textContent = `${{reviewedCount}}/${{total}} reviewed`;
}}

function goPrev() {{ save(); idx = Math.max(0, idx-1); render(); }}
function goNext() {{ save(); idx = Math.min(items.length-1, idx+1); render(); }}

document.getElementById('prev').onclick = goPrev;
document.getElementById('next').onclick = goNext;
document.getElementById('visObvious').onclick = () => {{ stagedField('visibility','obvious'); setVisButtons('obvious'); save(); }};
document.getElementById('visModerate').onclick = () => {{ stagedField('visibility','moderate'); setVisButtons('moderate'); save(); }};
document.getElementById('visMissed').onclick = () => {{ stagedField('visibility','would_have_missed'); setVisButtons('would_have_missed'); save(); }};

for (const id of ['flagInstrument','flagGlare','flagFills','flagRedaction']) {{
  document.getElementById(id).onchange = save;
}}
let noteTimer = null;
document.getElementById('note').oninput = () => {{
  clearTimeout(noteTimer);
  noteTimer = setTimeout(save, 800);
}};
document.getElementById('note').onblur = save;

document.getElementById('baseImg').addEventListener('click', (e) => {{
  const img = e.target;
  const rect = img.getBoundingClientRect();
  const x = (e.clientX - rect.left) / rect.width;
  const y = (e.clientY - rect.top) / rect.height;
  stagedField('location_x', x); stagedField('location_y', y);
  drawMarker(x, y);
  save();
}});

document.addEventListener('keydown', (e) => {{
  if (document.activeElement && document.activeElement.tagName === 'TEXTAREA') return;
  if (e.key === 'ArrowLeft') goPrev();
  else if (e.key === 'ArrowRight') goNext();
  else if (e.key === '1') {{ stagedField('visibility','obvious'); setVisButtons('obvious'); save(); }}
  else if (e.key === '2') {{ stagedField('visibility','moderate'); setVisButtons('moderate'); save(); }}
  else if (e.key === '3') {{ stagedField('visibility','would_have_missed'); setVisButtons('would_have_missed'); save(); }}
}});

document.getElementById('toggleNeg').onclick = async () => {{
  const strip = document.getElementById('negStrip');
  if (strip.style.display === 'block') {{ strip.style.display = 'none'; return; }}
  const negs = await (await fetch('/rare25/negatives?hospital=' + items[idx].centre)).json();
  const holder = document.getElementById('negThumbs');
  holder.innerHTML = '';
  for (const n of negs) {{
    const t = document.createElement('img');
    t.src = n.img_url; t.style.height = '80px'; t.style.cursor = 'pointer'; t.title = n.filepath;
    t.onclick = () => {{ const lg = document.getElementById('negLarge'); lg.src = n.img_url; lg.style.display = 'block'; }};
    holder.appendChild(t);
  }}
  strip.style.display = 'block';
}};

load();
</script>
</body></html>"""


MONTAGE_HTML = f"""<!doctype html><html><head><meta charset="utf-8"><title>Mode C -- montage viewer</title>{BASE_STYLE}</head>
<body>
<header><a href="/">&larr; home</a><span class="active">Mode C: montage viewer</span>
  <span id="counter" class="muted"></span></header>
<div class="wrap">
  <div class="row" style="margin-bottom:10px;">
    <button id="prev">&larr; Prev</button>
    <button id="next">Next &rarr;</button>
    <span id="fname" class="muted"></span>
  </div>
  <img id="big" style="max-width:100%; border:1px solid #333;">
</div>
<script>
let files = [], idx = 0;
async function load() {{
  files = await (await fetch('/montage/data')).json();
  render();
}}
function render() {{
  if (files.length === 0) {{ document.getElementById('fname').textContent = 'No review_*.png files found in manifests/.'; return; }}
  document.getElementById('big').src = '/montage_img/' + files[idx];
  document.getElementById('fname').textContent = files[idx];
  document.getElementById('counter').textContent = `${{idx+1}} / ${{files.length}}`;
}}
document.getElementById('prev').onclick = () => {{ idx = Math.max(0, idx-1); render(); }};
document.getElementById('next').onclick = () => {{ idx = Math.min(files.length-1, idx+1); render(); }};
document.addEventListener('keydown', (e) => {{
  if (e.key === 'ArrowLeft') {{ idx = Math.max(0, idx-1); render(); }}
  else if (e.key === 'ArrowRight') {{ idx = Math.min(files.length-1, idx+1); render(); }}
}});
load();
</script>
</body></html>"""


@app.route("/")
def index():
    return INDEX_HTML


@app.route("/evc")
def evc_page():
    return EVC_HTML


@app.route("/rare25")
def rare25_page():
    return RARE25_HTML


@app.route("/montage")
def montage_page():
    return MONTAGE_HTML


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()
    print(f"\nOpen http://127.0.0.1:{args.port}/ in a browser.")
    print(f"Positive reviews autosave to {REVIEW_CSV}")
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
