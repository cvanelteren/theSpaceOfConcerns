#!/usr/bin/env python3
"""Build a blind human audit for the held-meeting output-title classifier.

The audit samples formal outputs, not paper--output lineage links.  Model
predictions are written to a separate key and are never embedded in the coding
page.  The page is self-contained and stores each coder's work in the browser.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS = (
    ROOT
    / "output"
    / "category_treatment_comparison"
    / "fractional_multilabel"
    / "outcome_topic_predictions_title_model.csv"
)
DEFAULT_PROBABILITIES = DEFAULT_PREDICTIONS.with_name(
    "outcome_topic_probabilities_title_model.csv"
)
DEFAULT_OUTDIR = (
    ROOT / "output" / "category_treatment_comparison" / "human_output_audit"
)
SEED = 20260814
SAMPLE_SIZE = 120
EXPECTED_OUTPUTS = 584
EXPECTED_TOPICS = 45
TIME_BANDS = (
    (1995, 2004, "1995--2004"),
    (2005, 2014, "2005--2014"),
    (2015, 2025, "2015--2025"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def time_band(year: int) -> str:
    for lower, upper, label in TIME_BANDS:
        if lower <= int(year) <= upper:
            return label
    raise ValueError(f"Year {year} falls outside the audit period")


def allocate_sample(frame: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    """Sample crossed strata with baseline coverage and proportional remainder."""
    if sample_size > len(frame):
        raise ValueError(f"Cannot sample {sample_size} rows from {len(frame)} outputs")
    group_columns = ["instrument", "time_band", "confidence_band"]
    groups = {
        tuple(key): value.copy()
        for key, value in frame.groupby(group_columns, sort=True, observed=True)
    }
    allocations = {key: min(2, len(value)) for key, value in groups.items()}
    allocated = sum(allocations.values())
    if allocated > sample_size:
        allocations = {key: 0 for key in groups}
        allocated = 0

    remaining = sample_size - allocated
    while remaining:
        capacities = {
            key: len(groups[key]) - allocations[key]
            for key in groups
            if len(groups[key]) > allocations[key]
        }
        if not capacities:
            raise ValueError("Stratified allocation ran out of eligible outputs")
        total_capacity = sum(capacities.values())
        quotas = {
            key: remaining * capacity / total_capacity
            for key, capacity in capacities.items()
        }
        floors = {
            key: min(capacities[key], int(np.floor(quota)))
            for key, quota in quotas.items()
        }
        floor_total = sum(floors.values())
        if floor_total:
            for key, amount in floors.items():
                allocations[key] += amount
            remaining -= floor_total
            continue
        order = sorted(
            capacities,
            key=lambda key: (-(quotas[key] - np.floor(quotas[key])), key),
        )
        for key in order[:remaining]:
            allocations[key] += 1
        remaining = 0

    selected = []
    for offset, key in enumerate(sorted(groups)):
        group = groups[key]
        selected.append(
            group.sample(n=allocations[key], random_state=seed + 997 * offset)
        )
    sample = pd.concat(selected, ignore_index=True)
    if len(sample) != sample_size or sample["outcome_id"].duplicated().any():
        raise AssertionError("Audit sample is not the requested set of unique outputs")
    return sample


def load_inputs(
    predictions_path: Path, probabilities_path: Path
) -> tuple[pd.DataFrame, list[str]]:
    predictions = pd.read_csv(predictions_path)
    required = {
        "outcome_id",
        "meeting",
        "year",
        "instrument",
        "title",
        "topic_top1",
        "topic_top2",
        "topic_top3",
        "probability_top1",
        "crossfit_fold",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Prediction file is missing columns: {missing}")
    if len(predictions) != EXPECTED_OUTPUTS:
        raise ValueError(
            f"Expected {EXPECTED_OUTPUTS} regular-ATCM outputs, found {len(predictions)}"
        )
    if predictions["outcome_id"].duplicated().any():
        raise ValueError("Outcome IDs must be unique; retain the year in every ID")
    if predictions["crossfit_fold"].isna().any() or (
        pd.to_numeric(predictions["crossfit_fold"], errors="coerce") <= 0
    ).any():
        raise ValueError("Every output must have a held-meeting cross-fit fold")
    probabilities = pd.read_csv(probabilities_path, usecols=["topic"])
    topics = sorted(probabilities["topic"].dropna().astype(str).unique())
    if len(topics) != EXPECTED_TOPICS:
        raise ValueError(f"Expected {EXPECTED_TOPICS} official concerns, found {len(topics)}")
    return predictions, topics


def prepare_sample(
    predictions: pd.DataFrame, sample_size: int, seed: int
) -> tuple[pd.DataFrame, dict[str, float]]:
    frame = predictions.copy()
    frame["time_band"] = frame["year"].map(time_band)
    lower, upper = frame["probability_top1"].quantile([1 / 3, 2 / 3]).tolist()
    frame["confidence_band"] = pd.cut(
        frame["probability_top1"],
        bins=[-np.inf, lower, upper, np.inf],
        labels=["lower", "middle", "higher"],
        include_lowest=True,
    ).astype(str)
    sample = allocate_sample(frame, sample_size=sample_size, seed=seed)
    rng = random.Random(seed + 1)
    order = list(range(len(sample)))
    rng.shuffle(order)
    sample = sample.iloc[order].reset_index(drop=True)
    sample["audit_order"] = np.arange(1, len(sample) + 1)
    cutpoints = {"lower": float(lower), "upper": float(upper)}
    return sample, cutpoints


def sample_identifier(sample: pd.DataFrame, topics: list[str], seed: int) -> str:
    payload = {
        "seed": seed,
        "outcome_ids": sorted(sample["outcome_id"].astype(str)),
        "topics": topics,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:20]


def blind_records(sample: pd.DataFrame) -> list[dict]:
    visible = [
        "outcome_id",
        "title",
        "instrument",
        "year",
        "meeting",
        "audit_order",
    ]
    records = sample[visible].copy()
    records["item_id"] = records["outcome_id"].map(
        lambda value: hashlib.sha256(str(value).encode()).hexdigest()[:16]
    )
    return records.to_dict(orient="records")


def render_page(items: list[dict], topics: list[str], manifest: dict) -> str:
    page = PAGE.replace("/*__ITEMS__*/", json.dumps(items, ensure_ascii=False))
    page = page.replace("/*__TOPICS__*/", json.dumps(topics, ensure_ascii=False))
    return page.replace("/*__MANIFEST__*/", json.dumps(manifest, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--probabilities", type=Path, default=DEFAULT_PROBABILITIES)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    predictions, topics = load_inputs(args.predictions, args.probabilities)
    sample, cutpoints = prepare_sample(predictions, args.sample_size, args.seed)
    sample_id = sample_identifier(sample, topics, args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)

    blind = pd.DataFrame(blind_records(sample))
    blind.insert(0, "sample_id", sample_id)
    blind.to_csv(args.outdir / "output_topic_audit_sample_blind.csv", index=False)

    key_columns = [
        "outcome_id",
        "topic_top1",
        "topic_top2",
        "topic_top3",
        "probability_top1",
        "margin_top1_top2",
        "crossfit_fold",
        "instrument",
        "year",
        "meeting",
        "time_band",
        "confidence_band",
    ]
    key = sample[key_columns].copy()
    key.insert(0, "sample_id", sample_id)
    key.to_csv(args.outdir / "output_topic_audit_model_key.csv", index=False)

    manifest = {
        "schema": "ats-output-topic-audit-v1",
        "sample_id": sample_id,
        "sample_size": int(len(sample)),
        "seed": args.seed,
        "source_predictions_sha256": sha256_file(args.predictions),
        "confidence_probability_terciles": cutpoints,
        "prediction_design": "held-out-meeting cross-fit",
        "blinding": "Model predictions are absent from the HTML and blind sample.",
        "coding_source": "independent_human",
    }
    (args.outdir / "output_topic_audit_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    html = render_page(blind_records(sample), topics, manifest)
    (args.outdir / "output_topic_audit.html").write_text(html)

    counts = (
        sample.groupby(["instrument", "time_band", "confidence_band"], observed=True)
        .size()
        .rename("n")
        .reset_index()
    )
    counts.to_csv(args.outdir / "output_topic_audit_strata.csv", index=False)
    print(f"Wrote blind audit {sample_id} with {len(sample)} outputs to {args.outdir}")
    print(counts.to_string(index=False))


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ATS formal-output concern audit</title>
<style>
:root{--bg:#f5f4f0;--card:#fff;--ink:#252326;--muted:#6e6a70;--line:#dedbd3;--accent:#2c6e9c;--ok:#5b7f45;--warn:#c96a2b}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
#top{position:sticky;top:0;z-index:3;background:var(--ink);color:#fff;padding:10px 18px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
#top h1{font-size:15px;margin:0}.grow{flex:1}#top input{width:150px;padding:6px 8px;border:1px solid #777;border-radius:5px;background:#38353a;color:#fff}
button{border:1px solid var(--line);border-radius:7px;background:#fff;padding:8px 11px;cursor:pointer}#top button{background:#454147;color:#fff;border-color:#666}
#nav{position:sticky;top:52px;z-index:2;background:#ebe9e3;border-bottom:1px solid var(--line);padding:7px 14px;display:flex;gap:4px;overflow-x:auto}
.dot{min-width:18px;height:18px;border:1px solid #d2cec5;border-radius:4px;background:#fff;color:#888;font:9px/16px monospace;text-align:center;cursor:pointer}.dot.done{background:#dbe8d5;color:#3f6730}.dot.cur{outline:2px solid var(--accent)}
.wrap{max-width:820px;margin:0 auto;padding:22px 18px 70px}.protocol,.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px 20px;margin-bottom:15px}
.protocol{font-size:13px;color:var(--muted)}.protocol b{color:var(--ink)}.meta{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px}.pill{background:#eeeae3;border-radius:12px;padding:2px 9px;font-size:12px}
.title{font-size:21px;line-height:1.28;font-weight:650;margin:8px 0 18px}.q{font-weight:650;margin:15px 0 5px}select,textarea{width:100%;border:1px solid var(--line);border-radius:7px;padding:9px;background:#fff;font:14px/1.4 inherit}
textarea{min-height:70px;resize:vertical}.opts{display:flex;gap:7px;flex-wrap:wrap}.opt{padding:7px 11px}.opt.sel{background:var(--accent);border-color:var(--accent);color:#fff}.row{display:flex;gap:8px;margin-top:18px}.row button{flex:1}.row .primary{background:var(--ink);border-color:var(--ink);color:#fff}
#stats{text-align:center;color:var(--muted);font-size:12px}.attest{font-size:12px;display:flex;gap:7px;align-items:center;color:#ddd}.attest input{width:auto!important}
</style></head><body>
<div id="top"><h1>ATS output concern audit</h1><span id="progress">0 / 0</span><div class="grow"></div><label>Coder <input id="coder" placeholder="unique ID"></label><button id="download">Download</button><button id="import">Import</button><input id="file" type="file" accept="application/json" hidden></div>
<div id="nav"></div><main class="wrap">
<div class="protocol"><b>Code from the title shown here.</b> Choose the official concern that best captures the formal output. Add a secondary concern only when the title clearly spans two concerns. Confidence describes your own judgment. The classifier's prediction is deliberately hidden. Work independently and do not consult another coder's choices.</div>
<div id="app"></div><div id="stats"></div></main>
<script>
const ITEMS=/*__ITEMS__*/; const TOPICS=/*__TOPICS__*/; const MANIFEST=/*__MANIFEST__*/;
const STORE='ats_output_topic_audit_'+MANIFEST.sample_id; let index=0;
function esc(value){return String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function coder(){return (document.getElementById('coder').value||'').trim()}
function all(){try{return JSON.parse(localStorage.getItem(STORE)||'{}')}catch(e){return {}}}
function put(value){localStorage.setItem(STORE,JSON.stringify(value))}
function mine(){return all()[coder()]||{}}
function options(selected,optional=false){let values=optional?[''].concat(TOPICS):TOPICS;return values.map(v=>`<option value="${esc(v)}" ${v===selected?'selected':''}>${esc(v||'None')}</option>`).join('')}
function firstOpen(){const c=mine();for(let i=0;i<ITEMS.length;i++)if(!c[ITEMS[i].item_id])return i;return 0}
function nav(){const c=mine();document.getElementById('progress').textContent=`${Object.keys(c).length} / ${ITEMS.length}`;document.getElementById('nav').innerHTML=ITEMS.map((it,i)=>`<div class="dot ${c[it.item_id]?'done':''} ${i===index?'cur':''}" data-i="${i}">${i+1}</div>`).join('');document.querySelectorAll('.dot').forEach(el=>el.onclick=()=>show(+el.dataset.i))}
function render(){const it=ITEMS[index], prior=mine()[it.item_id]||{};document.getElementById('app').innerHTML=`<section class="card"><div class="meta"><span class="pill">${esc(it.instrument)}</span><span class="pill">ATCM ${it.meeting}</span><span class="pill">${it.year}</span></div><div class="title">${esc(it.title)}</div><div class="q">Primary concern</div><select id="primary"><option value="">Select one</option>${options(prior.primary)}</select><div class="q">Secondary concern <span style="font-weight:400;color:var(--muted)">(optional)</span></div><select id="secondary">${options(prior.secondary,true)}</select><div class="q">Confidence</div><div class="opts" id="confidence">${['high','medium','low'].map(v=>`<button class="opt ${prior.confidence===v?'sel':''}" data-v="${v}">${v}</button>`).join('')}</div><div class="q">Notes <span style="font-weight:400;color:var(--muted)">(optional)</span></div><textarea id="notes">${esc(prior.notes||'')}</textarea><div class="row"><button id="prev">Previous</button><button id="skip">Skip</button><button class="primary" id="save">Save and next</button></div></section>`;
let confidence=prior.confidence||'';document.querySelectorAll('#confidence .opt').forEach(el=>el.onclick=()=>{confidence=el.dataset.v;document.querySelectorAll('#confidence .opt').forEach(x=>x.classList.remove('sel'));el.classList.add('sel')});
document.getElementById('prev').onclick=()=>show(Math.max(0,index-1));document.getElementById('skip').onclick=()=>show(Math.min(ITEMS.length-1,index+1));document.getElementById('save').onclick=()=>{const primary=document.getElementById('primary').value,secondary=document.getElementById('secondary').value;if(!coder()){alert('Enter a unique coder ID first.');return}if(!primary){alert('Select a primary concern.');return}if(!confidence){alert('Select your confidence.');return}if(primary===secondary){alert('Primary and secondary concerns must differ.');return}const state=all();state[coder()]=state[coder()]||{};state[coder()][it.item_id]={outcome_id:it.outcome_id,primary,secondary,confidence,notes:document.getElementById('notes').value,ts:new Date().toISOString()};put(state);show(Math.min(ITEMS.length-1,index+1))}}
function show(i){index=i;render();nav();stats();document.querySelectorAll('.dot')[i]?.scrollIntoView({block:'nearest',inline:'center'})}
function stats(){document.getElementById('stats').textContent=coder()?`${Object.keys(mine()).length} items saved for ${coder()}`:'Enter a unique coder ID to begin.'}
document.getElementById('coder').oninput=()=>show(firstOpen());
document.getElementById('download').onclick=()=>{const name=coder(),coding=mine();if(!name){alert('Enter a unique coder ID.');return}if(Object.keys(coding).length!==ITEMS.length){alert(`Complete all ${ITEMS.length} items before exporting.`);return}if(!confirm('I coded these items myself, independently, without viewing the model predictions.'))return;const payload={schema:MANIFEST.schema,sample_id:MANIFEST.sample_id,coder:name,coding_source:'independent_human',blind_to_model_predictions:true,exported_at:new Date().toISOString(),codings:coding};const blob=new Blob([JSON.stringify(payload,null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=`output_topic_audit_${name}.json`;a.click();URL.revokeObjectURL(a.href)};
document.getElementById('import').onclick=()=>document.getElementById('file').click();document.getElementById('file').onchange=e=>{const f=e.target.files[0];if(!f)return;const reader=new FileReader();reader.onload=()=>{try{const incoming=JSON.parse(reader.result);if(incoming.schema!==MANIFEST.schema||incoming.sample_id!==MANIFEST.sample_id)throw Error('wrong audit sample');if(incoming.coding_source!=='independent_human')throw Error('not an independent human export');const state=all();state[incoming.coder]=Object.assign({},state[incoming.coder]||{},incoming.codings||{});put(state);document.getElementById('coder').value=incoming.coder;show(firstOpen())}catch(err){alert('Could not import: '+err.message)}};reader.readAsText(f)};
show(0);
</script></body></html>"""


if __name__ == "__main__":
    main()
