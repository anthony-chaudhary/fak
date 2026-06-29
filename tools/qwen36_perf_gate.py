#!/usr/bin/env python3
"""Gate Qwen3.6 pure-fak speed artifacts against llama.cpp artifacts.

This is intentionally artifact-only: it does not rerun either engine. The benchmark run
produces the witness; this script makes the witness fail-closed when a checked fak point
falls below the corresponding llama.cpp point.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCHEMA = "fak.qwen36-perf-gate.v1"
ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CASES = (
    (
        "amd-p16-64-256",
        "experiments/qwen36/native-gguf-q8-hybrid-headscan-p16-64-256-20260619.json",
        "experiments/qwen36/llamacpp-vulkan-qwen36-pp16-64-256-tg1-20260619.json",
    ),
    (
        "amd-p512-1024",
        "experiments/qwen36/native-gguf-q8-hybrid-headscan-p512-1024-dp256-d16-20260619.json",
        "experiments/qwen36/llamacpp-vulkan-qwen36-pp512-1024-tg16-20260619.json",
    ),
)

# The Apple-Silicon Metal arm (#64, epic #300). fak-Metal Q4_K tok/s vs the
# (provenance-caveated #459/#452) llama.cpp-Metal SOTA bar, both witnessed on the
# M3 Pro verify node (node-macos-a). Greedy/first-token parity for the Metal lane
# already holds (TestMetalQ4KDecodeMatchesCPU, token seq == CPU); this is the
# perf half. The #300 acceptance is "within 2x llama.cpp Metal" -> run with
# --min-ratio 0.5. The decode metric is length-agnostic and asserted here; the
# prefill bar (51.55 @ pp22) is recorded but only scored once a matched-length
# fak-Metal prefill point exists (#64 comment#2: fix P=256/512, not P~22).
METAL_CASES = (
    (
        "metal-q4k-27b",
        "experiments/qwen36/metal-fak-q4k-27b-20260626.json",
        "experiments/qwen36/llamacpp-metal-qwen36-bar-20260626.json",
    ),
)

# #300 acceptance bar: fak within 2x of the llama.cpp-Metal SOTA -> fak/llama >= 0.5.
METAL_TARGET_RATIO = 0.5

# #64 comment#2: a Metal prefill assertion must FIX the prompt length at an agentic
# value (P=256/512), NOT the tiny-prompt P~22/29 artifact. Prefill reads each q4_k
# weight ~once per 128-token tile, so a tiny prompt is weight-read-dominated and reads
# ~an order of magnitude below the agentic rate (0.6 tok/s @ P=29 vs 4.5 @ P=421). The
# gate REFUSES to score a Metal prefill pair below this floor: it is recorded for
# provenance but never asserted against the (provenance-caveated #459/#452) bar, so a
# tiny-prompt point can never masquerade as the within-2x verdict.
METAL_MIN_PREFILL_TOKENS = 256


def repo_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return ROOT / p


def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def as_float(value):
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value):
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def fak_metrics(path: Path) -> dict:
    data = load_json(path)
    prefill = {}
    for row in data.get("prefill") or []:
        tokens = as_int(row.get("tokens"))
        tps = as_float(row.get("tok_per_sec"))
        if tokens and tps and tps > 0:
            prefill[tokens] = tps
    dec = data.get("decode") or {}
    decode_tps = as_float(dec.get("tok_per_sec"))
    decode = None
    if decode_tps and decode_tps > 0:
        decode = {
            "tok_per_sec": decode_tps,
            "prompt_tokens": as_int(dec.get("prompt_tokens")),
            "decode_steps": as_int(dec.get("decode_steps")),
        }
    return {
        "path": rel(path),
        "engine": data.get("engine"),
        "model": data.get("model"),
        "prefill": prefill,
        "decode": decode,
    }


def llama_metrics(path: Path) -> dict:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"{rel(path)} is not a llama-bench JSON array")
    prefill = {}
    decode = None
    meta = {}
    for row in data:
        if not meta:
            meta = {
                "build_commit": row.get("build_commit"),
                "build_number": row.get("build_number"),
                "backends": row.get("backends"),
                "model_type": row.get("model_type"),
            }
        # Carry forward bar-provenance caveats (#459/#452) from any row that
        # declares them, so an observed-external bar is never silently asserted
        # as a fak-controlled witness.
        if row.get("provenance") and "provenance" not in meta:
            meta["provenance"] = row.get("provenance")
        if row.get("caveat") and "caveat" not in meta:
            meta["caveat"] = row.get("caveat")
        n_prompt = as_int(row.get("n_prompt")) or 0
        n_gen = as_int(row.get("n_gen")) or 0
        tps = as_float(row.get("avg_ts"))
        if not tps or tps <= 0:
            continue
        if n_prompt > 0 and n_gen == 0:
            prefill[n_prompt] = tps
        elif n_prompt == 0 and n_gen > 0:
            decode = {"tok_per_sec": tps, "n_gen": n_gen}
    return {"path": rel(path), "prefill": prefill, "decode": decode, **meta}


def compare_case(
    label: str,
    fak_path: Path,
    llama_path: Path,
    min_ratio: float,
    min_prefill_tokens: int = 0,
) -> dict:
    fak = fak_metrics(fak_path)
    llama = llama_metrics(llama_path)
    rows = []
    failures = []

    common_prefill = sorted(set(fak["prefill"]) & set(llama["prefill"]))
    for tokens in common_prefill:
        ft = fak["prefill"][tokens]
        lt = llama["prefill"][tokens]
        ratio = ft / lt
        if tokens < min_prefill_tokens:
            # Below the agentic prefill floor (#64 comment#2): a tiny-prompt point is
            # weight-read-dominated and reads ~an order of magnitude below the agentic
            # rate. Record it for provenance but NEVER assert it against the bar -- it
            # contributes no pass/fail, so it can never masquerade as the verdict.
            rows.append({
                "metric": f"prefill_P{tokens}",
                "kind": "prefill",
                "tokens": tokens,
                "fak_tok_per_sec": ft,
                "llama_tok_per_sec": lt,
                "ratio": ratio,
                "passed": True,
                "scored": False,
                "note": f"P{tokens} below agentic prefill floor P>={min_prefill_tokens} "
                        "(#64 comment#2): recorded, not asserted",
            })
            continue
        passed = ratio >= min_ratio
        if not passed:
            failures.append(f"{label} P{tokens}: fak {ft:.4g} < {min_ratio:g}x llama {lt:.4g}")
        rows.append({
            "metric": f"prefill_P{tokens}",
            "kind": "prefill",
            "tokens": tokens,
            "fak_tok_per_sec": ft,
            "llama_tok_per_sec": lt,
            "ratio": ratio,
            "passed": passed,
            "scored": True,
        })

    if fak["decode"] and llama["decode"]:
        ft = fak["decode"]["tok_per_sec"]
        lt = llama["decode"]["tok_per_sec"]
        ratio = ft / lt
        passed = ratio >= min_ratio
        fsteps = fak["decode"].get("decode_steps")
        lsteps = llama["decode"].get("n_gen")
        if not passed:
            failures.append(f"{label} decode: fak {ft:.4g} < {min_ratio:g}x llama {lt:.4g}")
        rows.append({
            "metric": "decode",
            "kind": "decode",
            "fak_decode_steps": fsteps,
            "llama_n_gen": lsteps,
            "fak_tok_per_sec": ft,
            "llama_tok_per_sec": lt,
            "ratio": ratio,
            "passed": passed,
        })

    if not rows:
        failures.append(f"{label}: no comparable prefill or decode metrics")

    return {
        "label": label,
        "fak": fak,
        "llama": llama,
        "rows": rows,
        "passed": not failures,
        "failures": failures,
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Qwen3.6 Perf Gate",
        "",
        f"- Verdict: {'PASS' if report['passed'] else 'FAIL'}",
        f"- Minimum ratio: {report['min_ratio']:.3g}x",
        "",
        "| case | metric | fak tok/s | llama.cpp tok/s | ratio | verdict |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for case in report["cases"]:
        for row in case["rows"]:
            if row.get("scored") is False:
                verdict = "RECORDED"
            else:
                verdict = "PASS" if row["passed"] else "FAIL"
            lines.append(
                f"| {case['label']} | `{row['metric']}` | "
                f"{row['fak_tok_per_sec']:.2f} | {row['llama_tok_per_sec']:.2f} | "
                f"{row['ratio']:.2f}x | {verdict} |"
            )
    caveats = []
    for case in report["cases"]:
        caveat = (case.get("llama") or {}).get("caveat")
        if caveat:
            caveats.append(f"- {case['label']}: {caveat}")
    if caveats:
        lines.extend(["", "Bar provenance (the llama.cpp-Metal bar is observed-external, not a fak witness):"])
        lines.extend(caveats)
    if report["failures"]:
        lines.extend(["", "Failures:"])
        lines.extend(f"- {failure}" for failure in report["failures"])
    lines.append("")
    return "\n".join(lines)


def build_report(cases, min_ratio: float, min_prefill_tokens: int = 0) -> dict:
    checked = []
    failures = []
    for label, fak_path, llama_path in cases:
        case = compare_case(
            label, repo_path(fak_path), repo_path(llama_path), min_ratio, min_prefill_tokens
        )
        checked.append(case)
        failures.extend(case["failures"])
    return {
        "schema": SCHEMA,
        "min_ratio": min_ratio,
        "min_prefill_tokens": min_prefill_tokens,
        "passed": not failures,
        "failures": failures,
        "cases": checked,
    }


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--case",
        action="append",
        nargs=3,
        metavar=("LABEL", "FAK_JSON", "LLAMA_JSON"),
        help="comparison case; may be repeated. Defaults to the committed AMD Qwen3.6 artifacts.",
    )
    ap.add_argument(
        "--metal",
        action="store_true",
        help="run the Apple-Silicon Metal arm (#64/#300: fak-Metal vs the llama.cpp-Metal bar) "
        "instead of the default AMD cases. The #300 acceptance is within-2x -> pair with --min-ratio 0.5.",
    )
    ap.add_argument("--min-ratio", type=float, default=1.0, help="required fak/llama tok/s ratio")
    ap.add_argument("--out", help="write machine-readable gate report JSON")
    ap.add_argument("--markdown", help="write markdown report")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.case:
        cases = args.case
        min_prefill_tokens = 0
    elif args.metal:
        cases = METAL_CASES
        min_prefill_tokens = METAL_MIN_PREFILL_TOKENS
    else:
        cases = DEFAULT_CASES
        min_prefill_tokens = 0
    report = build_report(cases, args.min_ratio, min_prefill_tokens)
    if args.out:
        out = repo_path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown:
        md = repo_path(args.markdown)
        md.parent.mkdir(parents=True, exist_ok=True)
        md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report), end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
