#!/usr/bin/env python3
"""glm_throughput_record.py — assemble a glm-throughput/1 record from glmdsatput output.

The synthetic throughput runner (cmd/glmdsatput -json) prints, per config it sweeps, one line:

    GLMTPUT_JSON {"schema":"glm-throughput/1","backend":...,"decode_tok_s":...,...}

The real-checkpoint path (cmd/modelbench -out) is wrapped by the runner as:

    GLMMODELBENCH_JSON {"model_config":{"model_type":"glm_moe_dsa"},...}

This tool reads either log shape, pulls every run record, and folds them into ONE catalog
record with caller-stamped fields (utc, machine_id, head_sha, arch). It is the throughput
analogue of glm_witness_record.py (which records the cosine-CORRECTNESS witness); this
records the DECODE/PREFILL TOK/S.

HONEST SCOPE travels in the data. Every per-run record already carries a `scope` field
("synthetic-weights;...;NOT-the-753B"); this tool refuses to emit if any run is missing it,
so a number can never be quoted out of its caveat. The full-753B serving rate is the
llama.cpp CPU-offload baseline (~2.66 tok/s decode), recorded elsewhere as a comparison —
these numbers are fak's NATIVE per-token device cost at a reduced scale, not that.

Pure stdlib, no network. Determinism: identical (log, --utc, --machine-id) -> identical bytes.

  python tools/glm_throughput_record.py run.log --utc 2026-06-24T21:00:00Z --machine-id dgx
  python tools/glm_throughput_record.py run.log --utc <iso> --machine-id a100   # public rollup
  python tools/glm_throughput_record.py --self-test                             # offline golden
"""
import argparse
import hashlib
import json
import re
import sys

SCHEMA = "glm-throughput/1"
RECORD_TOOL = "glm_throughput_record.py@1"
LINE_RE = re.compile(r"^.*GLMTPUT_JSON\s+(\{.*\})\s*$", re.M)
MODELBENCH_RE = re.compile(r"^.*GLMMODELBENCH_JSON\s+(\{.*\})\s*$", re.M)
HEAD_RE = re.compile(r"^=== HEAD ([0-9a-f]{7,40}) ===", re.M)
NODE_RE = re.compile(r"^=== node (\S+) ", re.M)
ARCH_RE = re.compile(r"arch=(\S+)")


def parse_log(text):
    runs = []
    for m in LINE_RE.finditer(text):
        runs.append(json.loads(m.group(1)))
    modelbench = []
    for m in MODELBENCH_RE.finditer(text):
        modelbench.append(json.loads(m.group(1)))
    head = HEAD_RE.search(text)
    node = NODE_RE.search(text)
    arch = ARCH_RE.search(text)
    return {
        "runs": runs,
        "modelbench": modelbench,
        "head_sha": head.group(1) if head else "unknown",
        "node": node.group(1) if node else "",
        "arch": arch.group(1) if arch else "",
    }


def _number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _backend_label(report):
    backend = report.get("backend")
    if isinstance(backend, dict):
        selected = backend.get("selected") or "unknown"
        tier = backend.get("tier")
        cls = backend.get("class")
        suffix = " ".join(x for x in [f"tier={tier}" if tier else "", f"class={cls}" if cls else ""] if x)
        return f"{selected} ({suffix})" if suffix else selected
    return str(backend or "")


def _real_glm_config(report):
    cfg = report.get("model_config") if isinstance(report.get("model_config"), dict) else {}
    family = " ".join([str(cfg.get("model_type") or "")] + [str(a) for a in cfg.get("architectures") or []]).lower()
    return cfg, "glm" in family and "dsa" in family


def normalize_modelbench_report(report):
    cfg, is_glm_dsa = _real_glm_config(report)
    if not is_glm_dsa:
        raise ValueError("GLMMODELBENCH_JSON report is not a glm_moe_dsa checkpoint; refusing to record")
    decode = report.get("decode") if isinstance(report.get("decode"), dict) else {}
    prefills = report.get("prefill") if isinstance(report.get("prefill"), list) else []
    decode_tps = _number(decode.get("tok_per_sec"))
    decode_ms = _number(decode.get("per_token_median_ms"))
    best_prefill = None
    for row in prefills:
        if not isinstance(row, dict):
            continue
        row_tokens = _number(row.get("tokens")) or 0
        best_tokens = _number(best_prefill.get("tokens")) if isinstance(best_prefill, dict) else 0
        if best_prefill is None or row_tokens > (best_tokens or 0):
            best_prefill = row
    return {
        "schema": SCHEMA,
        "harness": "cmd/modelbench",
        "backend": _backend_label(report),
        "model": "glm_moe_dsa",
        "model_label": report.get("model"),
        "source": report.get("source"),
        "precision": report.get("precision"),
        "model_config": cfg,
        "prompt_len": decode.get("prompt_tokens"),
        "decode_steps": decode.get("decode_steps"),
        "decode_ms_tok": round(decode_ms, 3) if decode_ms is not None else None,
        "decode_tok_s": round(decode_tps, 2) if decode_tps is not None else None,
        "prefill": prefills,
        "prefill_ms": best_prefill.get("median_ms") if isinstance(best_prefill, dict) else None,
        "prefill_tok_s": best_prefill.get("tok_per_sec") if isinstance(best_prefill, dict) else None,
        "scope": "real-checkpoint;arch-blind-modelbench;not-synthetic",
    }


def build_record(log_bytes, utc, machine_id, public):
    text = log_bytes.decode("utf-8", "replace")
    parsed = parse_log(text)
    runs = list(parsed["runs"])
    runs.extend(normalize_modelbench_report(r) for r in parsed["modelbench"])
    content_sha = hashlib.sha256(log_bytes).hexdigest()

    # Every run must carry its honest scope, or we refuse — a number without its caveat is a leak.
    for r in runs:
        if not r.get("scope"):
            raise ValueError("a GLMTPUT_JSON run is missing its `scope` caveat; refusing to record")

    # rc / verdict: PASS iff we got at least one run and every run has a positive decode rate.
    ok = bool(runs) and all(isinstance(r.get("decode_tok_s"), (int, float)) and r["decode_tok_s"] > 0 for r in runs)
    rc = 0 if ok else 1

    # best decode row (highest decode_tok_s) surfaced as a convenience headline; full sweep kept.
    best = max(runs, key=lambda r: r.get("decode_tok_s", 0) or 0) if runs else None
    scopes = sorted(set(r.get("scope", "") for r in runs if r.get("scope")))
    top_scope = scopes[0] if len(scopes) == 1 else "mixed;see-runs"

    rec = {
        "schema": SCHEMA,
        "utc": utc,
        "machine_id": machine_id,
        "head_sha": parsed["head_sha"],
        "arch": parsed["arch"],
        "model": "glm_moe_dsa",
        "n_configs": len(runs),
        "runs": runs,
        "best_decode_tok_s": round(_number(best.get("decode_tok_s")) or 0, 2) if best else None,
        "best_decode_config": (best.get("config") or best.get("model_config") or {"source": best.get("source")}) if best else None,
        "rc": rc,
        "verdict": "PASS" if rc == 0 else "FAIL",
        "content_sha256": content_sha,
        "log_bytes": len(log_bytes),
        "record_tool": RECORD_TOOL,
        "scope": top_scope,
    }
    # node: private only. Public rollup drops it (same discipline as glm_witness_record.py).
    if not public and parsed["node"]:
        rec["node"] = parsed["node"]
    return rec


GOLDEN_LOG = b"""=== node node-dgx-a gpu=0 arch=sm_80 nonce=deadbeef ===
=== HEAD f39796e ===
GLMTPUT_JSON {"schema":"glm-throughput/1","backend":"cuda (tier=sm_80 class=approx)","build_ms":12.3,"config":{"heads":16,"hidden":2048,"inter":8192,"layers":8,"vocab":8192},"decode_ms_tok":4.0,"decode_steps":64,"decode_tok_s":250.0,"mla_dsa":{"index_dim":128,"index_heads":16,"index_topk":256,"kv_lora":512,"q_lora":1536,"qk_nope":128,"qk_rope":64,"v_head":128},"model":"glm_moe_dsa","precision":"Q8_0","prefill_ms":80.0,"prefill_tok_s":6400.0,"prompt_len":512,"reps":5,"scope":"synthetic-weights;reduced-layers;dense-FFN(no-MoE);optimistic-lower-bound;NOT-the-753B"}
GLMTPUT_JSON {"schema":"glm-throughput/1","backend":"cuda (tier=sm_80 class=approx)","build_ms":12.3,"config":{"heads":16,"hidden":4096,"inter":8192,"layers":16,"vocab":8192},"decode_ms_tok":9.0,"decode_steps":64,"decode_tok_s":111.0,"mla_dsa":{"index_dim":128,"index_heads":16,"index_topk":256,"kv_lora":512,"q_lora":1536,"qk_nope":128,"qk_rope":64,"v_head":128},"model":"glm_moe_dsa","precision":"Q8_0","prefill_ms":160.0,"prefill_tok_s":3200.0,"prompt_len":512,"reps":5,"scope":"synthetic-weights;reduced-layers;dense-FFN(no-MoE);optimistic-lower-bound;NOT-the-753B"}
"""

GOLDEN_MODELBENCH_LOG = b"""=== node node-dgx-a gpu=2 arch=sm_80 nonce=bead ===
=== HEAD abc1234 ===
GLMMODELBENCH_JSON {"app_version":"0.32.0","backend":{"class":"approx","selected":"cuda","tier":"sm_80"},"decode":{"decode_steps":64,"per_token_median_ms":12.5,"prompt_tokens":512,"reps":5,"tok_per_sec":80.0},"engine":"fak-in-kernel via compute HAL backend \\"cuda\\"","lean":true,"model":"glm52-tiny (glm_moe_dsa) [lean]","model_config":{"architectures":["GlmMoeDsaForCausalLM"],"hidden_size":128,"index_topk":16,"model_type":"glm_moe_dsa","num_hidden_layers":4},"precision":"Q8_0","prefill":[{"median_ms":100.0,"reps":3,"tokens":512,"tok_per_sec":5120.0}],"source":"/models/glm52/model.safetensors (quantize-at-load)"}
"""


def self_test():
    priv = build_record(GOLDEN_LOG, "2026-06-24T21:00:00Z", "dgx", public=False)
    pub = build_record(GOLDEN_LOG, "2026-06-24T21:00:00Z", "a100", public=True)

    def check(c, msg):
        if not c:
            print("SELF-TEST FAIL:", msg)
            sys.exit(1)

    check(priv["schema"] == SCHEMA, "schema")
    check(priv["n_configs"] == 2, "n_configs")
    check(priv["verdict"] == "PASS", "verdict")
    check(priv["best_decode_tok_s"] == 250.0, "best decode")
    check(priv["best_decode_config"]["layers"] == 8, "best config")
    check(priv.get("node") == "node-dgx-a", "private keeps node")
    check("node" not in pub, "public drops node")
    check(pub["machine_id"] == "a100", "public machine_id")
    check(pub["content_sha256"] == priv["content_sha256"], "sha stable across public flag")
    real = build_record(GOLDEN_MODELBENCH_LOG, "2026-06-25T05:00:00Z", "dgx", public=False)
    check(real["scope"] == "real-checkpoint;arch-blind-modelbench;not-synthetic", "real scope")
    check(real["n_configs"] == 1, "real n_configs")
    check(real["runs"][0]["harness"] == "cmd/modelbench", "real harness")
    check(real["best_decode_tok_s"] == 80.0, "real best decode")
    check(real["best_decode_config"]["model_type"] == "glm_moe_dsa", "real model type")
    # missing-scope run is refused
    bad = GOLDEN_LOG.replace(b';NOT-the-753B', b'').replace(b'"scope":"synthetic-weights;reduced-layers;dense-FFN(no-MoE);optimistic-lower-bound', b'"scope":""')
    try:
        build_record(bad, "x", "dgx", public=False)
        check(False, "should refuse scope-less run")
    except ValueError:
        pass
    print("SELF-TEST OK — glm-throughput/1: 2 configs, best decode=250.0 tok/s, scrub + scope-guard verified")


def main():
    ap = argparse.ArgumentParser(description="assemble a glm-throughput/1 record from glmdsatput -json output")
    ap.add_argument("log", nargs="?", help="path to the captured -json run log ('-' for stdin)")
    ap.add_argument("--utc", help="ISO-8601 UTC run stamp (REQUIRED for a real record)")
    ap.add_argument("--machine-id", default="dgx", help="catalog machine id (default dgx; use a100 for the public rollup)")
    ap.add_argument("--public", action="store_true", help="public rollup: drop node (use with --machine-id a100)")
    ap.add_argument("--self-test", action="store_true", help="offline golden check, no I/O")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.log or not args.utc:
        ap.error("log and --utc are required for a real record")
    data = sys.stdin.buffer.read() if args.log == "-" else open(args.log, "rb").read()
    rec = build_record(data, args.utc, args.machine_id, args.public)
    json.dump(rec, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
