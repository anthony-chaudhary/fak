#!/usr/bin/env python3
r"""vLLM adjudication-tax head-to-head witness.

This makes vLLM a first-class MEASURED comparison peer the way SGLang already is
in tools/glm52_serving_witness.py. It does NOT claim fak is faster than vLLM.

It measures the *adjudication TAX*: the cost of fronting a vLLM OpenAI-compatible
``/v1`` endpoint with a ``fak serve`` gateway. fak is EXPECTED to TRAIL raw vLLM
because the gateway adds an adjudication/governance/coherence plane (pre-send
quarantine, result-admission, cache discipline) on every turn. The value of fak
here is that plane and the fact the tax is MEASURED, not raw tok/s. This witness
records that tax honestly so a regression in it can be caught -- it never frames
the tax as a fak win.

The same chat payload (temperature 0, fixed max_tokens) is run N times against
(a) the raw vLLM ``--base-url`` and (b) the fak gateway started in front of it,
interleaved per sample to cancel server variance. The TAX is a delta:

    latency_tax   = gateway median latency / raw median latency   (>=1 => slower)
    decode_tps_tax = raw median decode_tps / gateway median decode_tps (>=1 => slower)

An optional regression gate (borrowed from tools/bench_witness.py) verdicts the
latency tax against a recorded baseline:

    KEEP     (exit 0)  latency_tax <= baseline * (1 + tolerance/100)  [or no baseline: recorded]
    REVERT   (exit 3)  latency_tax regressed past tolerance
    NO_BENCH (exit 4)  no measurement was possible (a leg failed) -- never a silent KEEP
    ERROR    (exit 5)  the witness could not run a leg / build the gateway

``--dry-run`` records the planned ``fak serve`` command and the planned raw +
gateway endpoints with NO traffic, and exits 0 (a PLANNED report).

This file is intentionally stdlib-only so it runs anywhere CI runs. It REUSES the
glm52 serving-witness machinery (build_fak_serve_command / start_gateway /
stop_gateway / perf_metrics / run_chat / json_post / utc_now / model_ids /
gpu_snapshot) rather than re-implementing it.

    python tools/vllm_tax_witness.py --base-url http://node:8000/v1 --model qwen --record
    python tools/vllm_tax_witness.py --dry-run --base-url http://node:8000/v1 --model qwen
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# Reuse the glm52 serving-witness machinery rather than re-implementing the
# Windows/POSIX subprocess handling, the OpenAI HTTP plumbing, and the perf
# parser. Importing it keeps a SINGLE source of truth for the spawn-without-a-
# console-flash behaviour (CREATE_NEW_PROCESS_GROUP|CREATE_NO_WINDOW / taskkill
# on nt, start_new_session / os.killpg on posix).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import glm52_serving_witness as gw  # noqa: E402

SCHEMA = "fak.vllm-adjudication-tax.v1"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

KEEP, REVERT, NO_BENCH, ERROR = "KEEP", "REVERT", "NO_BENCH", "ERROR"
EXIT = {KEEP: 0, REVERT: 3, NO_BENCH: 4, ERROR: 5}
DEFAULT_TOLERANCE_PCT = 25.0  # the gateway tax carries run-to-run variance; 25% is a real regression
DEFAULT_BASELINE_FILE = ".vllm-tax-baseline.json"

# The honest framing the report must carry: fak TRAILS raw vLLM by design.
TAX_FRAMING = (
    "This measures the fak gateway adjudication tax: fak TRAILS raw vLLM by "
    "design because the gateway adds a governance/coherence plane on every "
    "turn. The value is that plane and the MEASURED tax, not raw tok/s -- the "
    "tax is never framed as a fak win."
)


def chat_payload(model: str, max_tokens: int) -> dict[str, Any]:
    """The single deterministic payload run against both legs."""
    return {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: VLLM_TAX_OK"}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }


def sample_leg(name: str, base_url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    """One chat call against a /v1 base, returning the glm52 run_chat row.

    run_chat appends /chat/completions to the base, so pass a base ending /v1.
    """
    return gw.run_chat(name, base_url, payload, timeout_s)


def leg_latency(sample: dict[str, Any]) -> float | None:
    perf = sample.get("perf") if isinstance(sample.get("perf"), dict) else {}
    return gw.number(perf.get("duration_s"))


def leg_decode_tps(sample: dict[str, Any]) -> float | None:
    perf = sample.get("perf") if isinstance(sample.get("perf"), dict) else {}
    return gw.number(perf.get("decode_tps"))


def medians(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Fold a list of run_chat rows into median latency + decode_tps + an ok count."""
    ok = [s for s in samples if s.get("status") == "PASS"]
    lats = [v for v in (leg_latency(s) for s in ok) if v is not None]
    tps = [v for v in (leg_decode_tps(s) for s in ok) if v is not None]
    return {
        "samples": len(samples),
        "ok_samples": len(ok),
        "median_latency_s": round(median(lats), 6) if lats else None,
        "median_decode_tps": round(median(tps), 3) if tps else None,
    }


def compute_tax(raw: dict[str, Any], gateway: dict[str, Any]) -> dict[str, Any]:
    """Latency tax = gateway median / raw median (>=1 => fak slower).
    decode-tps tax = raw median / gateway median (>=1 => fak slower)."""
    raw_lat = raw.get("median_latency_s")
    gw_lat = gateway.get("median_latency_s")
    raw_tps = raw.get("median_decode_tps")
    gw_tps = gateway.get("median_decode_tps")
    latency_tax = round(gw_lat / raw_lat, 4) if raw_lat and gw_lat and raw_lat > 0 else None
    decode_tps_tax = round(raw_tps / gw_tps, 4) if raw_tps and gw_tps and gw_tps > 0 else None
    return {
        "latency_tax": latency_tax,
        "decode_tps_tax": decode_tps_tax,
        "raw_median_latency_s": raw_lat,
        "gateway_median_latency_s": gw_lat,
        "raw_median_decode_tps": raw_tps,
        "gateway_median_decode_tps": gw_tps,
        "interpretation": (
            "latency_tax/decode_tps_tax >= 1 means fak's gateway is slower than "
            "raw vLLM, which is the expected, by-design cost of adjudication"
        ),
    }


def load_baseline(path: Path, key: str) -> float | None:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    entry = (doc.get("baselines") or {}).get(key)
    if isinstance(entry, dict):
        try:
            return float(entry.get("latency_tax"))
        except (TypeError, ValueError):
            return None
    return None


def save_baseline(path: Path, key: str, latency_tax: float, model: str) -> None:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(doc, dict):
            doc = {}
    except (OSError, ValueError):
        doc = {}
    doc.setdefault("schema", "fak.vllm-adjudication-tax-baseline.v1")
    doc.setdefault("baselines", {})
    doc["baselines"][key] = {"latency_tax": latency_tax, "model": model}
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def gate(tax: dict[str, Any], *, baseline_latency_tax: float | None, baseline_file: Path,
         key: str, tolerance_pct: float, model: str, record: bool) -> dict[str, Any]:
    """Verdict the measured latency tax against a recorded baseline.

    No measurement => NO_BENCH (never a silent KEEP). No baseline => record it
    (KEEP). Otherwise KEEP iff latency_tax <= baseline * (1 + tolerance/100)."""
    measured = tax.get("latency_tax")
    if measured is None:
        return {
            "verdict": NO_BENCH,
            "ok": False,
            "reason": "no latency tax could be measured (a leg failed); "
                      "witness this run by the leg errors above, not a keep-bit",
            "tolerance_pct": tolerance_pct,
            "baseline_latency_tax": baseline_latency_tax,
            "baseline_file": str(baseline_file),
        }
    base = baseline_latency_tax if baseline_latency_tax is not None else load_baseline(baseline_file, key)
    if base is None:
        if record:
            save_baseline(baseline_file, key, float(measured), model)
        return {
            "verdict": KEEP,
            "ok": True,
            "measured_latency_tax": measured,
            "baseline_latency_tax": None,
            "tolerance_pct": tolerance_pct,
            "regression_pct": None,
            "baseline_recorded": bool(record),
            "baseline_file": str(baseline_file),
            "reason": (f"no baseline for {key}; measured latency_tax {measured}"
                       + (" and recorded it as the baseline" if record
                          else " (pass --record to set it as the keep floor)")),
        }
    ceiling = base * (1.0 + tolerance_pct / 100.0)
    regression_pct = round((measured - base) / base * 100.0, 2) if base > 0 else None
    keep = measured <= ceiling
    verdict = KEEP if keep else REVERT
    if record and keep:
        save_baseline(baseline_file, key, float(measured), model)
    return {
        "verdict": verdict,
        "ok": keep,
        "measured_latency_tax": measured,
        "baseline_latency_tax": base,
        "ceiling_latency_tax": round(ceiling, 4),
        "tolerance_pct": tolerance_pct,
        "regression_pct": regression_pct,
        "baseline_recorded": bool(record and keep),
        "baseline_file": str(baseline_file),
        "reason": (f"latency_tax {measured} vs baseline {base} "
                   f"({regression_pct:+.1f}% if base>0, tolerance {tolerance_pct:.0f}%) -> {verdict}"),
    }


def acceptance(report: dict[str, Any]) -> dict[str, dict[str, str]]:
    raw = report.get("raw_vllm") or {}
    gateway = report.get("gateway") or {}
    tax = report.get("tax") or {}
    gate_row = report.get("gate") or {}
    gw_info = report.get("gateway_process") or {}
    raw_ok = raw.get("ok_samples", 0) > 0
    gw_ok = gateway.get("ok_samples", 0) > 0
    tax_measured = tax.get("latency_tax") is not None
    cmd = gw_info.get("command")
    gw_started = bool(cmd) or gw_info.get("mode") == "existing"
    return {
        "gateway_command": {
            "status": "PASS" if gw_started else "FAIL",
            "detail": gw_info.get("mode", "missing"),
        },
        "raw_vllm_chat": {
            "status": "PASS" if raw_ok else "FAIL",
            "detail": f"ok_samples={raw.get('ok_samples', 0)}/{raw.get('samples', 0)}",
        },
        "gateway_chat": {
            "status": "PASS" if gw_ok else "FAIL",
            "detail": f"ok_samples={gateway.get('ok_samples', 0)}/{gateway.get('samples', 0)}",
        },
        "tax_measured": {
            "status": "PASS" if tax_measured else "FAIL",
            "detail": (f"latency_tax={tax.get('latency_tax')} "
                       f"decode_tps_tax={tax.get('decode_tps_tax')}"),
        },
        "regression_gate": {
            "status": "PASS" if gate_row.get("verdict") in {KEEP, NO_BENCH} else "FAIL",
            "detail": gate_row.get("reason", "missing"),
        },
        "honest_framing": {
            "status": "PASS",
            "detail": "report frames the tax as a by-design cost, never a fak win",
        },
    }


def summarize(report: dict[str, Any]) -> None:
    checks = acceptance(report)
    report["acceptance"] = checks
    if report.get("dry_run"):
        state = "PLANNED"
    elif all(row["status"] == "PASS" for row in checks.values()):
        state = "PASS"
    else:
        state = "FAIL"
    report["summary"] = {
        "vllm_adjudication_tax_witness": state,
        "verdict": (report.get("gate") or {}).get("verdict"),
        "latency_tax": (report.get("tax") or {}).get("latency_tax"),
        "decode_tps_tax": (report.get("tax") or {}).get("decode_tps_tax"),
        "framing": TAX_FRAMING,
        "passed": sum(1 for row in checks.values() if row["status"] == "PASS"),
        "checks": len(checks),
    }


def markdown(report: dict[str, Any]) -> str:
    summ = report.get("summary", {})
    tax = report.get("tax") or {}
    lines = [
        "# vLLM Adjudication-Tax Witness",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Model: `{report['model']}`",
        f"- Raw vLLM base URL: `{report['base_url']}`",
        f"- Witness: `{summ.get('vllm_adjudication_tax_witness')}`",
        f"- Latency tax (gateway/raw): `{tax.get('latency_tax')}`",
        f"- Decode-tps tax (raw/gateway): `{tax.get('decode_tps_tax')}`",
        "",
        f"> {TAX_FRAMING}",
        "",
        "| check | status | detail |",
        "|---|---|---|",
    ]
    for name, row in report["acceptance"].items():
        lines.append(f"| `{name}` | {row['status']} | {row.get('detail', '')} |")
    lines += [
        "",
        "fak is EXPECTED to TRAIL raw vLLM here; the value is the measured "
        "governance plane, not raw tok/s.",
        "",
    ]
    return "\n".join(lines)


def write_report(path: str, report: dict[str, Any]) -> None:
    out = Path(path)
    if out.parent and str(out.parent) not in ("", "."):
        out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _gateway_base(origin: str) -> str:
    """The /v1 base run_chat expects, rebuilt from a bare gateway origin."""
    return gw.root_url(origin).rstrip("/") + "/v1"


def run_samples(args: argparse.Namespace, raw_base: str, gw_base: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run N interleaved raw/gateway samples to cancel per-server variance."""
    payload = chat_payload(args.model, args.max_tokens)
    raw_samples: list[dict[str, Any]] = []
    gw_samples: list[dict[str, Any]] = []
    for _ in range(args.count):
        raw_samples.append(sample_leg("raw_vllm", raw_base, payload, args.http_timeout_s))
        gw_samples.append(sample_leg("gateway", gw_base, payload, args.http_timeout_s))
    return raw_samples, gw_samples


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Measure the fak gateway adjudication tax over a vLLM /v1 endpoint")
    ap.add_argument("--base-url", required=True, help="raw vLLM OpenAI-compatible endpoint, e.g. http://node:8000/v1")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--provider", default="openai")
    ap.add_argument("--api-key-env", default="")
    ap.add_argument("--count", type=int, default=6, help="samples per leg, interleaved (default: 6)")
    ap.add_argument("--max-tokens", type=int, default=32, help="fixed completion budget per sample")
    ap.add_argument("--engine-cache-engine", default="vllm", choices=["", "sglang", "vllm"],
                    help="engine cache reset fallback passed through to the gateway (default: vllm)")
    ap.add_argument("--engine-cache-base-url", default="")
    ap.add_argument("--engine-cache-admin-key-env", default="")
    ap.add_argument("--engine-cache-idle-timeout-s", type=int, default=0)
    ap.add_argument("--engine-cache-require-exact-span", action="store_true")
    ap.add_argument("--gateway-url", default="", help="existing fak gateway origin; if omitted this runner starts one in front of vLLM")
    ap.add_argument("--gateway-port", type=int, default=0)
    ap.add_argument("--gateway-start-timeout-s", type=float, default=45.0)
    ap.add_argument("--http-timeout-s", type=float, default=120.0)
    ap.add_argument("--fak-command", default="go run ./cmd/fak")
    ap.add_argument("--tolerance-pct", type=float, default=DEFAULT_TOLERANCE_PCT,
                    help=f"latency-tax regression tolerance %% (default: {DEFAULT_TOLERANCE_PCT})")
    ap.add_argument("--baseline-latency-tax", type=float, default=None,
                    help="explicit baseline latency tax (overrides the baseline file)")
    ap.add_argument("--baseline-file", default=DEFAULT_BASELINE_FILE,
                    help=f"baseline JSON, resolved under repo root (default: {DEFAULT_BASELINE_FILE})")
    ap.add_argument("--record", action="store_true",
                    help="record the measured latency tax as the baseline (on KEEP / no-baseline)")
    ap.add_argument("--out", default="experiments/vllm/adjudication-tax-witness.json")
    ap.add_argument("--markdown", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    env = os.environ.copy()
    raw_base = args.base_url.rstrip("/")
    baseline_file = (ROOT / args.baseline_file) if not os.path.isabs(args.baseline_file) else Path(args.baseline_file)
    key = f"{args.model}::{args.engine_cache_engine or 'no-cache'}"

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": gw.utc_now(),
        "model": args.model,
        "base_url": raw_base,
        "provider": args.provider,
        "count": args.count,
        "max_tokens": args.max_tokens,
        "engine_cache_engine": args.engine_cache_engine,
        "dry_run": args.dry_run,
        "framing": TAX_FRAMING,
    }

    proc: subprocess.Popen[str] | None = None
    try:
        if args.dry_run:
            port = args.gateway_port or 0
            planned_cmd = gw.build_fak_serve_command(args, port)
            report["gateway_process"] = {"mode": "planned", "command": planned_cmd}
            report["planned_endpoints"] = {
                "raw_vllm": raw_base + "/chat/completions",
                "gateway": (f"http://127.0.0.1:{args.gateway_port}" if args.gateway_port else "http://127.0.0.1:<port>") + "/v1/chat/completions",
            }
            report["gpu"] = gw.gpu_snapshot()
            report["raw_vllm"] = {"status": "PLANNED", "samples": 0, "ok_samples": 0}
            report["gateway"] = {"status": "PLANNED", "samples": 0, "ok_samples": 0}
            report["tax"] = {"latency_tax": None, "decode_tps_tax": None,
                             "interpretation": "no traffic in --dry-run"}
            report["gate"] = {"verdict": NO_BENCH, "ok": True,
                              "reason": "dry-run: planned only, no measurement",
                              "tolerance_pct": args.tolerance_pct,
                              "baseline_file": str(baseline_file)}
        else:
            status, models, body = gw.json_get(raw_base + "/models", timeout_s=args.http_timeout_s)
            report["upstream_models"] = {"http_status": status, "model_ids": gw.model_ids(models), "body_excerpt": body}
            report["gpu"] = gw.gpu_snapshot()

            proc, gateway_origin, gw_info = gw.start_gateway(args, env)
            report["gateway_process"] = gw_info
            if gw_info.get("mode") in {"start-failed", "start-timeout"}:
                report["raw_vllm"] = {"status": "FAIL", "samples": 0, "ok_samples": 0,
                                      "detail": "gateway not up; skipped to keep legs comparable"}
                report["gateway"] = {"status": "FAIL", "samples": 0, "ok_samples": 0,
                                     "detail": gw_info.get("mode")}
                report["tax"] = compute_tax({}, {})
                report["gate"] = gate(report["tax"], baseline_latency_tax=args.baseline_latency_tax,
                                      baseline_file=baseline_file, key=key,
                                      tolerance_pct=args.tolerance_pct, model=args.model, record=args.record)
            else:
                gw_base = _gateway_base(gateway_origin)
                raw_samples, gw_samples = run_samples(args, raw_base, gw_base)
                report["raw_vllm"] = medians(raw_samples)
                report["raw_vllm"]["leg_samples"] = raw_samples
                report["gateway"] = medians(gw_samples)
                report["gateway"]["leg_samples"] = gw_samples
                report["tax"] = compute_tax(report["raw_vllm"], report["gateway"])
                report["gate"] = gate(report["tax"], baseline_latency_tax=args.baseline_latency_tax,
                                      baseline_file=baseline_file, key=key,
                                      tolerance_pct=args.tolerance_pct, model=args.model, record=args.record)
    finally:
        gw.stop_gateway(proc)

    summarize(report)
    write_report(args.out, report)
    if args.markdown:
        Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown).write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))

    state = report["summary"]["vllm_adjudication_tax_witness"]
    if state == "PLANNED":
        return 0
    if state == "FAIL":
        # A failed leg / failed gate: surface the gate verdict's exit code so the
        # dispatch loop branches like bench_witness. A REVERT or ERROR is a no-keep.
        return EXIT.get(str((report.get("gate") or {}).get("verdict")), 1)
    # PASS: the gate verdict drives the keep-bit exit code (KEEP=0 / NO_BENCH=4).
    return EXIT.get(str((report.get("gate") or {}).get("verdict")), 0)


if __name__ == "__main__":
    raise SystemExit(main())
