#!/usr/bin/env python3
"""Grade a Qwen3.6-27B Mac parity-gate WITNESS, fail-closed and host-independent.

`tools/qwen36_mac_parity_gate.sh` runs on an Apple-Silicon M3 Pro and emits one
witness JSON (`experiments/agent-live/qwen36-mac-parity-gate-<ts>.json`) with three
arms: correctness (the token ids fak vs the llama.cpp b9707 oracle + the first
divergence index), the #71 Metal hybrid-prefill gate result, and speed vs the
51.55/7.29 bar. That script computes a one-shot inline verdict and exits -- nothing
re-grades the witness afterward, so there is no across-run REGRESSION tracking and no
gate that runs on a plain CPU box.

This is that gate. It is intentionally artifact-only (it reruns no engine): it ingests
the witness and grades it against a FROZEN correctness contract so that

  * a kernel FIX that closes the token-3 drift auto-flips the verdict to PARITY,
  * a REGRESSION that diverges EARLIER than today's known index fails closed,
  * the reference (llama.cpp oracle) silently changing is caught as ORACLE_DRIFT,
  * and a leaked host / IP / `user@host` SSH form in the witness fails the SCRUB.

Because it is pure JSON logic it runs anywhere -- on the win32 orchestrator and in CI --
unlike the Mac gate it grades. The Metal and speed arms are RECORDED, never gated here:
fak is honestly under the bar today, and the Metal arm is gated by the gate script
itself; re-gating them would only redden a CI lane that this tool is meant to keep green
until a real correctness regression appears.

Honesty rule (docs/proofs/00-METHOD.md): a SKIP / missing witness is never a PASS. With
--require-witness an absent witness fails closed; by default it is reported as NO_WITNESS
and the tool exits 0 (a plain CPU box legitimately has no Mac witness yet).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SCHEMA = "fak.qwen36-parity-witness-gate.v1"
ROOT = Path(__file__).resolve().parents[1]
WITNESS_DIR = "experiments/agent-live"
WITNESS_GLOB = "qwen36-mac-parity-gate-*.json"

# ---- the FROZEN correctness contract (source: tools/qwen36_mac_parity_gate.sh header,
#      docs/benchmarks/QWEN36-PARITY-ROLLUP-2026-06-28.md §5). Every number here traces
#      to a committed witness; changing one is a deliberate contract edit, not a tweak. --
ORACLE_IDS = [248068, 198, 90700]          # llama.cpp b9707 greedy (`<think>\nThinking`)
KNOWN_FAK_IDS = [248068, 198, 8160]        # fak GGUF->Q8 greedy today (`<think>\nHere`)
KNOWN_DIVERGENCE_INDEX = 2                  # two-token match, then the token-3 near-tie flip
BAR_PREFILL = 51.55                         # llama.cpp b9707 Metal prefill bar (M3 Pro)
BAR_DECODE = 7.29                           # llama.cpp b9707 Metal decode bar (M3 Pro)

# Open M4 issue map for this witness surface. The witness does not close every
# linked issue by itself; it gives each issue a stable artifact to inspect once the
# Mac node produces a run.
ISSUE_LINKS = {
    "correctness": [64, 1458],
    "metal_gate": [71, 1458],
    "speed": [64, 1382],
    "q6k_fused_mlp": [1381],
}

# The sanctioned PUBLIC placeholder host (docs/fak/scrubbing-real-values.md). Any other
# `.local` hostname, an IP, a `user@host`, or a tailnet `.ts.net` in the witness is a leak.
SCRUB_PLACEHOLDER_HOSTS = ("node-macos-a.local", "node-macos-a")
_IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_SSH_USERHOST = re.compile(r"\b[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+\b")
_TAILNET = re.compile(r"\b[a-zA-Z0-9-]+\.ts\.net\b")
_DOTLOCAL = re.compile(r"\b([a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*\.local)\b")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def find_latest_witness(directory: Path) -> Path | None:
    """Newest witness by filename timestamp (the gate names them ...-<UTC ts>.json)."""
    hits = sorted(directory.glob(WITNESS_GLOB))
    return hits[-1] if hits else None


def _as_int_list(value) -> list[int] | None:
    if not isinstance(value, list):
        return None
    out = []
    for v in value:
        if isinstance(v, bool) or not isinstance(v, int):
            return None
        out.append(v)
    return out


def first_divergence(a: list[int], b: list[int]) -> int:
    """First index where a and b differ; -1 if equal through the shorter length AND
    the lengths match (a true prefix that runs out is NOT parity -- it is unproven)."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n  # ran out before proving equality -> diverges at the short length
    return -1


def grade_correctness(witness: dict) -> dict:
    """Classify the correctness arm against the frozen contract.

    Verdicts:
      PARITY        - fak ids == oracle ids through the window (the GOAL). pass.
      KNOWN_DRIFT   - the exact recorded token-3 signature, unchanged. pass (expected today).
      PROGRESS      - diverges LATER than the known index (partial improvement). pass + review.
      DRIFT_CHANGED - diverges at the same index but on different ids. pass + review (not worse).
      REGRESSION    - diverges EARLIER than the known index (strictly worse). FAIL.
      ORACLE_DRIFT  - the llama.cpp reference ids changed (setup/measurement bug). FAIL.
      MALFORMED     - ids missing / non-int / reported index disagrees with the arrays. FAIL.
    """
    fak = _as_int_list(witness.get("fak_token_ids"))
    oracle = _as_int_list(witness.get("llamacpp_token_ids"))
    reported = witness.get("first_divergence_index")

    if fak is None or oracle is None:
        return _corr("MALFORMED", fak, oracle, None, reported, True,
                     "fak_token_ids or llamacpp_token_ids missing / not an int array")

    # The oracle must EXTEND the frozen reference: its first len(ORACLE_IDS) ids must match.
    # (A longer compared window N>3 keeps the same prefix; only a changed reference fails.)
    if oracle[:len(ORACLE_IDS)] != ORACLE_IDS:
        return _corr("ORACLE_DRIFT", fak, oracle, first_divergence(fak, oracle), reported, True,
                     f"llama.cpp oracle prefix changed from the frozen {ORACLE_IDS} -> "
                     f"{oracle[:len(ORACLE_IDS)]}; the reference moved (setup/measurement bug), "
                     "not a fak result")

    div = first_divergence(fak, oracle)
    # The witness reports its own index; it must agree with the arrays or the file is inconsistent.
    if isinstance(reported, int) and not isinstance(reported, bool) and reported != div:
        return _corr("MALFORMED", fak, oracle, div, reported, True,
                     f"witness first_divergence_index={reported} disagrees with the ids (computed {div})")

    if div == -1:
        return _corr("PARITY", fak, oracle, div, reported, False,
                     "fak reproduces the llama.cpp token stream through the compared window -- "
                     "the token-3 drift is CLOSED")
    if div < KNOWN_DIVERGENCE_INDEX:
        return _corr("REGRESSION", fak, oracle, div, reported, True,
                     f"fak diverges at index {div}, EARLIER than the known index "
                     f"{KNOWN_DIVERGENCE_INDEX} -- strictly worse than the recorded state")
    if div > KNOWN_DIVERGENCE_INDEX:
        return _corr("PROGRESS", fak, oracle, div, reported, False,
                     f"fak diverges at index {div}, LATER than the known index "
                     f"{KNOWN_DIVERGENCE_INDEX} -- partial improvement; confirm it is not a fluke")
    # div == KNOWN_DIVERGENCE_INDEX
    if fak[:len(KNOWN_FAK_IDS)] == KNOWN_FAK_IDS:
        return _corr("KNOWN_DRIFT", fak, oracle, div, reported, False,
                     "the exact recorded token-3 near-tie drift, unchanged (expected today)")
    return _corr("DRIFT_CHANGED", fak, oracle, div, reported, False,
                 f"diverges at the known index {div} but on different ids "
                 f"(got {fak}, recorded {KNOWN_FAK_IDS}) -- not worse, but the signature moved")


def _corr(verdict, fak, oracle, div, reported, is_regression, note) -> dict:
    return {
        "verdict": verdict,
        "fak_ids": fak,
        "oracle_ids": oracle,
        "first_divergence_index": div,
        "reported_divergence_index": reported,
        "known_divergence_index": KNOWN_DIVERGENCE_INDEX,
        "regression": is_regression,
        "note": note,
    }


def scan_for_leaks(witness: dict) -> list[str]:
    """Flag a real host / IP / SSH user@host / tailnet name leaked into the witness.

    Wires the public/private scrub (docs/fak/scrubbing-real-values.md) INTO the gate: the
    witness `host` is meant to be the scrubbed `uname -srm M3Pro-Metal` form, never the
    tailnet node name. A careless future edit that reintroduces a real value fails closed.
    """
    leaks: list[str] = []
    blob = json.dumps(witness)
    for ip in set(_IPV4.findall(blob)):
        # the frozen bars / token ids are not IPs; a dotted-quad in the witness is a host leak
        leaks.append(f"ipv4-address:{ip}")
    for uh in set(_SSH_USERHOST.findall(blob)):
        leaks.append(f"ssh-user@host:{uh}")
    for tn in set(_TAILNET.findall(blob)):
        leaks.append(f"tailnet-host:{tn}")
    for dl in set(_DOTLOCAL.findall(blob)):
        if dl not in SCRUB_PLACEHOLDER_HOSTS:
            leaks.append(f"non-placeholder-host:{dl}")
    return sorted(set(leaks))


def _as_float(value):
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def grade_speed(witness: dict, min_ratio: float) -> dict:
    pref = _as_float(witness.get("fak_prefill_tok_s"))
    dec = _as_float(witness.get("fak_decode_tok_s"))
    bar_p = _as_float(witness.get("bar_prefill_tok_s")) or BAR_PREFILL
    bar_d = _as_float(witness.get("bar_decode_tok_s")) or BAR_DECODE
    pratio = (pref / bar_p) if (pref and bar_p) else 0.0
    dratio = (dec / bar_d) if (dec and bar_d) else 0.0
    failures = []
    if min_ratio > 0:
        if pref and pratio < min_ratio:
            failures.append(f"prefill {pref:.4g} tok/s < {min_ratio:g}x bar {bar_p:.4g}")
        if dec and dratio < min_ratio:
            failures.append(f"decode {dec:.4g} tok/s < {min_ratio:g}x bar {bar_d:.4g}")
    return {
        "prefill_tok_s": pref, "decode_tok_s": dec,
        "bar_prefill_tok_s": bar_p, "bar_decode_tok_s": bar_d,
        "prefill_ratio": round(pratio, 4), "decode_ratio": round(dratio, 4),
        "min_ratio": min_ratio, "gated": min_ratio > 0, "failures": failures,
    }


def grade_witness(witness: dict, witness_path: str = "<inline>", min_ratio: float = 0.0) -> dict:
    correctness = grade_correctness(witness)
    speed = grade_speed(witness, min_ratio)
    leaks = scan_for_leaks(witness)
    metal_pass = bool(witness.get("metal_gate_pass"))

    failures: list[str] = []
    if correctness["regression"]:
        failures.append(f"correctness {correctness['verdict']}: {correctness['note']}")
    if leaks:
        failures.append("scrub leak(s): " + ", ".join(leaks))
    failures.extend(speed["failures"])

    return {
        "schema": SCHEMA,
        "witness": witness_path,
        "issues": ISSUE_LINKS,
        "commit": witness.get("commit"),
        "captured_at": witness.get("captured_at"),
        "host": witness.get("host"),
        "correctness": correctness,
        "metal_gate": {"pass": metal_pass, "gated": False,
                       "note": "recorded; gated by the gate script itself, not re-gated here"},
        "speed": speed,
        "scrub": {"clean": not leaks, "leaks": leaks},
        "passed": not failures,
        "failures": failures,
    }


def no_witness_report(require: bool) -> dict:
    note = ("no Mac parity witness found -- a SKIP is not a PASS "
            f"({WITNESS_DIR}/{WITNESS_GLOB})")
    return {
        "schema": SCHEMA, "witness": "<none>", "status": "NO_WITNESS",
        "issues": ISSUE_LINKS,
        "passed": not require, "failures": [note] if require else [], "note": note,
    }


def issue_links_markdown(issues: dict[str, list[int]]) -> str:
    parts = []
    for key in ("correctness", "metal_gate", "speed", "q6k_fused_mlp"):
        nums = issues.get(key, [])
        label = key.replace("_", " ")
        parts.append(f"{label}: " + "/".join(f"#{n}" for n in nums))
    return "; ".join(parts)


def render_markdown(report: dict) -> str:
    if report.get("status") == "NO_WITNESS":
        return (
            "# Qwen3.6 Parity Witness Gate\n\n"
            "- Verdict: NO_WITNESS\n"
            f"- Issues: {issue_links_markdown(report.get('issues', ISSUE_LINKS))}\n"
            f"- {report['note']}\n"
        )
    c = report["correctness"]
    s = report["speed"]
    lines = [
        "# Qwen3.6 Parity Witness Gate",
        "",
        f"- Verdict: {'PASS' if report['passed'] else 'FAIL'}",
        f"- Witness: `{report['witness']}`  (commit `{report.get('commit')}`)",
        f"- Issues: {issue_links_markdown(report.get('issues', ISSUE_LINKS))}",
        f"- Correctness: **{c['verdict']}** -- {c['note']}",
        f"  - fak ids `{c['fak_ids']}`  vs oracle `{c['oracle_ids']}`  "
        f"(first divergence index {c['first_divergence_index']}, known {c['known_divergence_index']})",
        f"- Metal hybrid-prefill gate (#71): {'PASS' if report['metal_gate']['pass'] else 'not-yet'} "
        "(recorded, not gated here)",
        f"- Speed (recorded): prefill {s['prefill_tok_s']} tok/s ({s['prefill_ratio']}x bar "
        f"{s['bar_prefill_tok_s']}), decode {s['decode_tok_s']} tok/s ({s['decode_ratio']}x bar "
        f"{s['bar_decode_tok_s']})",
        f"- Scrub: {'clean' if report['scrub']['clean'] else 'LEAK -> ' + ', '.join(report['scrub']['leaks'])}",
    ]
    if report["failures"]:
        lines += ["", "Failures (fail-closed):"]
        lines += [f"- {f}" for f in report["failures"]]
    lines.append("")
    return "\n".join(lines)


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--witness", help="path to a witness JSON; default = newest in "
                    f"{WITNESS_DIR}/")
    ap.add_argument("--require-witness", action="store_true",
                    help="fail closed when no witness is found (the Mac CI lane)")
    ap.add_argument("--min-ratio", type=float, default=0.0,
                    help="if >0, gate fak speed at this fraction of the bar (default 0 = report only)")
    ap.add_argument("--out", help="write the machine-readable gate report JSON here")
    ap.add_argument("--markdown", help="write the markdown report here")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    if args.witness:
        path = Path(args.witness)
    else:
        path = find_latest_witness(ROOT / WITNESS_DIR)
    if path is None:
        report = no_witness_report(args.require_witness)
    else:
        try:
            witness = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            report = {"schema": SCHEMA, "witness": rel(Path(path)), "status": "UNREADABLE",
                      "passed": False, "failures": [f"cannot read witness: {exc}"],
                      "note": str(exc)}
        else:
            report = grade_witness(witness, rel(Path(path)), args.min_ratio)

    if args.out:
        out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown:
        md = ROOT / args.markdown if not Path(args.markdown).is_absolute() else Path(args.markdown)
        md.parent.mkdir(parents=True, exist_ok=True)
        md.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report), end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
