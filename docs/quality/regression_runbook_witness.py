#!/usr/bin/env python3
"""Hermetic captured-proof harness for the output-quality regression runbook.

This is the WITNESS for docs/quality/output-quality-regression-runbook.md
(issue #4587, epic #4509). It is a *tabletop*, not a production incident: a
tiny, deterministic, stdlib-only stand-in for an engine decode path over a
frozen golden set, wired so a planted representative defect makes it FAIL and
removing the defect makes it PASS -- in a clean, independently replayed
environment (same file, no network, no deps, no host state).

It exercises the runbook method end to end:

    detect  -> compare produced token stream against the frozen golden
    freeze  -> the goldens below are the frozen reference (seed + oracle)
    replay  -> re-run deterministically; identical inputs -> identical output
    classify-> label the first divergence (tokenizer / normalize / decode)
    bisect  -> report the FIRST actionable divergence (case, then position)
    mitigate/fix/recover -> drop the defect flag; re-run PASSES

Contract (acceptance criteria of #4587):
  * A failure identifies the *first* actionable divergence and emits a
    scrubbed replay artifact carrying full case provenance.
  * Missing or inconclusive evidence is NEVER a pass (exit 2, not 0).
  * Every case carries model, tokenizer, engine/backend, seed-or-oracle,
    code revision, and tolerance/baseline provenance.
  * Every case is assigned a tier (pr | nightly | release) with a cost.

Exit codes: 0 = PASS (all cases match), 1 = FAIL (actionable divergence),
2 = INCONCLUSIVE (missing/empty evidence -- never a pass).

Usage:
    python regression_runbook_witness.py                      # clean -> PASS (0)
    python regression_runbook_witness.py --defect bos-drop    # FAIL (1)
    python regression_runbook_witness.py --defect tie-break   # FAIL (1)
    python regression_runbook_witness.py --tier pr            # only PR-tier cases
    python regression_runbook_witness.py --selftest           # asserts FAIL-then-PASS
    python regression_runbook_witness.py --artifact out.json  # write replay artifact
"""
from __future__ import annotations

import argparse
import json
import sys

# ----------------------------------------------------------------------------
# Frozen reference (freeze phase): a deterministic oracle. `seed` pins the
# decode; `baseline` names where the golden came from so a reader can audit its
# provenance. Kept tiny on purpose -- the point is the METHOD, not coverage.
# ----------------------------------------------------------------------------
VOCAB = {
    "<bos>": 1, "the": 2, "cat": 3, "sat": 4, "on": 5, "mat": 6,
    "dog": 7, "ran": 8, "fast": 9, "<eos>": 10,
}
REVISION = "witness-v1"  # bump when the oracle itself changes

CASES = [
    {
        "id": "decode-greedy-basic",
        "prompt": "the cat sat on mat",
        "model": "toy-decoder-8", "tokenizer": "toy-bpe-v1",
        "engine": "reference-cpu", "seed": 0, "oracle": "exact-token-match",
        "tolerance": "exact", "baseline": "frozen@witness-v1",
        "tier": "pr", "cost": "sub-second, no hardware",
        "expected": [1, 2, 3, 4, 5, 6, 10],
    },
    {
        "id": "decode-greedy-alt",
        "prompt": "the dog ran fast",
        "model": "toy-decoder-8", "tokenizer": "toy-bpe-v1",
        "engine": "reference-cpu", "seed": 0, "oracle": "exact-token-match",
        "tolerance": "exact", "baseline": "frozen@witness-v1",
        "tier": "pr", "cost": "sub-second, no hardware",
        "expected": [1, 2, 7, 8, 9, 10],
    },
    {
        "id": "decode-tiebreak-guard",
        "prompt": "the cat ran fast",
        "model": "toy-decoder-8", "tokenizer": "toy-bpe-v1",
        "engine": "reference-cpu", "seed": 0, "oracle": "argmax-lowest-id-tiebreak",
        "tolerance": "exact", "baseline": "frozen@witness-v1",
        "tier": "nightly", "cost": "sub-second, no hardware",
        "expected": [1, 2, 3, 8, 9, 10],
    },
]

DEFECTS = {
    "none": "the fixed build",
    "bos-drop": "tokenizer normalization drops the leading <bos> (a real "
                "class: prompt-template / special-token regression)",
    "tie-break": "decode argmax tie-break flips to highest-id on ties (a real "
                 "class: sampler/logits-processor regression)",
}


def tokenize(prompt: str, *, defect: str) -> list[int]:
    """tokenize -> normalize. The bos-drop defect corrupts the normalize step."""
    ids = [VOCAB["<bos>"]] + [VOCAB[w] for w in prompt.split()]
    if defect == "bos-drop":
        ids = [i for i in ids if i != VOCAB["<bos>"]]  # planted: BOS lost
    return ids


def decode(ids: list[int], *, defect: str) -> list[int]:
    """decode step: append <eos>. The tie-break defect perturbs a tie position.

    The toy tie is at the third emitted position of `decode-tiebreak-guard`:
    the fixed oracle keeps the lower id (`cat`=3); the defect flips it (`dog`=7).
    """
    out = list(ids)
    if defect == "tie-break" and out[:3] == [1, 2, 3]:
        # planted: a tie between id 3 (cat) and id 7 (dog) resolves the wrong way
        out[2] = VOCAB["dog"]
    out.append(VOCAB["<eos>"])
    return out


def produce(case: dict, *, defect: str) -> list[int]:
    return decode(tokenize(case["prompt"], defect=defect), defect=defect)


def first_divergence(expected: list[int], got: list[int]) -> tuple[int, int, int] | None:
    """Return (position, expected_id, got_id) of the first mismatch, or None."""
    n = max(len(expected), len(got))
    for pos in range(n):
        e = expected[pos] if pos < len(expected) else None
        g = got[pos] if pos < len(got) else None
        if e != g:
            return (pos, e, g)
    return None


def classify(pos: int, exp_id, got_id) -> str:
    """Label the divergence class -- what a first responder needs to route it."""
    if got_id is None or exp_id is None:
        return "length-divergence (truncated or over-generated stream)"
    if pos == 0:
        return "tokenizer/normalize (stream misaligned from position 0)"
    return "decode/sampler (in-stream token substitution)"


def run(cases: list[dict], *, defect: str, artifact_path: str | None) -> int:
    if not cases:
        # never-pass-on-missing-evidence: no goldens is INCONCLUSIVE, not PASS.
        print("INCONCLUSIVE: no golden cases to evaluate -> not a pass", file=sys.stderr)
        return 2

    replay = {
        "schema": "fak-quality-regression-witness/1",
        "revision": REVISION,
        "injected_defect": defect,
        "injected_defect_desc": DEFECTS.get(defect, "unknown"),
        "cases_evaluated": len(cases),
        "result": None,
        "first_actionable_divergence": None,
        "provenance": None,
    }

    for case in cases:
        expected = case.get("expected")
        if not expected:  # a case with no oracle can never be scored PASS
            print(f"INCONCLUSIVE: case {case['id']} has no frozen golden -> not a pass",
                  file=sys.stderr)
            replay["result"] = "inconclusive"
            _emit(replay, artifact_path)
            return 2
        got = produce(case, defect=defect)
        div = first_divergence(expected, got)
        if div is not None:
            pos, exp_id, got_id = div
            replay["result"] = "fail"
            replay["first_actionable_divergence"] = {
                "case": case["id"], "position": pos,
                "expected_id": exp_id, "got_id": got_id,
                "classification": classify(pos, exp_id, got_id),
            }
            # scrubbed provenance record (no host paths, no operator identity)
            replay["provenance"] = {
                k: case[k] for k in
                ("model", "tokenizer", "engine", "seed", "oracle",
                 "tolerance", "baseline", "tier", "cost")
            }
            replay["provenance"]["code_revision"] = REVISION
            _emit(replay, artifact_path)
            print(f"FAIL: first actionable divergence in case '{case['id']}' "
                  f"at position {pos}: expected {exp_id!r}, got {got_id!r} "
                  f"[{replay['first_actionable_divergence']['classification']}]")
            print(f"  tier={case['tier']} cost={case['cost']} "
                  f"baseline={case['baseline']} rev={REVISION}")
            return 1

    replay["result"] = "pass"
    _emit(replay, artifact_path)
    print(f"PASS: {len(cases)} case(s) match the frozen golden "
          f"(defect={defect}, rev={REVISION})")
    return 0


def _emit(replay: dict, artifact_path: str | None) -> None:
    blob = json.dumps(replay, indent=2, sort_keys=True)
    if artifact_path:
        with open(artifact_path, "w", encoding="utf-8") as fh:
            fh.write(blob + "\n")


def _selftest() -> int:
    """Single-command proof: the planted defect FAILS, the fix PASSES."""
    fail = run(CASES, defect="bos-drop", artifact_path=None)
    tie = run(CASES, defect="tie-break", artifact_path=None)
    ok = run(CASES, defect="none", artifact_path=None)
    problems = []
    if fail != 1:
        problems.append(f"bos-drop expected exit 1, got {fail}")
    if tie != 1:
        problems.append(f"tie-break expected exit 1, got {tie}")
    if ok != 0:
        problems.append(f"clean build expected exit 0, got {ok}")
    if run([], defect="none", artifact_path=None) != 2:
        problems.append("empty evidence must be INCONCLUSIVE (exit 2), never PASS")
    if problems:
        print("SELFTEST FAILED:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    print("SELFTEST OK: planted defect FAILS (exit 1) and the fix PASSES (exit 0); "
          "empty evidence is INCONCLUSIVE (exit 2), never a pass.")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--defect", choices=sorted(DEFECTS), default="none",
                    help="inject a representative planted defect (default: none = fixed build)")
    ap.add_argument("--tier", choices=("pr", "nightly", "release"), default=None,
                    help="restrict to one validation tier")
    ap.add_argument("--artifact", default=None,
                    help="write the scrubbed replay artifact JSON to this path")
    ap.add_argument("--selftest", action="store_true",
                    help="assert the fail-then-pass contract in one command")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()

    cases = CASES if args.tier is None else [c for c in CASES if c["tier"] == args.tier]
    return run(cases, defect=args.defect, artifact_path=args.artifact)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
