#!/usr/bin/env python3
"""Tests for the 4-way benchmark-provenance classifier.

Two layers:
1. UNIT tests of the priority ladder — each rule (engine field, timed throughput
   field, functional-beats-benchmark tag precedence, run_id lift, fail-closed
   default) and the load-vs-decode engine-precedence edge that mac-battery hit.
2. A GROUND-TRUTH table drawn from the adversarially-verified workflow
   classification (derived + functional classes confirmed 0-correction; throughput
   class corrected: qwen36 'agent' runs are functional). These pin the classifier
   to the authority-grounded verdict so a future signal-table edit that breaks a
   known run is caught.

Run: `python tools/bench_provenance_test.py`  (exit 0 = all pass),
or `python -m pytest tools/bench_provenance_test.py -q`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_provenance as bp  # noqa: E402


def run(tags=None, run_id="", model=None, peak=None, engines=None, provenance=None):
    r = {"run_id": run_id, "model": model, "tags": tags or [], "peak_tok_per_sec": peak}
    if engines is not None:
        r["artifact_engines"] = engines
    if provenance is not None:
        r["provenance"] = provenance
    return r


# --- Rule 1: artifact engine field (highest trust) --------------------------

def test_engine_decode_is_measured() -> None:
    assert bp.classify(run(engines=["fak-in-kernel Q8_0 (pure-Go) decode"])) == "measured"


def test_engine_load_only_is_functional() -> None:
    # A run whose ONLY engine is load-only times the load, not throughput.
    assert bp.classify(run(engines=["fak model load"])) == "functional"


def test_engine_decode_beats_load_in_same_run() -> None:
    # The mac-battery edge: a multi-bench run with BOTH a real decode and a
    # load-only sibling is MEASURED — the live forward wins over the load.
    engines = ["fak-in-kernel Q8_0 ... decode", "fak model load", "fak radixbench"]
    assert bp.classify(run(engines=engines, run_id="mac-battery-20260625")) == "measured"


def test_engine_geometry_is_modeled() -> None:
    assert bp.classify(run(engines=["webbench geometry work floor"])) == "modeled"


def test_engine_overrides_disagreeing_tag() -> None:
    # Engine field is the most trustworthy signal — it overrides a tag that says
    # otherwise (a real decode artifact under an 'experiment' tag is still measured).
    assert bp.classify(run(tags=["experiment"], engines=["fak radixbench decode"])) == "measured"


# --- Rule 2: a populated timed throughput field is the witness --------------

def test_peak_tok_present_is_measured() -> None:
    assert bp.classify(run(tags=["model-benchmark"], peak=31.02)) == "measured"


def test_peak_tok_null_does_not_force_measured() -> None:
    # A weak family tag with NULL metrics must not claim a wall-clock — fail-closed.
    assert bp.classify(run(tags=["model-benchmark"], model="unknown", peak=None)) == "unknown"


def test_peak_tok_zero_is_not_a_witness() -> None:
    assert bp.classify(run(tags=["experiment"], peak=0)) == "unknown"


# --- Rule 3: tags, functional-first -----------------------------------------

def test_functional_tag_beats_benchmark_family() -> None:
    # csv (a functional side-artifact) beats the fanout benchmark family.
    assert bp.classify(run(tags=["fanout", "csv"], model="csv-data")) == "functional"


def test_parity_tag_is_functional() -> None:
    assert bp.classify(run(tags=["parity", "experiment"])) == "functional"


def test_agent_live_tag_is_functional() -> None:
    assert bp.classify(run(tags=["agent-live", "experiment"])) == "functional"


def test_strong_measured_tag_resolves() -> None:
    assert bp.classify(run(tags=["radix-benchmark", "cache-hit-rate"])) == "measured"


def test_modeled_tag_resolves() -> None:
    assert bp.classify(run(tags=["turn-tax", "experiment"])) == "modeled"


def test_value_sweep_csv_is_functional_but_plain_value_sweep_is_modeled() -> None:
    # The subtle pair: the CSV export is functional, the sweep itself is modeled.
    assert bp.classify(run(tags=["value-sweep", "csv"], model="csv-data")) == "functional"
    assert bp.classify(run(tags=["value-sweep", "experiment"])) == "modeled"


# --- Rule 4: run_id lift (functional run_id rescues neutral residue) --------

def test_surface_smoke_runid_is_functional() -> None:
    # A bare qwen36+experiment run would be unknown, but a surface-smoke run_id
    # lifts it to functional (authority: '4/4 surfaces PASS' is a witness).
    r = run(tags=["qwen36", "experiment"], run_id="workstation-a-qwen36-surface-smoke-20260619")
    assert bp.classify(r) == "functional"


def test_dogfood_runid_is_functional() -> None:
    r = run(tags=["agent-live", "experiment"], run_id="node-agent-live-dogfood-claude-probe")
    assert bp.classify(r) == "functional"


# --- Rule 6: fail-closed default --------------------------------------------

def test_bare_experiment_is_unknown() -> None:
    assert bp.classify(run(tags=["experiment"], model="experiment")) == "unknown"


def test_third_party_qwen36_is_unknown() -> None:
    # A third-party ollama/llama.cpp qwen36 run with no fak engine, null metrics —
    # correctly fail-closed (we cannot claim it as our measurement).
    r = run(tags=["qwen36", "experiment"], model="qwen3.6-27b",
            run_id="workstation-a-qwen36-local-ollama-qwen25--20260619")
    assert bp.classify(r) == "unknown"


def test_empty_run_is_unknown() -> None:
    assert bp.classify(run()) == "unknown"


# --- pre-stamped provenance is trusted --------------------------------------

def test_prestamped_provenance_is_trusted() -> None:
    # bench_catalog stamps the verdict from then-local artifacts; a later
    # gitignored-artifact clone trusts that capture.
    assert bp.classify(run(tags=["experiment"], provenance="measured")) == "measured"


def test_invalid_prestamped_provenance_is_ignored() -> None:
    # A garbage provenance value falls through to live classification, not trusted.
    assert bp.classify(run(tags=["parity"], provenance="bogus")) == "functional"


# --- histogram + summary helpers --------------------------------------------

def test_classify_all_and_summary() -> None:
    runs = [
        run(tags=["radix-benchmark"]),            # measured
        run(peak=38.0, tags=["model-benchmark"]),  # measured
        run(tags=["turn-tax"]),                    # modeled
        run(tags=["parity"]),                      # functional
        run(tags=["agent-live"]),                  # functional
        run(tags=["experiment"]),                  # unknown
    ]
    counts = bp.classify_all(runs)
    assert counts == {"measured": 2, "modeled": 1, "functional": 2, "unknown": 1}
    line = bp.summary_line(counts)
    assert line == "2 measured / 1 modeled / 2 functional / 1 unknown"


# --- GROUND TRUTH: the adversarially-verified workflow classification --------
# Each (run_id-substring, tags, expected) is a real catalog run the workflow
# classified and a verifier confirmed. A signal-table edit that breaks one of
# these is a regression against the authority-grounded verdict.

GROUND_TRUTH = [
    # measured: a wall-clock exists
    ("radixbench-smollm2-135m", ["radix-benchmark", "cache-hit-rate"], "measured"),
    ("cpu-q8-parity-qwen25-1.5b", ["model-benchmark", "cpu-q8-parity", "decode"], "measured"),
    ("radix-qwen25-1.5b-uncontended", ["radix-benchmark"], "measured"),
    ("gpu-qwen2.5-3b", ["gpu-benchmark", "cuda"], "measured"),
    # modeled: a closed-form work floor
    ("fleet-writeheavy-50x50", ["fleet"], "modeled"),
    ("turn-tax-turntax-happy", ["turn-tax", "experiment"], "modeled"),
    ("fanbench-research-goal", ["fan-benchmark", "multi-agent"], "modeled"),
    ("value-sweep-value-sweep", ["value-sweep", "experiment"], "modeled"),
    # functional: not a throughput number
    ("agent-live-offline-mock", ["agent-live", "experiment"], "functional"),
    ("recall-recall-report", ["recall", "experiment"], "functional"),
    ("parity-reference-front", ["parity", "experiment"], "functional"),
    ("api-host-bridge-api-host-roster", ["api-host-bridge", "experiment"], "functional"),
    ("permission-systems", ["permission-systems", "experiment"], "functional"),
    ("safetensors-load-rss", ["safetensors-load-rss", "experiment"], "functional"),
    ("subsystem-checks-latest-tooling", ["subsystem-checks", "experiment"], "functional"),
    ("fanout-csv-fanbench", ["fanout", "csv"], "functional"),
    ("value-sweep-csv-value-sweep", ["value-sweep", "csv"], "functional"),
    ("qwen36-surface-smoke", ["qwen36", "experiment"], "functional"),
    ("contextq-context-query", ["contextq", "experiment"], "functional"),
    ("visualgen", ["visual-gen", "rsi"], "functional"),
    # unknown: fail-closed residue (third-party / bare)
    ("workstation-a-unknown-unknown", ["model-benchmark"], "unknown"),
    ("qwen36-local-ollama", ["qwen36", "experiment"], "unknown"),
    ("qwen36-native-gguf-q8-hybri", ["qwen36", "experiment"], "unknown"),
    ("qwen36-llamacpp-vulkan", ["qwen36", "experiment"], "unknown"),
]


def test_ground_truth_table() -> None:
    failures = []
    for run_id, tags, expected in GROUND_TRUTH:
        got = bp.classify(run(tags=tags, run_id=run_id))
        if got != expected:
            failures.append(f"{run_id} tags={tags}: expected {expected}, got {got}")
    assert not failures, "ground-truth regressions:\n  " + "\n  ".join(failures)


def _run_all() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"ok   {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
