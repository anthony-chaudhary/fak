#!/usr/bin/env python3
"""bench_provenance.py — the closed 4-way benchmark-provenance classifier.

A benchmark catalog run carries a provenance: is its headline number a real
WALL-CLOCK measurement, a closed-form MODELED work floor, a FUNCTIONAL
correctness/witness run that is *not a throughput number at all*, or genuinely
UNKNOWN? The old binary measured/modeled classifier in fresh_status tagged 51 of
55 runs "unknown" because their metadata names neither word — so it understated
the real measurements AND lumped agent-live / parity / load-only runs (which are
not throughput numbers) into the same bucket as a SmolLM2 decode whose tags just
didn't say "measured".

This is the shared, evidence-grounded fix. The taxonomy and the per-signal tables
below were derived from BENCHMARK-AUTHORITY.md's own row language and
adversarially verified against all 55 catalog runs (the derived + functional
classes confirmed 0-correction; the throughput class took 3 corrections — qwen36
"agent" runs are functional, not unknown). The four categories, with the authority
phrase that grounds each:

  measured   — a real wall-clock throughput/latency number (tok/s, decode/prefill
               ms, end-to-end run time, or a ratio of two live wall-clocks).
               Authority: "8.2 min (llama.cpp Metal forward ...) MEASURED, M3 Pro".
  modeled    — a closed-form geometry / projection / arithmetic WORK floor, no
               wall-clock; integer-exact and hardware-independent because computed.
               Authority: "PREFILL-TOKEN WORK FLOOR from a deterministic geometry
               model ... no model, no wall-clock, not 'measured'".
  functional — NOT a throughput number: an agent-live run, a correctness/
               containment/coherence witness, a token-parity verdict, a
               subsystem/permission/api check, a load-only/RSS run, a recall/
               context report, a visual-gen deck, or a CSV export of a derived
               model. Authority: "a containment/coherence witness, not a throughput
               number ... a correctness verdict (PASS / max|Δ|=0), not a wall-clock".
  unknown    — the FAIL-CLOSED residue: no engine, no timed field, no resolving
               tag, no run_id pattern. Never where a real measurement lands by
               default — always preferred over asserting a wrong positive.

The classifier is a deterministic priority ladder (highest-trust signal first):

  Rule 1  artifact engine/backend field (when a run's artifact is local) — the
          most trustworthy signal; a "fak-in-kernel ... decode" / "fak radixbench"
          / "fak model load" engine is a real wall-clock and overrides any tag.
  Rule 2  a populated timed throughput field (peak_tok_per_sec) IS the witness.
  Rule 3  catalog tags — but a FUNCTIONAL tag (parity/agent-live/csv/...) beats a
          co-occurring benchmark-family tag, because such a run is explicitly not
          a throughput number even when it sits in a benchmark family.
  Rule 4  run_id substrings (the last positive layer).
  Rule 5  collision -> the layer naming provenance most precisely wins.
  Rule 6  fail-closed: nothing fired -> unknown.

Used by fresh_status.py (the rollup pane) and bench_catalog.py (which stamps the
verdict onto each catalog run at scan time, so the signal survives even after the
per-run artifacts are gitignored on a non-bench-node clone). Pure-stdlib, no I/O.
"""
from __future__ import annotations

from typing import Any, Iterable

# The closed vocabulary. UNKNOWN is the fail-closed default, never a positive tag.
TAGS = ("measured", "modeled", "functional", "unknown")

# Whether each tag is a throughput/latency NUMBER (vs a witness / residue). Used by
# consumers that want to report "real benchmark numbers" vs "functional runs".
IS_THROUGHPUT_NUMBER = {
    "measured": True, "modeled": False, "functional": False, "unknown": False,
}

# --- Rule 1: artifact engine-field substrings (highest trust) ---------------
# A run's artifact (when local) names the engine that produced it. A live forward /
# decode / timed harness is a real wall-clock; a load-only run is functional.
ENGINE_MEASURED = (
    "fak-in-kernel", "radixbench", "sessionbench", "decode", "prefill",
)
ENGINE_FUNCTIONAL = (
    "fak model load",   # load-only: time-to-load, not a throughput number
)
ENGINE_MODELED = (
    "geometry", "projection", "work floor",
)

# --- Rule 2: a populated timed field is itself the wall-clock witness --------
THROUGHPUT_FIELDS = ("peak_tok_per_sec",)

# --- Rule 3: catalog tags ----------------------------------------------------
# FUNCTIONAL tags are checked FIRST so a parity/csv/agent-live run is pulled out of
# a co-occurring benchmark family (Rule 3 + Rule 5: functional witness beats a
# generic benchmark tag — it is explicitly "not a throughput number").
#
# NOTE on session-benchmark / value-stack (an adversarial-audit fix): these are
# DELIBERATELY NOT functional tags. The authority shows session value-add IS a real
# wall-clock measurement ("Arms B and C run fully live"). So a session run with a
# recorded number is measured (Rule 1 engine `fak sessionbench`, or Rule 2 a timed
# field) — and a session row carrying NO number, no engine, and null metrics is
# genuinely unverifiable, so it must fail-closed to UNKNOWN, NOT be mislabeled
# "functional" (which would falsely assert it is "not a throughput number").
FUNCTIONAL_TAGS = frozenset({
    "agent-live", "parity", "subsystem-checks", "permission-systems",
    "api-host-bridge", "safetensors-load-rss", "recall", "contextq", "csv",
    "visual-gen", "rsi",
})
MODELED_TAGS = frozenset({
    "fan-benchmark", "fanout", "value-sweep", "turn-tax", "fleet",
})
MEASURED_TAGS = frozenset({
    "radix-benchmark", "model-benchmark", "gpu-benchmark", "cuda", "gpu", "ada",
    "cpu-q8-parity", "decode", "cpu-forward", "engine-fak-cpu", "engine-fak-cuda",
    "engine-llama", "headtohead", "kernel", "batch", "phase0",
})

# --- Rule 4: run_id substrings (last positive layer) ------------------------
RUNID_FUNCTIONAL = (
    "agent-live", "parity-", "subsystem-checks", "permission-systems", "api-host",
    "safetensors-load-rss", "recall", "contextq", "visualgen", "surface-smoke",
    "dogfood",
)
RUNID_MODELED = (
    "webvoyager-geometry", "ultra-long-context-floor", "projection", "fanbench",
    "fanout", "value-sweep", "turn-tax", "fleet-writeheavy",
)
RUNID_MEASURED = (
    "cpu-q8-parity", "radixbench-smollm2", "radix-", "smollm2-135m-q8-batch",
    "gpu-qwen2.5", "gcp-g2-l4", "mac-battery",
)

# A bare model-benchmark on an unknown model with NO timed field must NOT be
# claimed measured — the authority requires a traceable wall-clock. A run whose
# only measured-tag is a generic family tag, with null metrics and no engine, is
# held back to unknown (fail-closed). These are the family tags that, ALONE, are
# not enough to assert a measured wall-clock.
WEAK_MEASURED_TAGS = frozenset({"model-benchmark", "gpu-benchmark"})

# Tags that mean "this run is a third-party / no-fak-engine experiment" — they do
# not by themselves discriminate, so a run carrying only these falls through to
# the fail-closed default unless a run_id pattern lifts it out.
NEUTRAL_TAGS = frozenset({"experiment", "qwen36", "multi-agent", "csv-data"})


def _has(substrs: Iterable[str], text: str) -> bool:
    return any(s in text for s in substrs)


def _classify_engines(engines: list[str]) -> str | None:
    """Rule 1: the artifact engine field. Returns a tag or None if undecisive.

    Precedence within the engine blob: modeled (a geometry/projection engine is
    unambiguous) -> measured (ANY live forward/decode/prefill is a real wall-clock,
    and that wins even when a load-only sibling engine — "fak model load" — is also
    present in the same multi-bench run, e.g. mac-battery) -> functional (load-only
    is the verdict only when load is ALL the run did, no real forward beside it).
    """
    blob = " ".join(engines).lower()
    if _has(ENGINE_MODELED, blob):
        return "modeled"
    if _has(ENGINE_MEASURED, blob):
        return "measured"
    if _has(ENGINE_FUNCTIONAL, blob):
        return "functional"
    return None


def _classify_tags(tags: set[str]) -> str | None:
    """Rule 3: catalog tags, functional-first so a witness beats a benchmark family.

    A measured verdict from a WEAK family tag (model-benchmark/gpu-benchmark) is
    deliberately NOT returned here — those need a timed field or a stronger signal
    (handled by the caller's null-metric guard); a strong measured tag (e.g.
    radix-benchmark, cuda, decode) does resolve.
    """
    if tags & FUNCTIONAL_TAGS:
        return "functional"
    if tags & MODELED_TAGS:
        return "modeled"
    strong_measured = (tags & MEASURED_TAGS) - WEAK_MEASURED_TAGS
    if strong_measured:
        return "measured"
    return None


def _classify_runid(run_id: str) -> str | None:
    """Rule 4: run_id substrings, functional-first (same precedence as tags)."""
    low = run_id.lower()
    if _has(RUNID_FUNCTIONAL, low):
        return "functional"
    if _has(RUNID_MODELED, low):
        return "modeled"
    if _has(RUNID_MEASURED, low):
        return "measured"
    return None


def classify(run: dict[str, Any], *, artifact_engines: list[str] | None = None) -> str:
    """Return the provenance tag for one catalog run via the verified priority ladder.

    ``run`` is a catalog run dict (``run_id``, ``model``, ``tags``, ``precision``,
    ``peak_tok_per_sec``, optionally a pre-stamped ``provenance``).
    ``artifact_engines`` is the list of engine strings read from the run's local
    artifacts (Rule 1), if available — bench_catalog passes these at scan time.

    A run that already carries a valid ``provenance`` field (stamped earlier by
    bench_catalog from its then-local artifacts) is trusted: that captured the
    highest-trust signal before the artifacts were gitignored away.
    """
    pre = run.get("provenance")
    if isinstance(pre, str) and pre in TAGS:
        return pre

    # Rule 1: artifact engine field (highest trust).
    engines = artifact_engines if artifact_engines is not None else run.get("artifact_engines")
    if isinstance(engines, list) and engines:
        verdict = _classify_engines([str(e) for e in engines])
        if verdict:
            return verdict

    tags = {str(t).lower() for t in (run.get("tags") or [])}
    run_id = str(run.get("run_id") or "")

    # Rule 2: a populated timed throughput field IS the wall-clock witness.
    for field in THROUGHPUT_FIELDS:
        v = run.get(field)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v:
            return "measured"

    # Rule 3 then Rule 4 (run_id lift), with Rule 5 precedence: a RESOLVING tag is
    # more specific than a run_id substring, so tags win over the run_id lift.
    by_tag = _classify_tags(tags)
    by_runid = _classify_runid(run_id)

    # A resolving tag (functional-first inside _classify_tags) is the most specific
    # signal — it wins. This keeps a `decode`/`cpu-q8-parity` measured run from
    # being dragged functional by a generic `parity-` substring in its run_id.
    if by_tag:
        return by_tag
    # Only when tags DON'T resolve does the run_id lift apply — it rescues the
    # neutral qwen36+experiment residue (surface-smoke / dogfood / agent-live).
    if by_runid:
        return by_runid

    # Rule 6: fail-closed. No engine, no timed field, no resolving tag/run_id.
    return "unknown"


def classify_all(runs: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Provenance histogram over a run list: {measured, modeled, functional, unknown}."""
    counts = {t: 0 for t in TAGS}
    for r in runs:
        counts[classify(r)] += 1
    return counts


def summary_line(counts: dict[str, int]) -> str:
    """The canonical one-line provenance read used by the rollup."""
    return (f"{counts.get('measured', 0)} measured / {counts.get('modeled', 0)} modeled / "
            f"{counts.get('functional', 0)} functional / {counts.get('unknown', 0)} unknown")
