#!/usr/bin/env python3
"""Tests for fanout_longctx_probe — the #431 no-silent-extrapolation contract.

The probe (`tools/fanout_longctx_probe.py`) is the measured-path half of GitHub
issue #431: for each fanbench prefix-scale point (P = 262144 / 524288 / 1048576)
it either MEASURES a real long-context prefill or records the structured CEILING
that stopped it — and it must NEVER dress the modeled token economics up as a
wall-clock measurement. That guarantee was shipped but unguarded; this file pins
it as an executable invariant.

The load-bearing checks:
  * a SKIPPED path carries no wall-clock number (ttft_ms / prefill_ms /
    measured_kv_bytes stay null) — the literal "records ceilings instead of
    silently extrapolating" acceptance bullet;
  * a checkpoint that *admits* the context length is still NOT_RUN, never a
    fabricated "measured" — the probe does not invent numbers it cannot take;
  * KV-cache memory is sized from the documented transformer geometry (a
    quantity, not a guess), and the memory ceiling is recorded when it exceeds
    host RAM;
  * the artifact reproduces byte-for-byte from identical inputs (the determinism
    the doc names as the gate);
and a live regression sentinel over the REAL checked-in artifacts so a hand-edited
artifact that injects a fake measurement under a `skipped` status reddens the gate.

Run: `python tools/fanout_longctx_probe_test.py`  (exit 0 = all pass),
or `python -m pytest tools/fanout_longctx_probe_test.py -q`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fanout_longctx_probe as probe  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
PSCALE_DIR = REPO_ROOT / "experiments" / "fanout" / "pscale"
TARGET_PREFIXES = set(probe.TARGET_PREFIXES)
VALID_OVERALL = {"MEASURED", "SKIPPED_NO_LONGCTX_PATH"}

# Fixtures: host facts and discovery results, supplied directly so the unit tests
# never touch this machine's real RAM/checkpoints/servers (those are exercised by
# the live smoke at the bottom).
BIG_HOST = {
    "platform": "linux",
    "cpu_count": 32,
    "total_ram_bytes": 256 * 2 ** 30,
    "total_ram_gib": 256.0,
    "dedicated_gpu_vram_bytes": 0,
}
TINY_HOST = {**BIG_HOST, "total_ram_bytes": 1 * 2 ** 30, "total_ram_gib": 1.0}
CKPT_SHORT = {"context_tokens": 512, "config": "internal/model/.cache/x/config.json"}
CKPT_LONG = {"context_tokens": 2_000_000, "config": "internal/model/.cache/big/config.json"}
LLAMA_PRESENT_NO_GGUF = {"binaries": {"llama-cli": "/usr/bin/llama-cli"}, "gguf_count": 0}
LLAMA_ABSENT = {"binaries": {}, "gguf_count": 0}
SERVER_DOWN = (False, "none of 127.0.0.1:8000, 127.0.0.1:30000")


def _wall_clock_is_null(path: dict) -> bool:
    return (path["ttft_ms"] is None
            and path["prefill_ms"] is None
            and path["measured_kv_bytes"] is None)


def test_kv_cache_bytes_matches_documented_geometry():
    geom = probe.MODELS["qwen25-7b"]
    # 2 (K,V) * 28 layers * 4 kv-heads * 128 head-dim * P * 2 (fp16) bytes.
    assert probe.kv_cache_bytes(geom, 262144) == 2 * 28 * 4 * 128 * 262144 * 2
    assert probe.kv_cache_bytes(geom, 262144) == 15_032_385_536  # == 14.0 GiB
    assert probe.kv_cache_bytes(geom, 1048576) == 60_129_542_144  # == 56.0 GiB, 4x


def test_kv_bytes_per_token_is_prefix_independent():
    geom = probe.MODELS["qwen25-7b"]
    per_token = {probe.kv_cache_bytes(geom, p) // p for p in (262144, 524288, 1048576)}
    assert per_token == {57344}  # one width per token, regardless of P


def test_skip_never_carries_a_wall_clock_number():
    s = probe._skip("some-path", "SOME_REASON", "why")
    assert s["status"] == "skipped"
    assert _wall_clock_is_null(s)


def test_no_qualifying_path_is_skipped_with_null_wall_clock():
    # The mirror of this reference host: short checkpoint, llama present but no
    # long-context GGUF, no server. This is the #431 acceptance case.
    art = probe.probe_prefix(262144, "qwen25-7b", BIG_HOST, CKPT_SHORT,
                             LLAMA_PRESENT_NO_GGUF, SERVER_DOWN)
    assert art["overall"] == "SKIPPED_NO_LONGCTX_PATH"
    assert {p["status"] for p in art["paths"]} == {"skipped"}
    # Not one skipped path may smuggle in a wall-clock number.
    assert all(_wall_clock_is_null(p) for p in art["paths"])

    by_path = {p["path"]: p for p in art["paths"]}
    assert by_path["fak-in-kernel"]["reason"] == "CONTEXT_CEILING"
    assert by_path["fak-in-kernel"]["context_ceiling_tokens"] == 512
    assert by_path["llama.cpp"]["reason"] == "MODEL_UNAVAILABLE"
    assert by_path["vllm-sglang"]["reason"] == "SERVER_UNAVAILABLE"
    assert "No silent extrapolation" in art["note"]


def test_admitting_checkpoint_is_not_run_never_fabricated_measurement():
    # Even when the checkpoint context >= P, the probe records NOT_RUN — it does
    # NOT invent a measurement it cannot actually take. The honesty hinge.
    art = probe.probe_prefix(262144, "qwen25-7b", BIG_HOST, CKPT_LONG,
                             LLAMA_ABSENT, SERVER_DOWN)
    by_path = {p["path"]: p for p in art["paths"]}
    assert by_path["fak-in-kernel"]["reason"] == "NOT_RUN"
    assert by_path["fak-in-kernel"]["status"] == "skipped"
    assert _wall_clock_is_null(by_path["fak-in-kernel"])
    # No path measured anything, so the overall verdict stays a skip.
    assert art["overall"] == "SKIPPED_NO_LONGCTX_PATH"


def test_memory_ceiling_recorded_when_kv_exceeds_host_ram():
    # 56 GiB KV at P=1048576 against a 1 GiB host => the memory ceiling bites.
    art = probe.probe_prefix(1048576, "qwen25-7b", TINY_HOST, CKPT_SHORT,
                             LLAMA_PRESENT_NO_GGUF, SERVER_DOWN)
    assert art["kv_fits_host_ram"] is False
    llama = next(p for p in art["paths"] if p["path"] == "llama.cpp")
    assert "EXCEEDS" in llama["detail"]


def test_llama_backend_absent_is_distinct_reason():
    art = probe.probe_prefix(262144, "qwen25-7b", BIG_HOST, CKPT_SHORT,
                             LLAMA_ABSENT, SERVER_DOWN)
    llama = next(p for p in art["paths"] if p["path"] == "llama.cpp")
    assert llama["reason"] == "BACKEND_UNAVAILABLE"


def test_artifact_is_deterministic_for_identical_inputs():
    a = probe.probe_prefix(524288, "qwen25-7b", BIG_HOST, CKPT_SHORT,
                           LLAMA_PRESENT_NO_GGUF, SERVER_DOWN)
    b = probe.probe_prefix(524288, "qwen25-7b", BIG_HOST, CKPT_SHORT,
                           LLAMA_PRESENT_NO_GGUF, SERVER_DOWN)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_artifact_has_required_schema_keys():
    art = probe.probe_prefix(262144, "qwen25-7b", BIG_HOST, CKPT_SHORT,
                             LLAMA_PRESENT_NO_GGUF, SERVER_DOWN)
    required = {"schema", "prefix_tokens", "reference_model", "model_geometry",
               "kv_cache_bytes", "kv_cache_gib", "kv_fits_host_ram", "host",
               "paths", "overall", "note", "generated_by"}
    assert required <= set(art)
    assert art["schema"] == "fanout-longctx-measure/1"


# --- Live regression sentinel over the REAL checked-in #431 artifacts. ---

def test_committed_artifacts_exist_for_every_target_prefix():
    for p in probe.TARGET_PREFIXES:
        assert (PSCALE_DIR / f"longctx-measure-p{p}.json").is_file(), p
    assert (PSCALE_DIR / "longctx-measure.csv").is_file()


def test_committed_artifacts_honor_the_no_extrapolation_contract():
    for art_path in sorted(PSCALE_DIR.glob("longctx-measure-p*.json")):
        art = json.loads(art_path.read_text(encoding="utf-8"))
        assert art["schema"] == "fanout-longctx-measure/1", art_path.name
        assert art["overall"] in VALID_OVERALL, art_path.name
        assert art["prefix_tokens"] in TARGET_PREFIXES, art_path.name
        assert art["kv_cache_gib"] > 0, art_path.name
        assert isinstance(art["kv_fits_host_ram"], bool), art_path.name
        for path in art["paths"]:
            # The sentinel: a skipped path must never carry a wall-clock number,
            # so a hand-edited artifact that injects a fake measurement reddens.
            if path["status"] == "skipped":
                assert _wall_clock_is_null(path), (art_path.name, path["path"])
        # A SKIPPED overall must have NO measured path hiding inside it.
        if art["overall"] == "SKIPPED_NO_LONGCTX_PATH":
            assert all(p["status"] == "skipped" for p in art["paths"]), art_path.name


def test_committed_csv_separates_wall_clock_columns():
    header = (PSCALE_DIR / "longctx-measure.csv").read_text(
        encoding="utf-8").splitlines()[0].split(",")
    for col in ("prefix_tokens", "reference_model", "path", "status", "reason",
                "ttft_ms", "prefill_ms"):
        assert col in header, col


def main() -> int:
    failures: list[str] = []

    def check(name: str, fn) -> None:
        try:
            fn()
        except AssertionError as exc:
            failures.append(f"{name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: unexpected {type(exc).__name__}: {exc}")

    tests = {n: f for n, f in globals().items()
             if n.startswith("test_") and callable(f)}
    for name, fn in tests.items():
        check(name, fn)

    if failures:
        print(f"FAIL ({len(failures)}/{len(tests)}):")
        for f in failures:
            print("  -", f)
        return 1
    print(f"ok ({len(tests)} tests)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
