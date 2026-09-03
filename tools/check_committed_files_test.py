#!/usr/bin/env python3
"""Tests for the file-admission gate (`check_committed_files.py`).

Focuses on the PRIVATE_ONLY guard — the public-tree enforcement that keeps the
operator's private lab GPU-server *connection* subsystem (the private control bridge
client + its orchestrator) out of the public repo. This is the durable backstop
for the leak that put internal/dgxbridge + cmd/dgxbridge into public once: the
scrubber's export-time DELETE_PATHS never run as a public gate, and connection
code with placeholder ids passes the secret-needle scan, so a PATH rule is the
only thing that catches it. Closes with a LIVE regression assertion that the real
tracked tree carries no private-only path.

Run: `python tools/check_committed_files_test.py`  (exit 0 = all pass),
or `python -m pytest tools/check_committed_files_test.py -q`.
"""
from __future__ import annotations

import subprocess
import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_committed_files as cc  # noqa: E402

ROOT = str(Path(__file__).resolve().parent.parent)
MAX = cc.DEFAULT_MAX_BYTES


# --- PRIVATE_ONLY: the connection subsystem is refused ----------------------

def test_dgxbridge_client_refused() -> None:
    # The exact paths that leaked once — the bridge CLI and its internal pkg.
    assert cc._classify("cmd/dgxbridge/main.go", ROOT, MAX) is not None
    assert cc._classify("internal/dgxbridge/rpc.go", ROOT, MAX) is not None


def test_dgx_bench_orchestrator_refused() -> None:
    # cmd/dgxbench carries the `dgx` token under cmd/ — same private connection class.
    r = cc._classify("cmd/dgxbench/main.go", ROOT, MAX)
    assert r is not None and "private" in r.lower()


def test_future_dedicated_connection_tool_refused() -> None:
    # A NEW dedicated connection tool (e.g. cmd/dgxconn) is covered without an edit:
    # the rule keys on the `dgx` token under cmd//internal/, not a hard-coded name.
    assert cc._classify("cmd/dgxconn/main.go", ROOT, MAX) is not None
    assert cc._classify("internal/dgxlink/session.go", ROOT, MAX) is not None


def test_slackgc_sibling_refused() -> None:
    assert cc._classify("cmd/slackgc/main.go", ROOT, MAX) is not None


def test_future_slack_control_bridge_refused() -> None:
    assert cc._classify("cmd/slackbridge/main.go", ROOT, MAX) is not None
    assert cc._classify("internal/slackcontrol/client.go", ROOT, MAX) is not None


def test_sunset_python_bench_slack_refused() -> None:
    assert cc._classify("tools/bench_slack.py", ROOT, MAX) is not None
    assert cc._classify("tools/bench_slack_test.py", ROOT, MAX) is not None


# --- scope boundaries: only the connection subsystem, nothing legit ---------

def test_normal_packages_allowed() -> None:
    # Ordinary public packages must not trip the private-only rule.
    for p in ("cmd/fak/main.go", "internal/agent/agent.go",
              "cmd/loadgen/main.go", "internal/gateway/gateway.go"):
        assert cc._classify(p, ROOT, MAX) is None, p


def test_dgx_token_outside_cmd_internal_is_not_private_only() -> None:
    # The guard is deliberately scoped to the CONNECTION subsystem (cmd//internal/).
    # The lab automation under tools/*dgx* and the dgx result dirs are a separate,
    # not-yet-approved relocation, so they must NOT be classified private-only here
    # (else CI would go red on paths still intentionally present in the tree).
    assert not any(rx.search("tools/dgx_pure_kernel_bench.sh") for rx, _ in cc.PRIVATE_ONLY)
    assert not any(rx.search("experiments/qwen36/gpu-server-r4-20260622/compare.json")
                   for rx, _ in cc.PRIVATE_ONLY)


def test_generic_slack_names_are_not_private_only() -> None:
    assert cc._classify("internal/agent/send_slack_message.go", ROOT, MAX) is None
    assert cc._classify("examples/slack-policy.json", ROOT, MAX) is None


def test_token_must_be_in_first_component_not_substring_elsewhere() -> None:
    # `dgx` only triggers as the package-dir token, not as a stray substring in a
    # deeper filename under a normal package.
    assert cc._classify("internal/agent/dgx_notes.go", ROOT, MAX) is None


# --- SECRET_FILES: credentials / keys are refused ---------------------------

def test_sa_key_refused() -> None:
    # The exact path tools/create_gcp_admin_sa.sh writes (must NEVER reach git):
    # refused — here by the secrets/ rule, which wins first on the path.
    assert cc._classify("secrets/gcp/fak-admin-proj.sa.json", ROOT, MAX) is not None
    # A *.sa.json OUTSIDE secrets/ is refused by the key rule (message names the key).
    r = cc._classify("deploy/fak-admin-proj.sa.json", ROOT, MAX)
    assert r is not None and "key" in r.lower()
    # the -sa-key/-gcp-key JSON conventions are refused too.
    assert cc._classify("deploy/prod-sa-key.json", ROOT, MAX) is not None
    assert cc._classify("x-gcp-key.json", ROOT, MAX) is not None


def test_secrets_dir_refused() -> None:
    # Anything under a secrets/ dir, at root or nested.
    assert cc._classify("secrets/anything.txt", ROOT, MAX) is not None
    assert cc._classify("internal/foo/secrets/bar.json", ROOT, MAX) is not None


def test_ordinary_json_not_secret() -> None:
    # A normal config/data json must NOT trip the SECRET rule (no false positives).
    for p in ("internal/gateway/config.json", "experiments/x/report.json",
              "fak/testdata/sample.json", "tools/bench_nodes.example.json"):
        assert not any(rx.search(p) for rx, _ in cc.SECRET_FILES), p


def test_root_coverage_refused_as_hard_junk() -> None:
    r = cc._classify("coverage", ROOT, MAX)
    assert r is not None and "build artifact" in r.lower()


# --- oversized-blob cap: 25 MiB default + non-positive clamp ----------------

def test_max_bytes_default_is_25_mib() -> None:
    # The cap is 25 MiB, in lockstep with the Go twin (gate_fileadmission.go's
    # FAK_MAX_FILE_BYTES default). If one side moves, this pins the other.
    assert cc.DEFAULT_MAX_BYTES == 25 * 1024 * 1024


def test_nonpositive_max_bytes_matches_default() -> None:
    # A non-positive --max-bytes must fall back to the default, NOT become a 0-byte
    # cap that refuses every file — matching the Go gate's gateEnvInt(<=0 -> default).
    # Proven behaviourally and independent of tree state: `--max-bytes 0` yields the
    # SAME exit code as the default cap. Without the clamp, 0 would refuse the whole
    # tracked tree and the codes would differ.
    exe = [sys.executable,
           str(Path(__file__).resolve().parent / "check_committed_files.py"),
           "--audit-tree"]
    base = subprocess.run(exe, capture_output=True, text=True, cwd=ROOT)
    zeroed = subprocess.run(exe + ["--max-bytes", "0"], capture_output=True, text=True, cwd=ROOT)
    assert zeroed.returncode == base.returncode, (
        "non-positive --max-bytes should behave identically to the default cap; "
        f"base rc={base.returncode} zeroed rc={zeroed.returncode}\n{zeroed.stdout}\n{zeroed.stderr}")


# --- BY-MACHINE private-by-default (STAGED-ONLY) ----------------------------

def test_generated_claude_control_artifact_refused() -> None:
    with tempfile.TemporaryDirectory() as td:
        for rel in (
            ".claude/goal-prompts/frontdoor-6037-recovery.md",
            ".claude/goal-prompts/resfleet-6557.md",
            ".claude/goal-prompts/resolve-issue-5898-continuation.md",
        ):
            path = Path(td, rel)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("worker fuel", encoding="utf-8")
            reason = cc._classify(rel, td, cc.DEFAULT_MAX_BYTES)
            assert "generated one-run .claude goal prompt" in reason


def test_reusable_claude_goal_prompt_allowed() -> None:
    with tempfile.TemporaryDirectory() as td:
        rel = ".claude/goal-prompts/resolve-top-issue-witnessed.md"
        path = Path(td, rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("reusable project prompt", encoding="utf-8")
        assert cc._classify(rel, td, cc.DEFAULT_MAX_BYTES) is None


def test_bymachine_raw_drop_refused_by_helper() -> None:
    # A raw per-machine run drop is caught by the staged-only helper (root-hoisted form).
    assert cc._is_private_bymachine_addition(
        "experiments/benchmark/runs/by-machine/node-macos-a/20260718-x/score.json")
    # ...and the fak/-nested superrepo form.
    assert cc._is_private_bymachine_addition(
        "fak/experiments/benchmark/runs/by-machine/node-macos-a/20260718-x/score.json")


def test_bymachine_dgx_addition_also_caught() -> None:
    # A dgx* machine drop under by-machine/ is the historically-leaky class — caught too,
    # in BOTH layouts (the ignore rule that replaced the dgx-only glob was too narrow).
    # The machine names here are SYNTHETIC on purpose: this is a public tree, and a fixture
    # naming a real fleet node would itself be the leak the gate exists to stop (PUBLIC_LEAK
    # refuses the bare dgxN alias). What the rule matches is the by-machine/ prefix, not the
    # machine name, so a made-up name exercises it exactly as well as a real one.
    assert cc._is_private_bymachine_addition(
        "experiments/benchmark/runs/by-machine/dgx-a100-01/20260718-run/witness.json")
    assert cc._is_private_bymachine_addition(
        "fak/experiments/benchmark/runs/by-machine/dgx-a100-02/20260718-run/manifest.json")


def test_bymachine_helper_does_not_overfire() -> None:
    # A normal source path is NOT a by-machine drop (helper returns False).
    assert not cc._is_private_bymachine_addition("internal/foo/bar.go")
    # The tracked aggregate catalog lives OUTSIDE by-machine/ and stays committable.
    assert not cc._is_private_bymachine_addition("experiments/benchmark/catalog.json")
    # A sibling runs/ dir that is NOT by-machine/ is untouched.
    assert not cc._is_private_bymachine_addition("experiments/benchmark/runs/summary.md")


def test_bymachine_not_in_classify_so_tree_mode_unaffected() -> None:
    # CRITICAL: the by-machine rule is STAGED-ONLY. _classify (which runs in BOTH
    # --audit-staged and --audit-tree) must return None for a by-machine path, so the ~50
    # grandfathered evidence artifacts already tracked under by-machine/ keep --audit-tree
    # green. If someone folded the rule into _classify/PRIVATE_ONLY, this goes red.
    p = "experiments/benchmark/runs/by-machine/node-macos-a/20260622-q4k/score.json"
    assert cc._classify(p, ROOT, MAX) is None
    assert not any(rx.search(p) for rx, _ in cc.PRIVATE_ONLY)
    # And the fak/-nested form is likewise invisible to the tree-mode classifier.
    assert cc._classify("fak/" + p, ROOT, MAX) is None


def test_gitignore_keeps_both_bymachine_rules() -> None:
    # The private-by-default DEFAULT is the whole-tree .gitignore rule (62bed967e). The
    # commit-time helper is a backstop, NOT a replacement: if the ignore lines were silently
    # removed, raw drops would flow back in via a plain `git add`. Pin BOTH whole-tree rules
    # (root-hoisted + fak/-nested) so the default cannot be dropped without a red test.
    gitignore = (Path(ROOT) / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "experiments/benchmark/runs/by-machine/" in gitignore, \
        "root-hoisted by-machine ignore rule missing from .gitignore"
    assert "fak/experiments/benchmark/runs/by-machine/" in gitignore, \
        "fak/-nested by-machine ignore rule missing from .gitignore"


# --- live regression guard: the real tree is clean --------------------------

def test_tracked_tree_has_no_private_only_path() -> None:
    """The whole tracked public tree must carry zero private-only paths — the
    invariant the gate enforces. This is the assertion that would have flagged the
    dgxbridge leak."""
    r = subprocess.run(["git", "-C", ROOT, "ls-files"], capture_output=True, text=True)
    assert r.returncode == 0, "git ls-files failed"
    hits = [p for p in r.stdout.split()
            if any(rx.search(p) for rx, _ in cc.PRIVATE_ONLY)]
    assert not hits, "private-only paths tracked in the public tree:\n" + "\n".join(hits)


def test_tracked_tree_has_no_secret_file() -> None:
    """The whole tracked public tree must carry zero secret/key files — the
    invariant the SECRET rule enforces (a leaked SA key is forever in history)."""
    r = subprocess.run(["git", "-C", ROOT, "ls-files"], capture_output=True, text=True)
    assert r.returncode == 0, "git ls-files failed"
    hits = [p for p in r.stdout.split()
            if any(rx.search(p) for rx, _ in cc.SECRET_FILES)]
    assert not hits, "secret/key files tracked in the public tree:\n" + "\n".join(hits)


# --- SCRIPT admission: random scripts refused, grandfathered allowed ---------

def test_random_powershell_script_refused() -> None:
    assert cc._classify("scripts/dogfood-opencode.ps1", ROOT, MAX) is not None
    assert cc._classify("tools/random.ps1", ROOT, MAX) is not None
    assert cc._classify("random.ps1", ROOT, MAX) is not None
    r = cc._classify("scripts/test.ps1", ROOT, MAX)
    assert r is not None and "prefer making Go" in r


def test_random_shell_script_refused() -> None:
    assert cc._classify("scripts/dogfood-opencode.sh", ROOT, MAX) is not None
    assert cc._classify("tools/random.sh", ROOT, MAX) is not None
    assert cc._classify("random.bat", ROOT, MAX) is not None
    assert cc._classify("random.cmd", ROOT, MAX) is not None
    r = cc._classify("foo/bar.sh", ROOT, MAX)
    assert r is not None and "prefer making Go" in r


def test_grandfathered_script_allowed() -> None:
    assert cc._classify("test.ps1", ROOT, MAX) is None
    assert cc._classify("test.sh", ROOT, MAX) is None
    assert cc._classify("scripts/build.sh", ROOT, MAX) is None
    assert cc._classify("tools/build_cuda_windows.ps1", ROOT, MAX) is None


def test_tracked_tree_has_no_ungrandfathered_script() -> None:
    """The tracked tree must carry zero scripts outside the grandfathered baseline."""
    r = subprocess.run(["git", "-C", ROOT, "ls-files"], capture_output=True, text=True)
    assert r.returncode == 0, "git ls-files failed"
    hits = [p for p in r.stdout.split()
            if cc._is_script_file(p) and p not in cc.GRANDFATHERED_SCRIPTS]
    assert not hits, "ungrandfathered scripts tracked in tree:\n" + "\n".join(hits)


def _run() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"ok   {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run())
