#!/usr/bin/env python3
"""Tests for the file-admission gate (`check_committed_files.py`).

Focuses on the PRIVATE_ONLY guard — the public-tree enforcement that keeps the
operator's private lab GPU-server *connection* subsystem (the Slack control-bridge
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
    assert not any(rx.search("experiments/qwen36/dgx-r4-20260622/compare.json")
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
