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


def test_lsp_scratch_dotfiles_refused_as_hard_junk() -> None:
    for p in (
        ".gopls",
        "sub/.gopls",
        ".clangd",
        ".clangd/index",
        ".ccls",
        ".ccls-cache",
        ".ccls-cache/foo",
        ".lsp",
        ".pylsp",
        ".rust-analyzer",
    ):
        r = cc._classify(p, ROOT, MAX)
        assert r is not None and "build artifact" in r.lower(), (p, r)


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


# --- per-commit shipped-bytes budget ----------------------------------------

BUDGET = cc.DEFAULT_COMMIT_SHIPPED_BYTES
KIB = 1024


def _fake_git(answers, calls=None):
    """Stand-in for cc._git that answers by the first git argument; a None answer is a failure."""
    def run(args, root):
        if calls is not None:
            calls.append(args)
        out = answers.get(args[0])
        return subprocess.CompletedProcess(args, 0 if out is not None else 128, stdout=out or "", stderr="")
    return run


def _with_git(fake, fn):
    real = cc._git
    cc._git = fake
    try:
        fn()
    finally:
        cc._git = real


def _with_env(name, value, fn):
    import os
    old = os.environ.get(name)
    os.environ[name] = value
    try:
        fn()
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


def _real_git(root, *args):
    return subprocess.run(["git", "-C", root, *args], capture_output=True, text=True, check=True)


def test_commit_budget_refuses_bulk_add_and_names_largest() -> None:
    sizes = {"assets/big.bin": 300 * KIB, "assets/small.bin": 1024}
    msg = cc._commit_budget_refusal(list(sizes), BUDGET, [], set(), sizes)
    assert msg is not None, "300 KiB over a 256 KiB budget must be refused"
    for want in ("assets/big.bin", "export-ignore", "sha256", "per commit", "FAK_COMMIT_SHIPPED_BYTES="):
        assert want in msg, (want, msg)


def test_commit_budget_admits_ordinary_commit() -> None:
    sizes = {"assets/a.bin": 100 * KIB, "assets/b.bin": 100 * KIB}
    assert cc._commit_budget_refusal(list(sizes), BUDGET, [], set(), sizes) is None


def test_commit_budget_boundary_is_inclusive() -> None:
    assert cc._commit_budget_refusal(["a.bin"], 8, [], set(), {"a.bin": 8}) is None
    assert cc._commit_budget_refusal(["b.bin"], 8, [], set(), {"b.bin": 9}) is not None


def test_commit_budget_skips_export_ignored_bytes() -> None:
    sizes = {"docs/receipts/big.bin": 300 * KIB}
    assert cc._commit_budget_refusal(list(sizes), BUDGET, [], {"docs/receipts/big.bin"}, sizes) is None


def test_commit_budget_skips_nested_module_bytes() -> None:
    sizes = {"docs/nightrun/data/big.bin": 300 * KIB}
    tracked = ["docs/nightrun/go.mod", "docs/nightrun/data/big.bin"]
    assert cc._commit_budget_refusal(list(sizes), BUDGET, tracked, set(), sizes) is None


def test_commit_budget_skips_text_go_sources() -> None:
    sizes = {"internal/tables/generated.go": 300 * KIB}
    assert cc._commit_budget_refusal(list(sizes), BUDGET, [], set(), sizes,
                                     text_go={"internal/tables/generated.go"}) is None


def test_commit_budget_bills_binary_go_blob() -> None:
    # A .go name only exempts a text blob; a binary one ships like any other file.
    sizes = {"internal/blob/data.go": 300 * KIB}
    assert cc._commit_budget_refusal(list(sizes), BUDGET, [], set(), sizes) is not None


def test_staged_blob_sizes_reads_index_blob_size_in_one_listing() -> None:
    calls = []
    listing = ("100644 blob aaaa 307200\tassets/big.bin\x00"
               "160000 commit bbbb -\tvendor/sub\x00")
    fake = _fake_git({"write-tree": "tree1\n", "ls-tree": listing}, calls)

    def check() -> None:
        assert cc._staged_blob_sizes("/unused") == {"assets/big.bin": 307200, "vendor/sub": 0}
        assert [c[0] for c in calls] == ["write-tree", "ls-tree"], calls
    _with_git(fake, check)


def test_worktree_divergence_is_judged_on_index_blob() -> None:
    # The worktree copy is one byte; the commit ships the 300 KiB index blob.
    with tempfile.TemporaryDirectory() as d:
        Path(d, "assets").mkdir()
        Path(d, "assets", "big.bin").write_bytes(b"a")

        def check() -> None:
            msg = cc._commit_budget_refusal(["assets/big.bin"], BUDGET, [], set(),
                                            cc._staged_blob_sizes(d))
            assert msg is not None, "a truncated worktree copy must not hide the indexed bytes"
        _with_git(_fake_git({"write-tree": "tree1\n", "ls-tree": "100644 blob aaaa 307200\tassets/big.bin\x00"}),
                  check)


def test_unlisted_staged_path_fails_closed() -> None:
    sizes = {"assets/a.bin": 1024}
    msg = cc._commit_budget_refusal(["assets/a.bin", "assets/unlisted.bin"], BUDGET, [], set(), sizes)
    assert msg is not None and "assets/unlisted.bin" in msg and "fail closed" in msg, msg


def test_export_ignore_is_read_from_the_index() -> None:
    calls = []

    def check() -> None:
        ignored = cc._export_ignored("/unused", ["docs/receipts/big.bin"])
        assert ignored == set(), ignored
        assert calls and "--cached" in calls[0], calls
    _with_git(_fake_git({"check-attr": "docs/receipts/big.bin\0export-ignore\0unspecified\0"}, calls), check)


def test_export_ignore_set_in_index_is_honored() -> None:
    def check() -> None:
        assert cc._export_ignored("/unused", ["docs/receipts/big.bin"]) == {"docs/receipts/big.bin"}
    _with_git(_fake_git({"check-attr": "docs/receipts/big.bin\0export-ignore\0set\0"}), check)


def test_commit_shipped_budget_env_falls_back_on_garbage() -> None:
    import os
    old = os.environ.get("FAK_COMMIT_SHIPPED_BYTES")
    try:
        for raw, want in [("", cc.DEFAULT_COMMIT_SHIPPED_BYTES), ("nope", cc.DEFAULT_COMMIT_SHIPPED_BYTES),
                          ("0", cc.DEFAULT_COMMIT_SHIPPED_BYTES), ("4096", 4096)]:
            os.environ["FAK_COMMIT_SHIPPED_BYTES"] = raw
            assert cc._commit_shipped_budget() == want, raw
    finally:
        if old is None:
            os.environ.pop("FAK_COMMIT_SHIPPED_BYTES", None)
        else:
            os.environ["FAK_COMMIT_SHIPPED_BYTES"] = old


# Each git read on the budget path must refuse the commit. The base answers describe a commit that
# is admitted, so each case differs from it by exactly one failed read.
_BUDGET_BASE_ANSWERS = {
    "diff": "A\0assets/a.bin\0A\0internal/small.go\0",
    "rev-parse": ".git/MERGE_HEAD\n",
    "--literal-pathspecs": "internal/small.go\0",
    "check-attr": "",
    "ls-files": "assets/a.bin\0internal/small.go\0",
    "write-tree": "tree1\n",
    "ls-tree": "100644 blob aaaa 1024\tassets/a.bin\x00100644 blob bbbb 1024\tinternal/small.go\x00",
}


def test_budget_base_commit_is_admitted() -> None:
    _with_git(_fake_git(_BUDGET_BASE_ANSWERS), lambda: _assert_admitted())


def _assert_admitted() -> None:
    assert cc._staged_budget_refusal("/unused") is None


def test_failed_reads_fail_closed() -> None:
    cases = [
        ("diff", None, "list the staged names"),
        ("rev-parse", None, "read whether a merge is in progress"),
        ("rev-parse", "\n", "read whether a merge is in progress"),
        ("--literal-pathspecs", None, "check which staged .go files are text"),
        ("check-attr", None, "read export-ignore"),
        ("ls-files", None, "list the tracked paths"),
        ("write-tree", None, "size the staged blobs"),
        ("ls-tree", None, "size the staged blobs"),
    ]
    for key, value, phrase in cases:
        answers = dict(_BUDGET_BASE_ANSWERS, **{key: value})
        result = []
        _with_git(_fake_git(answers), lambda: result.append(cc._staged_budget_refusal("/unused")))
        msg = result[0]
        assert msg is not None and phrase in msg and "fail closed" in msg, (key, msg)


def test_git_that_cannot_run_fails_closed() -> None:
    def boom(args, root):
        raise FileNotFoundError("git")
    result = []
    _with_git(boom, lambda: result.append(cc._staged_budget_refusal("/unused")))
    assert result[0] is not None and "fail closed" in result[0], result


def test_export_query_failure_refuses_even_a_small_commit() -> None:
    # A failed attribute query must refuse, not bill the file as shipped and hope it is small.
    answers = dict(_BUDGET_BASE_ANSWERS, **{"check-attr": None})
    result = []
    _with_git(_fake_git(answers), lambda: result.append(cc._staged_budget_refusal("/unused")))
    assert result[0] is not None and "read export-ignore" in result[0], result


def _budget_refusal_in_real_repo(name, size, export_ignore=False):
    """Stage one file in a real repo with core.quotepath=true and return the budget refusal."""
    with tempfile.TemporaryDirectory() as d:
        _real_git(d, "init", "-q")
        _real_git(d, "config", "core.quotepath", "true")
        Path(d, name).write_bytes(b"a" * size)
        staged = [name]
        if export_ignore:
            Path(d, ".gitattributes").write_text(f"{name} export-ignore\n", encoding="utf-8")
            staged.append(".gitattributes")
        _real_git(d, "add", "--", *staged)
        return cc._staged_budget_refusal(d)


def test_non_ascii_bulk_name_refused_under_quotepath() -> None:
    msg = _budget_refusal_in_real_repo("naïve big.bin", 300 * KIB)
    assert msg is not None and "commit adds" in msg and "naïve big.bin" in msg, msg


def test_non_ascii_ordinary_name_admitted_under_quotepath() -> None:
    assert _budget_refusal_in_real_repo("naïve small.bin", 100 * KIB) is None


def test_non_ascii_export_ignored_bulk_admitted_under_quotepath() -> None:
    assert _budget_refusal_in_real_repo("bulké.bin", 300 * KIB, export_ignore=True) is None


def _commit_repo(d, files):
    """Init a repo in `d` with core.quotepath on and commit `files` (name -> bytes) as the parent."""
    _real_git(d, "init", "-q")
    _real_git(d, "config", "core.quotepath", "true")
    for name, data in files.items():
        Path(d, name).parent.mkdir(parents=True, exist_ok=True)
        Path(d, name).write_bytes(data)
    _real_git(d, "add", "--", *files)
    _real_git(d, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "base")


def test_binary_go_blob_is_billed_in_real_repo() -> None:
    with tempfile.TemporaryDirectory() as td:
        _commit_repo(td, {"README.md": b"x\n"})
        Path(td, "internal", "blob").mkdir(parents=True)
        Path(td, "internal", "blob", "data.go").write_bytes(b"\0" * (300 * KIB))
        _real_git(td, "add", "--", "internal/blob/data.go")
        msg = cc._staged_budget_refusal(td)
        assert msg is not None and "internal/blob/data.go" in msg, msg


def test_text_go_blob_is_admitted_in_real_repo() -> None:
    with tempfile.TemporaryDirectory() as td:
        _commit_repo(td, {"README.md": b"x\n"})
        Path(td, "gen.go").write_bytes(b"// generated\n" * 25000)  # 325 KiB of text
        _real_git(td, "add", "--", "gen.go")
        assert cc._staged_budget_refusal(td) is None


def test_pure_move_of_shipped_asset_is_refused_in_real_repo() -> None:
    # A rename bills its full new blob, so a pure move of 300 KiB is refused.
    with tempfile.TemporaryDirectory() as td:
        _commit_repo(td, {"assets/old.bin": b"a" * (300 * KIB)})
        _real_git(td, "mv", "assets/old.bin", "assets/new.bin")
        msg = cc._staged_budget_refusal(td)
        assert msg is not None and "commit adds 300 KiB" in msg and "pure move" in msg, msg
        assert f"FAK_COMMIT_SHIPPED_BYTES={300 * KIB}" in msg, msg


def test_pure_move_admitted_when_budget_raised_in_real_repo() -> None:
    with tempfile.TemporaryDirectory() as td:
        _commit_repo(td, {"assets/old.bin": b"a" * (300 * KIB)})
        _real_git(td, "mv", "assets/old.bin", "assets/new.bin")
        _with_env("FAK_COMMIT_SHIPPED_BYTES", str(300 * KIB), lambda: _assert_move_admitted(td))


def _assert_move_admitted(td) -> None:
    assert cc._staged_budget_refusal(td) is None


def test_small_pure_move_is_admitted_in_real_repo() -> None:
    with tempfile.TemporaryDirectory() as td:
        _commit_repo(td, {"assets/old.bin": b"a" * (100 * KIB)})
        _real_git(td, "mv", "assets/old.bin", "assets/new.bin")
        assert cc._staged_budget_refusal(td) is None


def _move_into_shipped_dir(td, old, new):
    Path(td, new).parent.mkdir(parents=True, exist_ok=True)
    _real_git(td, "mv", old, new)


def test_rename_from_export_ignored_source_is_billed_in_full() -> None:
    with tempfile.TemporaryDirectory() as td:
        _commit_repo(td, {"docs/receipts/old.bin": b"a" * (300 * KIB),
                          ".gitattributes": b"docs/receipts/old.bin export-ignore\n"})
        _move_into_shipped_dir(td, "docs/receipts/old.bin", "assets/new.bin")
        msg = cc._staged_budget_refusal(td)
        assert msg is not None and "commit adds 300 KiB" in msg, msg


def test_rename_from_nested_module_source_is_billed_in_full() -> None:
    with tempfile.TemporaryDirectory() as td:
        _commit_repo(td, {"docs/nightrun/go.mod": b"module x\n",
                          "docs/nightrun/data/old.bin": b"a" * (300 * KIB)})
        _move_into_shipped_dir(td, "docs/nightrun/data/old.bin", "assets/new.bin")
        msg = cc._staged_budget_refusal(td)
        assert msg is not None and "commit adds 300 KiB" in msg, msg


def test_rename_from_text_go_source_is_billed_in_full() -> None:
    with tempfile.TemporaryDirectory() as td:
        _commit_repo(td, {"gen.go": b"// generated\n" * 25000})  # 317 KiB of text
        _move_into_shipped_dir(td, "gen.go", "assets/gen.bin")
        msg = cc._staged_budget_refusal(td)
        assert msg is not None and "commit adds 317 KiB" in msg, msg


def test_rename_from_binary_go_is_billed_in_full() -> None:
    with tempfile.TemporaryDirectory() as td:
        _commit_repo(td, {"internal/blob/old.go": b"\0" * (300 * KIB)})
        _move_into_shipped_dir(td, "internal/blob/old.go", "assets/new.bin")
        msg = cc._staged_budget_refusal(td)
        assert msg is not None and "commit adds 300 KiB" in msg, msg


def test_rename_plus_growth_bills_full_new_blob_in_real_repo() -> None:
    # 400 KiB base, then 300 KiB appended: git still detects the rename (>=50% similar), and the
    # commit bills the whole 700 KiB new blob, not just the growth.
    with tempfile.TemporaryDirectory() as td:
        base = b"asset row\n" * 40960  # 400 KiB
        _commit_repo(td, {"assets/old.bin": base})
        _real_git(td, "mv", "assets/old.bin", "assets/new.bin")
        Path(td, "assets", "new.bin").write_bytes(base + b"extra row\n" * 30720)  # +300 KiB
        _real_git(td, "add", "--", "assets/new.bin")
        msg = cc._staged_budget_refusal(td)
        assert msg is not None and "commit adds 700 KiB" in msg, msg


def test_merge_in_progress_skips_budget_in_real_repo() -> None:
    with tempfile.TemporaryDirectory() as td:
        _commit_repo(td, {"README.md": b"x\n"})
        Path(td, "assets").mkdir()
        Path(td, "assets", "big.bin").write_bytes(b"a" * (300 * KIB))
        _real_git(td, "add", "--", "assets/big.bin")
        assert cc._staged_budget_refusal(td) is not None, "control: refused outside a merge"
        Path(td, ".git", "MERGE_HEAD").write_text("0" * 40 + "\n", encoding="utf-8")
        assert cc._staged_budget_refusal(td) is None


def test_tracked_listing_keeps_names_with_spaces_whole() -> None:
    # Whitespace-split listing would make "my dir/go.mod" tracked as "dir/go.mod" and exempt the
    # staged dir/big.bin under it. The NUL-delimited listing keeps the name whole.
    with tempfile.TemporaryDirectory() as td:
        _commit_repo(td, {"my dir/go.mod": b"module x\n", "README.md": b"x\n"})
        Path(td, "dir").mkdir()
        Path(td, "dir", "big.bin").write_bytes(b"a" * (300 * KIB))
        _real_git(td, "add", "--", "dir/big.bin")
        assert "my dir/go.mod" in cc._budget_tracked(td)
        assert cc._staged_budget_refusal(td) is not None


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
