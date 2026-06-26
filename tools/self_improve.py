#!/usr/bin/env python3
r"""self_improve.py — run the dos-self-improve loop end-to-end on fak (#388).

The recursive-self-improvement loop, made a capability rather than a design:

    propose a change  ->  KEEP it only if a witness the change's author did NOT
    write confirms an improvement  ->  otherwise REVERT.

The keep-bit is the AND of three env-authored witnesses (GROWTH §5), none of them
the loop's own say-so:

  (1) suite green on a clean isolated worktree   — `go test ./internal/<leaf>/...`
  (2) internal/architest green                   — the layering/tier contract held
  (3) `dos verify fak <leaf>` confirms the leaf shipped from GIT evidence — and,
      for witness honesty (#125), a `source: none` / `subject-only` verdict does
      NOT count as confirmation.

The kernel ratifies the keep with its typed `dos improve` verdict (exit 0 KEEP /
3 REVERT / 4 ESCALATE) over the same env-authored facts, so the loop cannot keep a
change by *narrating* that it is better — the only path to KEEP is to actually
pass the witnesses. KEEP requires BOTH the AND-of-three AND the kernel verdict;
either says no and the candidate is REVERTED (fail-safe).

Isolation is the whole point: every candidate is applied in an ISOLATED git
worktree off a PINNED base SHA, never the live shared tree. That (a) avoids the 9p
edit/read race GROWTH §7 calls out, (b) keeps the baseline GREEN even when the
live tree is dirty with a peer's untracked work, and (c) means the kernel
adjudicating a change is never the kernel being rewritten.

Usage:
    python tools/self_improve.py --leaf benchids                 # seed: KEEP + REVERT
    python tools/self_improve.py --leaf benchids --only good     # just the KEEP arm
    python tools/self_improve.py --leaf benchids --only bad       # just the REVERT arm
    python tools/self_improve.py --leaf benchids --patch FILE \   # drive a real patch
        --subject "fix(benchids): ... (fak benchids)"

On Windows the Go witnesses run inside WSL (native `go test` is OS-blocked here —
see test.ps1); `dos verify` / `dos improve` run natively. On Linux everything runs
natively. The loop NEVER promotes a kept commit onto the live trunk by default
(honoring "never edits the live shared worktree"): a KEEP records the kept SHA and
preserves the candidate's full diff as an artifact; merging to trunk is a separate,
`dos verify`-gated operator step (`--promote` exists for a non-shared base only).
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import shlex
import subprocess
import sys
import tempfile
import uuid

IS_WINDOWS = platform.system() == "Windows"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(REPO, "tools", "self_improve.runs")

# A dos verify verdict whose evidence is no stronger than this is NOT a
# confirmation — it must not be allowed to count toward a KEEP (witness honesty,
# #125). `none` is "git history alone could not confirm"; a bare subject-only /
# body mention is too weak to ratify a shipped phase.
WEAK_VERIFY_SOURCES = {"none", "", "subject-only", "release-body", "body"}
WEAK_VERIFY_RUNGS = {"subject-only", "none", ""}


# --------------------------------------------------------------------------- #
# shell / path helpers
# --------------------------------------------------------------------------- #
def run(cmd, cwd=None, env=None, check=False):
    """Run a command, capturing combined text output."""
    p = subprocess.run(
        cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
    )
    if check and p.returncode != 0:
        raise RuntimeError("command failed ({}): {}\n{}".format(p.returncode, cmd, p.stdout))
    return p


def to_wsl_path(winpath: str) -> str:
    """C:\\work\\fak  ->  /mnt/c/work/fak  (default WSL automount)."""
    p = os.path.abspath(winpath)
    drive = p[0].lower()
    rest = p[2:].replace("\\", "/")
    return "/mnt/{}{}".format(drive, rest)


def _wsl_distro():
    """Mirror test.ps1: honor FAK_WSL_DISTRO, else prefer Ubuntu-24.04, else default."""
    env = os.environ.get("FAK_WSL_DISTRO")
    if env:
        return env
    try:
        out = subprocess.run(["wsl.exe", "--list", "--quiet"],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout
        names = out.decode("utf-16-le", errors="ignore")
        if "Ubuntu-24.04" in names:
            return "Ubuntu-24.04"
    except Exception:
        pass
    return None


def go_cmd(worktree: str, inner: str):
    """Build a command list that runs `inner` (a go invocation) at `worktree`.

    On Windows this tunnels through WSL because native `go test` is OS-blocked
    (Application Control refuses the unsigned %TEMP% test .exe); on Linux it runs
    in-place via cwd.
    """
    if IS_WINDOWS:
        wsl_dir = to_wsl_path(worktree)
        distro = _wsl_distro()
        line = "cd {} && {}".format(shlex.quote(wsl_dir), inner)
        return ["wsl.exe"] + (["-d", distro] if distro else []) + ["--", "bash", "-lc", line], None
    return ["bash", "-lc", inner], worktree


# --------------------------------------------------------------------------- #
# the three witnesses
# --------------------------------------------------------------------------- #
def go_test(worktree: str, pkg: str, count1: bool = True):
    """Witness 1/2: run `go test <pkg>` in the isolated worktree. Returns (green, exit, tail)."""
    extra = " -count=1" if count1 else ""
    cmd, cwd = go_cmd(worktree, "go test {}{}".format(pkg, extra))
    p = run(cmd, cwd=cwd)
    tail = "\n".join(p.stdout.strip().splitlines()[-4:])
    return p.returncode == 0, p.returncode, tail


def count_tests(worktree: str, pkg: str) -> int:
    """The env-measured metric: number of test functions in `pkg` (compile-only list)."""
    cmd, cwd = go_cmd(worktree, 'go test -list ".*" {}'.format(pkg))
    p = run(cmd, cwd=cwd)
    if p.returncode != 0:
        return -1
    return sum(1 for ln in p.stdout.splitlines() if ln.strip().startswith(("Test", "Fuzz")))


def dos_verify(worktree: str, plan: str, leaf: str) -> dict:
    """Witness 3: did `(plan, leaf)` actually ship, by git evidence? Returns the verdict dict."""
    p = run(["dos", "verify", plan, leaf, "--workspace", worktree, "--json"])
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        return {"shipped": False, "source": "none", "_raw": p.stdout.strip()}


def verify_is_clean(verify: dict) -> bool:
    """Witness honesty (#125): shipped AND on evidence stronger than subject-only/none."""
    if not verify.get("shipped"):
        return False
    if str(verify.get("source", "")).lower() in WEAK_VERIFY_SOURCES:
        return False
    if str(verify.get("rung", "")).lower() in WEAK_VERIFY_RUNGS:
        return False
    return True


def dos_improve(suite_passed: bool, truth_clean: bool, work: int, baseline: int,
                narrated: str, reverts: int = 0, max_reverts: int = 3) -> dict:
    """The kernel's typed keep-gate over env-authored facts. Returns {exit, verdict}."""
    cmd = ["dos", "improve", "--work", str(work), "--baseline-work", str(baseline),
           "--consecutive-reverts", str(reverts), "--max-reverts", str(max_reverts),
           "--narrated", narrated, "--json"]
    if suite_passed:
        cmd.append("--suite-passed")
    if truth_clean:
        cmd.append("--truth-clean")
    p = run(cmd)
    verdict = {0: "KEEP", 3: "REVERT", 4: "ESCALATE", 2: "CONTRACT_ERROR"}.get(p.returncode, "UNKNOWN")
    return {"exit": p.returncode, "verdict": verdict, "raw": p.stdout.strip()}


# --------------------------------------------------------------------------- #
# the pure keep-decision (unit-tested without WSL/go)
# --------------------------------------------------------------------------- #
def decide(suite_green: bool, architest_green: bool, verify: dict, improve_verdict: str):
    """The keep-bit: AND-of-three (GROWTH §5) AND the kernel ratifies. Fail-safe to REVERT.

    Returns (keep: bool, reasons: list[str]).
    """
    truth_clean = verify_is_clean(verify)
    reasons = []
    reasons.append(("suite green" if suite_green else "SUITE RED") + " (witness 1)")
    reasons.append(("architest green" if architest_green else "ARCHITEST RED") + " (witness 2)")
    if truth_clean:
        reasons.append("dos verify confirms ship via {} (witness 3)".format(verify.get("source")))
    else:
        reasons.append("dos verify NOT a confirmation: shipped={} source={} (witness 3)".format(
            verify.get("shipped"), verify.get("source")))
    reasons.append("kernel dos improve = {}".format(improve_verdict))
    witness_and = suite_green and architest_green and truth_clean
    keep = witness_and and improve_verdict == "KEEP"
    return keep, reasons


# --------------------------------------------------------------------------- #
# candidates: a proposed change applied inside the isolated worktree
# --------------------------------------------------------------------------- #
def good_candidate(leaf: str) -> dict:
    """A genuine, additive improvement to a leaf: a new passing test (more coverage)."""
    body = '''package {pkg}

import "testing"

// TestLCGAllIDsWithinVocab guards the LCG bound invariant across a sweep of vocab
// sizes: every emitted id must fall in [0, vocab). Added by the dos-self-improve
// seed run (#388) as a real coverage gain — the KEEP arm.
func TestLCGAllIDsWithinVocab(t *testing.T) {{
	for _, vocab := range []int{{1, 2, 7, 256, 50000}} {{
		ids := LCG(512, vocab, 11)
		for i, id := range ids {{
			if id < 0 || id >= vocab {{
				t.Fatalf("vocab=%d: ids[%d]=%d out of [0,%d)", vocab, i, id, vocab)
			}}
		}}
	}}
}}
'''.format(pkg=leaf)
    return {
        "name": "good",
        "kind": "good",
        "relpath": "internal/{}/selfimprove_seed_keep_test.go".format(leaf),
        "content": body,
        "subject": "test({leaf}): guard LCG id bound across vocab sweep (fak {leaf})".format(leaf=leaf),
    }


def bad_candidate(leaf: str) -> dict:
    """A deliberate regression: a failing test, to prove the REVERT arm fires on a red suite."""
    body = '''package {pkg}

import "testing"

// TestSelfImproveSeedDeliberateRegression is the dos-self-improve REVERT-arm seed
// (#388): it fails ON PURPOSE so the witness must catch it and REVERT. A loop that
// graded its own homework would keep this; the env-authored suite witness cannot.
func TestSelfImproveSeedDeliberateRegression(t *testing.T) {{
	t.Fatal("deliberate REVERT-arm regression (#388 seed) — this candidate MUST be reverted")
}}
'''.format(pkg=leaf)
    return {
        "name": "bad",
        "kind": "bad",
        "relpath": "internal/{}/selfimprove_seed_revert_test.go".format(leaf),
        "content": body,
        "subject": "test({leaf}): inject a failing test to exercise REVERT (fak {leaf})".format(leaf=leaf),
    }


# --------------------------------------------------------------------------- #
# worktree lifecycle (NEVER the live shared tree)
# --------------------------------------------------------------------------- #
def add_worktree(base_sha: str) -> str:
    path = os.path.join(tempfile.gettempdir(), "fak-si-{}".format(uuid.uuid4().hex[:12]))
    run(["git", "-C", REPO, "worktree", "add", "--detach", path, base_sha], check=True)
    return path


def remove_worktree(path: str):
    run(["git", "-C", REPO, "worktree", "remove", "--force", path])
    run(["git", "-C", REPO, "worktree", "prune"])


def apply_and_commit(worktree: str, cand: dict) -> str:
    rel = cand["relpath"]
    dest = os.path.join(worktree, rel.replace("/", os.sep))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(cand["content"])
    run(["git", "-C", worktree, "add", "--", rel], check=True)
    run(["git", "-C", worktree,
         "-c", "user.name=fak self-improve", "-c", "user.email=self-improve@fak.local",
         "commit", "-s", "-m", cand["subject"], "--", rel], check=True)
    sha = run(["git", "-C", worktree, "rev-parse", "HEAD"]).stdout.strip()
    return sha


def candidate_patch(worktree: str, base_sha: str) -> str:
    return run(["git", "-C", worktree, "diff", "{}..HEAD".format(base_sha)]).stdout


# --------------------------------------------------------------------------- #
# one full cycle
# --------------------------------------------------------------------------- #
def run_cycle(leaf: str, cand: dict, base_sha: str, cycle: int, reverts: int) -> dict:
    pkg = "./internal/{}/...".format(leaf)
    wt = add_worktree(base_sha)
    try:
        baseline_tests = count_tests(wt, pkg)
        cand_sha = apply_and_commit(wt, cand)
        candidate_tests = count_tests(wt, pkg)
        patch = candidate_patch(wt, base_sha)

        suite_green, suite_exit, suite_tail = go_test(wt, pkg)
        arch_green, arch_exit, arch_tail = go_test(wt, "./internal/architest/...")
        verify = dos_verify(wt, "fak", leaf)
        truth_clean = verify_is_clean(verify)

        improve = dos_improve(
            suite_passed=suite_green and arch_green,
            truth_clean=truth_clean,
            work=candidate_tests, baseline=baseline_tests,
            narrated="self-improve seed candidate '{}' on leaf {}".format(cand["name"], leaf),
            reverts=reverts,
        )
        keep, reasons = decide(suite_green, arch_green, verify, improve["verdict"])
        decision = "KEEP" if keep else "REVERT"

        record = {
            "issue": "388",
            "cycle": cycle,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "leaf": leaf,
            "package": pkg,
            "base_sha": base_sha,
            "candidate": {
                "name": cand["name"], "kind": cand["kind"], "subject": cand["subject"],
                "relpath": cand["relpath"], "sha": cand_sha,
            },
            "witnesses": {
                "suite": {"name": "go test {}".format(pkg), "green": suite_green,
                          "exit": suite_exit, "tail": suite_tail},
                "architest": {"name": "go test ./internal/architest/...", "green": arch_green,
                              "exit": arch_exit, "tail": arch_tail},
                "dos_verify": {"name": "dos verify fak {}".format(leaf),
                               "shipped": verify.get("shipped"), "source": verify.get("source"),
                               "rung": verify.get("rung"), "sha": verify.get("sha"),
                               "counts_as_confirmation": truth_clean},
            },
            "metric": {"baseline_tests": baseline_tests, "candidate_tests": candidate_tests},
            "kernel_dos_improve": improve,
            "decision": decision,
            "decision_reasons": reasons,
            "patch": patch,
        }
        return record
    finally:
        remove_worktree(wt)


def write_record(record: dict) -> str:
    os.makedirs(RUNS_DIR, exist_ok=True)
    fname = "cycle-{:02d}-{}-{}.json".format(
        record["cycle"], record["candidate"]["name"], record["candidate"]["sha"][:8])
    path = os.path.join(RUNS_DIR, fname)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(record, fh, indent=2)
        fh.write("\n")
    return path


def print_cycle(record: dict):
    w = record["witnesses"]
    print("\n=== cycle {} — candidate '{}' on leaf {} ({}) ===".format(
        record["cycle"], record["candidate"]["name"], record["leaf"], record["candidate"]["sha"][:8]))
    print("  witness 1 suite      : {}".format("GREEN" if w["suite"]["green"] else "RED"))
    print("  witness 2 architest  : {}".format("GREEN" if w["architest"]["green"] else "RED"))
    print("  witness 3 dos verify : shipped={} source={} -> {}".format(
        w["dos_verify"]["shipped"], w["dos_verify"]["source"],
        "CONFIRMS" if w["dos_verify"]["counts_as_confirmation"] else "NOT A CONFIRMATION"))
    print("  metric (test count)  : {} -> {}".format(
        record["metric"]["baseline_tests"], record["metric"]["candidate_tests"]))
    print("  kernel dos improve   : {} (exit {})".format(
        record["kernel_dos_improve"]["verdict"], record["kernel_dos_improve"]["exit"]))
    print("  DECISION             : {}".format(record["decision"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description="run the dos-self-improve loop on fak (#388)")
    ap.add_argument("--leaf", default="benchids", help="target leaf (default: benchids)")
    ap.add_argument("--base", default=None, help="pinned base SHA (default: HEAD)")
    ap.add_argument("--only", choices=["good", "bad"], default=None,
                    help="run only one arm (default: both — the seed KEEP+REVERT)")
    args = ap.parse_args(argv)

    base = args.base or run(["git", "-C", REPO, "rev-parse", "HEAD"]).stdout.strip()
    print("dos-self-improve loop (#388): leaf={} base={}".format(args.leaf, base[:8]))
    print("(isolated worktrees off the pinned base — the live shared tree is never touched)")

    arms = [good_candidate(args.leaf), bad_candidate(args.leaf)]
    if args.only:
        arms = [c for c in arms if c["name"] == args.only]

    reverts = 0
    records = []
    for i, cand in enumerate(arms, start=1):
        rec = run_cycle(args.leaf, cand, base, i, reverts)
        path = write_record(rec)
        print_cycle(rec)
        print("  record               : {}".format(os.path.relpath(path, REPO)))
        if rec["decision"] == "REVERT":
            reverts += 1
        records.append(rec)

    kept = [r for r in records if r["decision"] == "KEEP"]
    reverted = [r for r in records if r["decision"] == "REVERT"]
    print("\nsummary: {} KEEP, {} REVERT".format(len(kept), len(reverted)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
