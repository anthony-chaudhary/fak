#!/usr/bin/env python3
"""Regression witness: the DOS verify-referee actually binds fleet's `(fak <leaf>)`
trailer ship-stamp (issue #387, RSI loop item #2).

The fleet's KEEP decision (the self-improve loop, #1) rests on `dos verify fak <leaf>`
returning SHIPPED from git evidence. But fleet writes Conventional-Commits subjects with
a `(fak <leaf>)` TRAILER (`fix(gateway): … (fak gateway)`), not a `fak/<leaf>:` start
prefix. #387 witnessed the danger: a kernel that binds only by subject-prefix grep reads a
correctly-stamped ship as NOT_SHIPPED, so the loop would REVERT a genuinely-good change —
a false-negative in the exact gate the loop trusts.

`dos.toml` already carries `[stamp] trailer_stamp = true` (docs/289), and the installed
kernel honors it. The sibling `commit_stamp_doctor_test.py` proves the LOCAL heuristic
recognizes the trailer shapes; this is its missing twin — it proves the REAL oracle
(`dos verify`, the source of truth) binds them, against the kernel actually installed on
this node. Without this pin, a kernel upgrade or a `dos.toml` edit that drops the flag
would silently reintroduce #387's false-negative and nothing would fail.

It is hermetic: it builds a throwaway git repo, stamps one commit per trailer form with a
leaf token that appears ONLY in the trailer (so a subject-prefix grep cannot bind it), and
asserts the oracle's verdict. If `dos` is not on PATH (a node without the kernel), the
tests SKIP rather than fail — the witness only fires where the referee actually runs.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

DOS = shutil.which("dos")

# The three trailer forms commit_stamp_doctor enumerates and dos.toml's [stamp] grammar
# is meant to bind. Each leaf token is unique and appears ONLY inside the trailer, so a
# legacy subject-prefix grep (the #387 failure mode) physically cannot match it — a bind
# proves the trailer rung, not an incidental subject word.
TRAILER_FORMS = [
    ("zztrailerparen", "chore(misc): unrelated subject text (fak zztrailerparen)"),
    ("zztrailercolon", "feat(misc): unrelated subject text (fak: zztrailercolon)"),
    ("zztrailerrefs", "fix(misc): unrelated subject text (refs fak zztrailerrefs)"),
]

_DOS_TOML = """\
workspace = "."
[lanes]
concurrent = ["zztrailerparen", "zztrailercolon", "zztrailerrefs"]
[lanes.trees]
zztrailerparen = ["zztrailerparen/**"]
zztrailercolon = ["zztrailercolon/**"]
zztrailerrefs = ["zztrailerrefs/**"]
[paths]
plans_glob = "docs/**/*-plan.md"
[stamp]
style = "grep"
subject_dirs = []
trailer_stamp = {trailer_stamp}
"""


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True,
                   capture_output=True, text=True)


def _commit(repo: Path, fname: str, subject: str) -> None:
    """Stage one new file and commit it by explicit path with the given subject."""
    (repo / fname).write_text("x\n", encoding="utf-8")
    _git(repo, "add", fname)
    _git(repo, "commit", "-q", "-s", "-m", subject, "--", fname)


def _make_repo(tmp: Path, *, trailer_stamp: bool) -> Path:
    """A throwaway git repo with a dos.toml and one stamped commit per trailer form.

    The leaf token in each subject lives ONLY in the trailer; nothing else in the
    history names it, so the only way `dos verify` can bind it is the trailer rung.
    """
    repo = tmp
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "witness@fak.test")
    _git(repo, "config", "user.name", "fak witness")
    _git(repo, "config", "commit.gpgsign", "false")
    (repo / "dos.toml").write_text(
        _DOS_TOML.format(trailer_stamp="true" if trailer_stamp else "false"),
        encoding="utf-8")
    _git(repo, "add", "dos.toml")
    _git(repo, "commit", "-q", "-s", "-m", "chore: seed dos.toml", "--", "dos.toml")
    for i, (_leaf, subject) in enumerate(TRAILER_FORMS):
        _commit(repo, f"f{i}.txt", subject)
    return repo


def _verify(repo: Path, leaf: str) -> dict:
    """Run the real `dos verify --json fak <leaf>` oracle and parse its verdict."""
    out = subprocess.run(
        [DOS, "verify", "--workspace", str(repo), "--json", "fak", leaf],
        capture_output=True, text=True)
    # --json emits a single JSON object; take the last line that parses, so any
    # incidental banner on stdout does not break the witness.
    for line in reversed(out.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise AssertionError(f"dos verify produced no JSON: {out.stdout!r} {out.stderr!r}")


@unittest.skipIf(DOS is None, "dos kernel not on PATH — referee witness skipped")
class TrailerStampBindsToShipped(unittest.TestCase):
    """Acceptance #1: a HEAD commit stamped ONLY with a `(fak <leaf>)` trailer must
    read as SHIPPED — for every trailer form, against the installed kernel."""

    def test_each_trailer_form_binds_shipped(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            repo = _make_repo(Path(d), trailer_stamp=True)
            for leaf, _subject in TRAILER_FORMS:
                verdict = _verify(repo, leaf)
                self.assertTrue(
                    verdict.get("shipped"),
                    f"trailer-only leaf {leaf!r} read NOT_SHIPPED — the #387 "
                    f"false-negative: {verdict}")
                # source must be a positive git witness, never the empty 'none'.
                self.assertNotEqual(verdict.get("source"), "none", verdict)
                # The bind must be the TRAILER rung specifically: the token appears
                # nowhere but the trailer, so a 'direct' subject-prefix rung here would
                # mean the kernel matched something else and the trailer is still blind.
                self.assertEqual(verdict.get("rung"), "trailer", verdict)

    def test_unstamped_leaf_stays_not_shipped(self) -> None:
        # Anti-vacuous floor: a leaf no commit names must NOT read as shipped, or a
        # kernel that blindly returns SHIPPED would pass the positive test for free.
        with tempfile.TemporaryDirectory() as d:
            repo = _make_repo(Path(d), trailer_stamp=True)
            verdict = _verify(repo, "neverappearsanywhere")
            self.assertFalse(verdict.get("shipped"), verdict)
            self.assertEqual(verdict.get("source"), "none", verdict)


@unittest.skipIf(DOS is None, "dos kernel not on PATH — referee witness skipped")
class TrailerBindingIsConfigDriven(unittest.TestCase):
    """The bind is owned by `[stamp] trailer_stamp = true`, not an incidental grep.
    This is the regression #387 guards against: drop the flag and the same trailer-only
    history goes dark. If this test ever starts FAILING (the leaves stay shipped with
    the flag off), the kernel changed its trailer semantics — investigate before trusting
    the loop's KEEP decisions again."""

    def test_dropping_trailer_stamp_unbinds_the_trailers(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            repo = _make_repo(Path(d), trailer_stamp=False)
            for leaf, _subject in TRAILER_FORMS:
                verdict = _verify(repo, leaf)
                self.assertFalse(
                    verdict.get("shipped"),
                    f"leaf {leaf!r} bound with trailer_stamp=false — the trailer "
                    f"rung is not actually gated by the config: {verdict}")


@unittest.skipIf(DOS is None, "dos kernel not on PATH — referee witness skipped")
class DoctorReportsTrailerSupport(unittest.TestCase):
    """Acceptance #2: `dos doctor` reports trailer-stamp support is live (the stamp
    convention names the trailer), not silently degraded to subject-grep."""

    def test_doctor_names_the_trailer_grammar(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            repo = _make_repo(Path(d), trailer_stamp=True)
            out = subprocess.run(
                [DOS, "doctor", "--workspace", str(repo)],
                capture_output=True, text=True)
            self.assertIn(
                "trailer", out.stdout.lower(),
                f"dos doctor does not report trailer-stamp support — it may have "
                f"silently degraded to subject-grep:\n{out.stdout}\n{out.stderr}")


if __name__ == "__main__":
    unittest.main()
