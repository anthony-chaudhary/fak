#!/usr/bin/env python3
"""Hermetic tests for tools/release_tag.py."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "release_tag.py"


def load():
    spec = importlib.util.spec_from_file_location("release_tag", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def git(root: Path, *args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)
    return proc.stdout.strip()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


MODULE_BUDGET_REL = "internal/architest/module_zip_budget_test.go"
MODULE_BUDGET_GO = """package architest

const (
	moduleZipProxyCapBytes  = 500 << 20
	moduleZipBudgetBytes    = {budget}
	moduleZipNonGoBudget    = 190 << 20
	moduleZipNonGoFileLimit = 5 << 20
)
"""


class ReleaseTagTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="release_tag_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.old_env = {k: os.environ.get(k) for k in (
            "GIT_CONFIG_GLOBAL", "GIT_CONFIG_SYSTEM", "GIT_CONFIG_NOSYSTEM",
            "GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL",
        )}
        os.environ.update({
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@example.com",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@example.com",
        })
        self.addCleanup(self._restore_env)

    def _restore_env(self) -> None:
        for key, value in self.old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _repo(self, version: str = "0.1.0") -> tuple[Path, str]:
        root = self.tmp / "repo"
        root.mkdir()
        git(root, "init", "-b", "master")
        write(root / "VERSION", version + "\n")
        write(root / "docs" / "releases" / f"v{version}.md", f"# v{version}\n")
        write(root / "go.mod", "module example.com/root\n\ngo 1.22\n")
        write(root / MODULE_BUDGET_REL, MODULE_BUDGET_GO.format(budget="375 << 20"))
        git(root, "add", "VERSION", f"docs/releases/v{version}.md", "go.mod", MODULE_BUDGET_REL)
        git(root, "commit", "-m", f"v{version}: seed release")
        sha = git(root, "rev-parse", "HEAD")
        return root, sha

    def _commit(self, root: Path, files: dict[str, object]) -> str:
        for rel, body in files.items():
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(body, bytes):
                path.write_bytes(body)
            else:
                write(path, str(body))
            git(root, "add", rel)
        git(root, "commit", "-m", "change module contents")
        return git(root, "rev-parse", "HEAD")

    def test_module_zip_refuses_over_budget_before_any_tag(self) -> None:
        rt = load()
        root, _ = self._repo()
        sha = self._commit(root, {
            MODULE_BUDGET_REL: MODULE_BUDGET_GO.format(budget="1 << 20"),
            "docs/blob.txt": b"x" * (2 << 20),
        })
        context = rt.build_context(
            root, version="0.1.0", ref=sha,
            skip_dry_run=True, skip_ci=True, skip_remote_tag=True,
        )
        self.assertFalse(context["ok"])
        errors = "\n".join(context["errors"])
        self.assertIn("module_zip_over_budget", errors)
        self.assertIn("2.0 MiB", errors)
        self.assertIn("1.0 MiB budget", errors)
        self.assertIn("export-ignore", errors)

        result = rt.execute_tag(root, context)
        self.assertFalse(result["ok"])
        self.assertFalse(result["tag_created"])
        self.assertEqual(git(root, "tag", "-l", "v0.1.0"), "")

    def test_module_zip_passes_with_export_ignored_and_nested_bulk(self) -> None:
        rt = load()
        root, _ = self._repo()
        sha = self._commit(root, {
            MODULE_BUDGET_REL: MODULE_BUDGET_GO.format(budget="2 << 20"),
            ".gitattributes": "evidence/** export-ignore\n",
            "pkg/payload.txt": b"p" * (3 << 19),
            "evidence/receipt.bin": b"e" * (3 << 20),
            "sub/go.mod": "module example.com/sub\n",
            "sub/blob.txt": b"s" * (3 << 20),
        })
        check = rt.module_zip_check(root, sha)
        self.assertTrue(check["pass"], check["errors"])
        self.assertEqual(check["verdict"], "WITHIN_BUDGET")

    def test_module_zip_fails_closed_when_budget_cannot_be_read(self) -> None:
        rt = load()
        root, _ = self._repo()
        sha = self._commit(root, {MODULE_BUDGET_REL: "package architest\n"})
        check = rt.module_zip_check(root, sha)
        self.assertFalse(check["pass"])
        self.assertIn("module_zip_unmeasurable", "\n".join(check["errors"]))

        git(root, "rm", "-q", "--", MODULE_BUDGET_REL)
        git(root, "commit", "-m", "drop budget file")
        sha = git(root, "rev-parse", "HEAD")
        check = rt.module_zip_check(root, sha)
        self.assertFalse(check["pass"])
        self.assertIn("is missing at the ref", "\n".join(check["errors"]))

    def test_module_zip_ref_cannot_raise_the_hard_cap(self) -> None:
        rt = load()
        original_cap = rt.MODULE_ZIP_HARD_CAP_BYTES
        rt.MODULE_ZIP_HARD_CAP_BYTES = 1 << 20
        self.addCleanup(setattr, rt, "MODULE_ZIP_HARD_CAP_BYTES", original_cap)
        root, _ = self._repo()
        sha = self._commit(root, {
            MODULE_BUDGET_REL: MODULE_BUDGET_GO.format(budget="600 << 20"),
            "docs/blob.txt": b"x" * (2 << 20),
        })
        check = rt.module_zip_check(root, sha)
        self.assertFalse(check["pass"])
        self.assertIn("module_zip_over_budget", "\n".join(check["errors"]))
        self.assertIn("1.0 MiB", "\n".join(check["errors"]))

        context = rt.build_context(
            root, version="0.1.0", ref=sha,
            skip_dry_run=True, skip_ci=True, skip_remote_tag=True,
        )
        self.assertFalse(context["ok"])
        self.assertIn("module_zip_over_budget", "\n".join(context["errors"]))

    def test_module_zip_only_cli_refuses_before_publish(self) -> None:
        root, _ = self._repo()
        sha = self._commit(root, {
            MODULE_BUDGET_REL: MODULE_BUDGET_GO.format(budget="1 << 20"),
            "docs/blob.txt": b"x" * (2 << 20),
        })
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--module-zip-only", "--ref", sha, "--json"],
            cwd=root, text=True, encoding="utf-8", capture_output=True,
        )
        self.assertEqual(proc.returncode, 1, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertFalse(payload["ok"])
        self.assertIn("module_zip_over_budget", "\n".join(payload["errors"]))

    def test_workflow_measures_module_zip_before_publishing_trunk(self) -> None:
        text = (ROOT / ".github" / "workflows" / "release-cadence.yml").read_text(encoding="utf-8")
        cut = text.index("python tools/release_cut.py --execute")
        refuse = text.index("--module-zip-only")
        publish = text.index("python tools/fleet_control_pane.py publish --apply")
        self.assertLess(cut, refuse)
        self.assertLess(refuse, publish)
        # The measure runs inside the cut-and-push step, before every publish attempt,
        # so a refused measure must exit the step before any trunk push.
        start = text.index("      - name: Cut and push release commit")
        block = text[start:text.index("      - name: Dispatch release CI", start)]
        self.assertNotIn("continue-on-error", block)
        self.assertGreater(refuse, start)
        self.assertIn("exit 1", text[refuse:publish])

    def test_module_zip_per_file_limit_is_capped_by_ref_raise(self) -> None:
        rt = load()
        root, _ = self._repo()
        sha = self._commit(root, {
            MODULE_BUDGET_REL: MODULE_BUDGET_GO.format(budget="1 << 30").replace(
                "moduleZipNonGoFileLimit = 5 << 20", "moduleZipNonGoFileLimit = 1 << 30",
            ),
            "docs/bulk.tsv": b"x" * (6 << 20),
        })
        check = rt.module_zip_check(root, sha)
        self.assertFalse(check["pass"])
        self.assertIn("module_zip_file_over_limit", "\n".join(check["errors"]))
        self.assertIn("5.0 MiB per-file limit", "\n".join(check["errors"]))

    def test_fold_ci_runs_classifies_green_red_pending_and_no_signal(self) -> None:
        rt = load()
        self.assertEqual(rt.fold_ci_runs([], require_ci=False)["verdict"], "NO_SIGNAL")
        self.assertFalse(rt.fold_ci_runs([], require_ci=True)["pass"])
        self.assertTrue(rt.fold_ci_runs([{"event": "push", "status": "completed", "conclusion": "success"}], require_ci=True)["pass"])
        red = rt.fold_ci_runs([{"event": "push", "status": "completed", "conclusion": "failure"}], require_ci=False)
        self.assertFalse(red["pass"])
        self.assertEqual(red["verdict"], "RED")
        pending = rt.fold_ci_runs([{"event": "push", "status": "in_progress"}], require_ci=False)
        self.assertTrue(pending["pass"])
        self.assertEqual(pending["verdict"], "PENDING")

    def test_fold_ci_runs_requires_push_event(self) -> None:
        rt = load()
        for event in ("workflow_dispatch", "pull_request", "schedule", "unknown", "future_event2"):
            diagnostic = {"event": event, "status": "completed", "conclusion": "success"}
            with self.subTest(event=event):
                for require_ci in (False, True):
                    folded = rt.fold_ci_runs([diagnostic], require_ci=require_ci)
                    self.assertEqual(folded["verdict"], "NO_SIGNAL")
                    self.assertEqual(folded["pass"], not require_ci)
                    self.assertEqual(folded["advisory"], not require_ci)
                    self.assertNotIn("run", folded)
                push = {"event": "push", "status": "completed", "conclusion": "failure"}
                folded = rt.fold_ci_runs([diagnostic, push], require_ci=True)
                self.assertEqual(folded["verdict"], "RED")
                self.assertEqual(folded["run"], push)
        push = {"event": "push", "status": "completed", "conclusion": "success"}
        diagnostic = {"event": "workflow_dispatch", "status": "completed", "conclusion": "failure"}
        self.assertEqual(rt.fold_ci_runs([diagnostic, push], require_ci=True)["verdict"], "GREEN")
        pending = {"event": "push", "status": "in_progress"}
        self.assertEqual(rt.fold_ci_runs([pending, push], require_ci=True)["verdict"], "PENDING")

    def test_ci_runs_refuse_unreadable_events_before_filtering(self) -> None:
        # Pure folds and stubbed runners add less than 20 ms; no Git or gh calls.
        rt = load()
        green = {"event": "push", "status": "completed", "conclusion": "success"}
        failure = {"status": "completed", "conclusion": "failure"}
        malformed = [failure, None, "invalid", []]
        malformed += [dict(failure, event=event) for event in
                      (None, "", " ", "PUSH", " push", "push ", "push\npush", "push\x00",
                       "push\tpush", "push\rpush", "push push", "push-push", "push.push",
                       "püsh", "push\u200b", "_push", "1push", [], {}, True, 1)]
        for bad in malformed:
            for rows in ([bad], [bad, green], [green, bad]):
                for require_ci in (False, True):
                    with self.subTest(rows=rows, require_ci=require_ci):
                        folded = rt.fold_ci_runs(rows, require_ci=require_ci)
                        self.assertEqual(folded["verdict"], "INVALID_EVENT")
                        self.assertFalse(folded["pass"])
                        self.assertFalse(folded["advisory"])
                        self.assertIn("event", folded["reason"])
                        self.assertNotIn("run", folded)

        for wait_for_pending in (False, True):
            for require_ci in (False, True):
                with self.subTest(wait_for_pending=wait_for_pending, require_ci=require_ci):
                    calls = []

                    def runner(cmd: list[str]) -> tuple[int, str]:
                        calls.append(cmd)
                        if wait_for_pending and len(calls) == 1:
                            self.assertEqual(cmd[:3], ["gh", "run", "list"])
                            return 0, json.dumps([{"event": "push", "status": "in_progress",
                                                   "databaseId": "push-1"}])
                        if wait_for_pending and len(calls) == 2:
                            self.assertEqual(cmd, ["gh", "run", "watch", "push-1", "--exit-status"])
                            return 0, "finished"
                        self.assertEqual(len(calls), 3 if wait_for_pending else 1)
                        self.assertEqual(cmd[:3], ["gh", "run", "list"])
                        return 0, json.dumps([failure, green])

                    folded = rt.ci_signal(Path("repo"), "abc123", workflow="ci.yml",
                                          require_ci=require_ci, wait=True, runner=runner)
                    self.assertEqual(folded["verdict"], "INVALID_EVENT")
                    self.assertFalse(folded["pass"])
                    self.assertFalse(folded["advisory"])
                    self.assertEqual(bool(folded.get("watched")), wait_for_pending)
                    self.assertEqual(len(calls), 3 if wait_for_pending else 1)

    def test_ci_signal_requires_array_before_and_after_watch(self) -> None:
        # Pure folds and stubbed runners add less than 20 ms; no Git or gh calls.
        rt = load()
        for response in (None, {}, False, 0, "push", []):
            for wait_for_pending in (False, True):
                for require_ci in (False, True):
                    with self.subTest(response=response, wait=wait_for_pending, require_ci=require_ci):
                        calls = []

                        def runner(cmd: list[str]) -> tuple[int, str]:
                            calls.append(cmd)
                            if wait_for_pending and len(calls) == 1:
                                self.assertEqual(cmd[:3], ["gh", "run", "list"])
                                return 0, json.dumps([{"event": "push", "status": "in_progress",
                                                       "databaseId": "push-1"}])
                            if wait_for_pending and len(calls) == 2:
                                self.assertEqual(cmd, ["gh", "run", "watch", "push-1", "--exit-status"])
                                return 0, "finished"
                            self.assertEqual(len(calls), 3 if wait_for_pending else 1)
                            self.assertEqual(cmd[:3], ["gh", "run", "list"])
                            return 0, json.dumps(response)

                        folded = rt.ci_signal(Path("repo"), "abc123", workflow="ci.yml",
                                              require_ci=require_ci, wait=True, runner=runner)
                        valid_empty = isinstance(response, list)
                        self.assertEqual(folded["verdict"], "NO_SIGNAL" if valid_empty else "INVALID_RESPONSE")
                        self.assertEqual(folded["pass"], valid_empty and not require_ci)
                        self.assertEqual(folded["advisory"], valid_empty and not require_ci)
                        self.assertEqual(bool(folded.get("watched")), wait_for_pending)
                        self.assertNotIn("run", folded)
                        self.assertEqual(len(calls), 3 if wait_for_pending else 1)
                        if not valid_empty:
                            self.assertIn("JSON array", folded["reason"])

    def test_ci_signal_filters_events_before_and_after_watch(self) -> None:
        rt = load()
        for conclusion, expected in ((None, "NO_SIGNAL"), ("success", "GREEN"), ("failure", "RED")):
            with self.subTest(conclusion=conclusion):
                calls = []
                lists = 0
                diagnostic = {"event": "workflow_dispatch", "status": "completed",
                              "conclusion": "success", "databaseId": "manual"}

                def runner(cmd: list[str]) -> tuple[int, str]:
                    nonlocal lists
                    calls.append(cmd)
                    if cmd[:3] == ["gh", "run", "list"]:
                        lists += 1
                        rows = [diagnostic]
                        if lists == 1:
                            rows.append({"event": "push", "status": "in_progress", "databaseId": "push-1"})
                        elif conclusion is not None:
                            rows.append({"event": "push", "status": "completed",
                                         "conclusion": conclusion, "databaseId": "push-1"})
                        return 0, json.dumps(rows)
                    if cmd == ["gh", "run", "watch", "push-1", "--exit-status"]:
                        return 0, "finished"
                    if cmd == ["gh", "run", "view", "push-1", "--json", "jobs"]:
                        return 0, json.dumps({"jobs": []})
                    raise AssertionError(cmd)

                folded = rt.ci_signal(Path("repo"), "abc123", workflow="ci.yml",
                                      require_ci=True, wait=True, runner=runner)
                self.assertEqual(folded["verdict"], expected)
                self.assertEqual(folded["pass"], conclusion == "success")
                self.assertTrue(folded["watched"])
                self.assertEqual(lists, 2)
                for cmd in calls:
                    if cmd[:3] == ["gh", "run", "list"]:
                        self.assertIn("--event", cmd)
                        self.assertEqual(cmd[cmd.index("--event") + 1], "push")
                        self.assertIn("event", cmd[cmd.index("--json") + 1].split(","))

    def test_diagnostic_success_cannot_create_required_ci_tag(self) -> None:
        rt = load()
        root, sha = self._repo()
        rt.load_release_dry_run = lambda root, ref: {"ok": True, "exit_code": 0, "trailer": "release-dry-run: fixture"}
        original_ci_signal = rt.ci_signal

        def runner(cmd: list[str]) -> tuple[int, str]:
            self.assertEqual(cmd[:3], ["gh", "run", "list"])
            return 0, json.dumps([{"event": "workflow_dispatch", "status": "completed",
                                   "conclusion": "success", "headSha": sha,
                                   "databaseId": "manual", "release_eligible": False}])

        rt.ci_signal = lambda root, sha, **kwargs: original_ci_signal(root, sha, runner=runner, **kwargs)
        context = rt.build_context(root, version="0.1.0", ref=sha, trunk="master",
                                   require_ci=True, wait_ci=True, skip_remote_tag=True)
        result = rt.execute_tag(root, context, use_lock=False)
        self.assertFalse(result["tag_created"], result)
        self.assertFalse(context["ok"], context)
        self.assertIn("ci_not_green:NO_SIGNAL", context["errors"])
        self.assertNotIn("v0.1.0", git(root, "tag", "-l"))

    # Release-dispatch evidence: pure folds over stubbed gh rows, no Git or network.
    SHA = "a" * 40
    HEAD = "c" * 40

    def _dispatch_row(self, **overrides) -> dict:
        row = {"event": "workflow_dispatch", "headBranch": "main", "headSha": self.HEAD,
               "status": "completed", "conclusion": "success", "databaseId": 4242, "attempt": 1,
               "displayTitle": f"ci-fast release · {self.SHA}"}
        row.update(overrides)
        return row

    def _provenance(self, **overrides) -> dict:
        prov = {"schema": "fak.ci-provenance.v1", "workflow": "ci-fast", "verified": True,
                "ci_class": "release", "requested_intent": "release", "event": "workflow_dispatch",
                "ref": "refs/heads/main", "tested_sha": self.SHA, "tested_tree": "b" * 40,
                "represented_sha": self.HEAD, "run_id": "4242", "run_attempt": "1",
                "required_jobs": ["build-vet-fast"] + [f"go-test-shard-{index}" for index in range(14)],
                "release_eligible": False}
        prov.update(overrides)
        return prov

    def _dispatch_signal(self, rt, rows, provenance, *, blobs=None, wait=False, sleeps=None):
        blobs = blobs if blobs is not None else {self.SHA: "blob1", self.HEAD: "blob1"}
        calls: list[list[str]] = []
        rows_iter = list(rows) if isinstance(rows, list) and rows and isinstance(rows[0], list) else None

        def runner(cmd: list[str]) -> tuple[int, str]:
            calls.append(cmd)
            if cmd[:3] == ["gh", "run", "list"]:
                self.assertEqual(cmd[cmd.index("--event") + 1], "workflow_dispatch")
                self.assertEqual(cmd[cmd.index("--branch") + 1], "main")
                if rows_iter is not None:
                    return 0, json.dumps(rows_iter.pop(0) if len(rows_iter) > 1 else rows_iter[0])
                return 0, json.dumps(rows)
            if cmd[:3] == ["gh", "run", "watch"]:
                return 0, "done"
            raise AssertionError(cmd)

        folded = rt.release_dispatch_signal(
            Path("repo"), self.SHA, workflow="ci-fast.yml", trunk="main", require_ci=True,
            wait=wait, runner=runner, fetch_provenance=lambda row: provenance,
            blob_at=lambda rev: blobs.get(rev), sleep=(sleeps.append if sleeps is not None else lambda s: None))
        return folded, calls

    def test_release_dispatch_accepts_exact_sha_verified_provenance(self) -> None:
        rt = load()
        folded, _ = self._dispatch_signal(rt, [self._dispatch_row()], self._provenance())
        self.assertEqual(folded["verdict"], "GREEN", folded)
        self.assertTrue(folded["pass"])
        self.assertEqual(folded["evidence"], "release-dispatch")

    def test_release_dispatch_rejects_wrong_sha_or_foreign_binding(self) -> None:
        rt = load()
        for name, prov, row, blobs in (
            ("tested another commit", self._provenance(tested_sha="d" * 40), self._dispatch_row(), None),
            ("represented sha is not the run head", self._provenance(represented_sha="e" * 40), self._dispatch_row(), None),
            ("other run's provenance", self._provenance(run_id="9999"), self._dispatch_row(), None),
            ("stale attempt", self._provenance(run_attempt="2"), self._dispatch_row(), None),
            ("unverified aggregate", self._provenance(verified=False), self._dispatch_row(), None),
            ("missing shard job", self._provenance(required_jobs=["build-vet-fast"]), self._dispatch_row(), None),
            ("off-trunk run", self._provenance(ref="refs/heads/feature"), self._dispatch_row(headBranch="feature"), None),
            ("workflow differs at run head", self._provenance(), self._dispatch_row(),
             {self.SHA: "blob1", self.HEAD: "blob2"}),
            ("workflow unreadable", self._provenance(), self._dispatch_row(), {}),
            ("no provenance", None, self._dispatch_row(), None),
        ):
            with self.subTest(name):
                folded, _ = self._dispatch_signal(rt, [row], prov, blobs=blobs)
                self.assertEqual(folded["verdict"], "UNVERIFIED", folded)
                self.assertFalse(folded["pass"])
        # A run titled for another sha is never selected.
        folded, _ = self._dispatch_signal(
            rt, [self._dispatch_row(displayTitle=f"ci-fast release · {'d' * 40}")], self._provenance())
        self.assertEqual(folded["verdict"], "NO_SIGNAL")
        self.assertFalse(folded["pass"])

    def test_release_dispatch_rejects_failed_or_cancelled_run(self) -> None:
        rt = load()
        for conclusion in ("failure", "cancelled", "timed_out", None):
            with self.subTest(conclusion=conclusion):
                folded, _ = self._dispatch_signal(
                    rt, [self._dispatch_row(conclusion=conclusion)], self._provenance())
                self.assertEqual(folded["verdict"], "RED", folded)
                self.assertFalse(folded["pass"])

    def test_release_dispatch_rejects_plain_diagnostic_dispatch(self) -> None:
        rt = load()
        for prov in (self._provenance(ci_class="diagnostic"),
                     self._provenance(ci_class="diagnostic", requested_intent="diagnostic"),
                     self._provenance(ci_class="fast")):
            with self.subTest(prov["ci_class"], intent=prov["requested_intent"]):
                folded, _ = self._dispatch_signal(rt, [self._dispatch_row()], prov)
                self.assertEqual(folded["verdict"], "UNVERIFIED", folded)
                self.assertFalse(folded["pass"])
        # A diagnostic run's title never names the release form.
        folded, _ = self._dispatch_signal(
            rt, [self._dispatch_row(displayTitle=f"ci-fast diagnostic (non-release) · {self.HEAD}")],
            self._provenance())
        self.assertEqual(folded["verdict"], "NO_SIGNAL")

    def test_release_dispatch_waits_for_appearance_and_completion(self) -> None:
        rt = load()
        sleeps: list[float] = []
        pending = self._dispatch_row(status="in_progress", conclusion="")
        folded, calls = self._dispatch_signal(
            rt, [[], [pending], [self._dispatch_row()]], self._provenance(), wait=True, sleeps=sleeps)
        self.assertEqual(folded["verdict"], "GREEN", folded)
        self.assertEqual(len(sleeps), 1)
        self.assertIn(["gh", "run", "watch", "4242", "--exit-status"], calls)

    def test_push_run_evidence_unchanged_with_release_dispatch_enabled(self) -> None:
        rt = load()
        for conclusion, expected in (("success", "GREEN"), ("failure", "RED")):
            with self.subTest(conclusion=conclusion):
                def runner(cmd: list[str]) -> tuple[int, str]:
                    if cmd[:3] == ["gh", "run", "list"]:
                        self.assertEqual(cmd[cmd.index("--event") + 1], "push")
                        return 0, json.dumps([{"event": "push", "status": "completed",
                                               "conclusion": conclusion, "databaseId": "push-1"}])
                    if cmd == ["gh", "run", "view", "push-1", "--json", "jobs"]:
                        return 0, json.dumps({"jobs": []})
                    raise AssertionError(cmd)

                def no_dispatch(*args, **kwargs):
                    raise AssertionError("a push signal must not consult release dispatches")

                folded = rt.ci_signal(Path("repo"), self.SHA, workflow="ci-fast.yml", require_ci=True,
                                      runner=runner, release_dispatch=True, dispatch_signal=no_dispatch)
                self.assertEqual(folded["verdict"], expected)
                self.assertNotIn("evidence", folded)

    def test_release_dispatch_only_consulted_when_enabled_and_no_push_run(self) -> None:
        rt = load()
        consulted = []

        def runner(cmd: list[str]) -> tuple[int, str]:
            return 0, "[]"

        def dispatch(root, sha, **kwargs):
            consulted.append(kwargs["trunk"])
            return {"pass": True, "verdict": "GREEN", "advisory": False, "evidence": "release-dispatch"}

        off = rt.ci_signal(Path("repo"), self.SHA, workflow="ci-fast.yml", require_ci=True,
                           runner=runner, dispatch_signal=dispatch)
        self.assertEqual(off["verdict"], "NO_SIGNAL")
        self.assertEqual(consulted, [])
        on = rt.ci_signal(Path("repo"), self.SHA, workflow="ci-fast.yml", require_ci=True,
                          runner=runner, release_dispatch=True, trunk="main", dispatch_signal=dispatch)
        self.assertEqual(on["verdict"], "GREEN")
        self.assertEqual(on["push_signal"]["verdict"], "NO_SIGNAL")
        self.assertEqual(consulted, ["main"])

    def test_ci_signal_diagnoses_billing_annotation_on_red_run(self) -> None:
        rt = load()

        def runner(cmd: list[str]) -> tuple[int, str]:
            if cmd[:5] == ["gh", "run", "list", "--workflow", "ci.yml"]:
                return 0, json.dumps([{
                    "event": "push",
                    "databaseId": "run-1",
                    "status": "completed",
                    "conclusion": "failure",
                    "url": "https://example.invalid/run",
                }])
            if cmd == ["gh", "run", "view", "run-1", "--json", "jobs"]:
                return 0, json.dumps({
                    "jobs": [{
                        "databaseId": "job-1",
                        "name": "build",
                        "status": "completed",
                        "conclusion": "failure",
                        "steps": [],
                    }]
                })
            if cmd == ["gh", "api", "repos/:owner/:repo/check-runs/job-1/annotations"]:
                return 0, json.dumps([{
                    "annotation_level": "failure",
                    "path": ".github",
                    "message": (
                        "The job was not started because recent account payments have failed "
                        "or your spending limit needs to be increased."
                    ),
                }])
            raise AssertionError(cmd)

        folded = rt.ci_signal(Path("repo"), "abc123", workflow="ci.yml", require_ci=True, runner=runner)

        self.assertFalse(folded["pass"])
        self.assertEqual(folded["verdict"], "CI_BILLING")
        self.assertEqual(folded["diagnosis"]["action"], "fix_ci_billing")
        self.assertEqual(folded["diagnosis"]["jobs"][0]["steps"], 0)

    def test_context_checks_committed_version_and_release_note(self) -> None:
        rt = load()
        root, sha = self._repo()
        context = rt.build_context(
            root,
            version="0.1.0",
            ref=sha,
            skip_dry_run=True,
            skip_ci=True,
            skip_remote_tag=True,
        )
        self.assertTrue(context["ok"], context["errors"])
        self.assertTrue(context["checks"]["version_at_ref"]["pass"])
        self.assertTrue(context["checks"]["release_note_at_ref"]["pass"])
        self.assertIn("release_dry_run_skipped", context["idempotent_skips"])

    def test_context_blocks_version_mismatch_and_missing_release_note(self) -> None:
        rt = load()
        root, sha = self._repo()
        context = rt.build_context(
            root,
            version="0.2.0",
            ref=sha,
            skip_dry_run=True,
            skip_ci=True,
            skip_remote_tag=True,
        )
        self.assertFalse(context["ok"])
        self.assertIn("version_mismatch_at_ref:0.1.0", context["errors"])
        self.assertIn("missing_release_note:docs/releases/v0.2.0.md", context["errors"])

    def test_execute_creates_annotated_tag_and_is_idempotent(self) -> None:
        rt = load()
        root, sha = self._repo()
        context = rt.build_context(
            root,
            version="0.1.0",
            ref=sha,
            skip_dry_run=True,
            skip_ci=True,
            skip_remote_tag=True,
        )
        first = rt.execute_tag(root, context)
        self.assertTrue(first["ok"], first["errors"])
        self.assertTrue(first["tag_created"])
        self.assertEqual(git(root, "rev-list", "-n1", "v0.1.0"), sha)
        tag_body = git(root, "tag", "-l", "v0.1.0", "--format=%(contents)")
        self.assertIn("release-dry-run: skipped", tag_body)

        second_context = rt.build_context(
            root,
            version="0.1.0",
            ref=sha,
            skip_dry_run=True,
            skip_ci=True,
            skip_remote_tag=True,
        )
        second = rt.execute_tag(root, second_context)
        self.assertTrue(second["ok"], second["errors"])
        self.assertFalse(second["tag_created"])
        self.assertIn("tag_create_skipped_existing_same_sha", second["idempotent_skips"])

    def test_execute_uses_release_lock_when_present(self) -> None:
        rt = load()
        root, sha = self._repo()
        self._write_lock_helper(root)
        context = rt.build_context(
            root,
            version="0.1.0",
            ref=sha,
            skip_dry_run=True,
            skip_ci=True,
            skip_remote_tag=True,
        )
        result = rt.execute_tag(root, context)
        self.assertTrue(result["ok"], result["errors"])
        self.assertTrue(result["release_lock"]["held"])
        self.assertTrue(result["release_lock_release"]["ok"])
        self.assertEqual(
            (root / "lock.log").read_text(encoding="utf-8").splitlines(),
            ["acquire:fallback-owner", "release:fallback-owner"],
        )

    def test_execute_uses_parent_release_lock_without_releasing_it(self) -> None:
        rt = load()
        root, sha = self._repo()
        self._write_lock_helper(root)
        context = rt.build_context(
            root,
            version="0.1.0",
            ref=sha,
            skip_dry_run=True,
            skip_ci=True,
            skip_remote_tag=True,
        )
        result = rt.execute_tag(root, context, lock_already_held=True)

        self.assertTrue(result["ok"], result["errors"])
        self.assertTrue(result["tag_created"])
        self.assertTrue(result["release_lock"]["held_by_parent"])
        self.assertEqual(
            (root / "lock.log").read_text(encoding="utf-8").splitlines(),
            ["verify:fallback-owner"],
        )

    def test_execute_refuses_parent_lock_owned_by_another_session(self) -> None:
        rt = load()
        root, sha = self._repo()
        write(root / "tools" / "release_lock.py", (
            "import json, pathlib, sys\n"
            "cmd = sys.argv[1]\n"
            "p = pathlib.Path('lock.log')\n"
            "old = p.read_text(encoding='utf-8') if p.exists() else ''\n"
            "p.write_text(old + cmd + '\\n', encoding='utf-8')\n"
            "print(json.dumps({'ok': False, 'reason': 'held by another session'}))\n"
            "sys.exit(3)\n"
        ))
        context = rt.build_context(
            root,
            version="0.1.0",
            ref=sha,
            skip_dry_run=True,
            skip_ci=True,
            skip_remote_tag=True,
        )

        result = rt.execute_tag(root, context, lock_already_held=True)

        self.assertFalse(result["ok"], result)
        self.assertIn("release_lock_verify_failed", result["errors"])
        self.assertFalse(result["tag_created"])
        self.assertEqual((root / "lock.log").read_text(encoding="utf-8").splitlines(), ["verify"])
        self.assertNotIn("v0.1.0", git(root, "tag", "-l"))

    def _write_lock_helper(self, root: Path) -> None:
        write(root / "tools" / "release_lock.py", (
            "import json, pathlib, sys\n"
            "cmd = sys.argv[1]\n"
            "owner = sys.argv[sys.argv.index('--owner') + 1] if '--owner' in sys.argv else 'fallback-owner'\n"
            "p = pathlib.Path('lock.log')\n"
            "old = p.read_text(encoding='utf-8') if p.exists() else ''\n"
            "p.write_text(old + cmd + ':' + owner + '\\n', encoding='utf-8')\n"
            "payload = {'ok': True, 'cmd': cmd}\n"
            "if cmd in ('acquire', 'verify'):\n"
            "    payload['lock'] = {'owner': owner}\n"
            "print(json.dumps(payload))\n"
        ))

    def test_tag_collision_blocks_different_commit(self) -> None:
        rt = load()
        root, first_sha = self._repo()
        git(root, "tag", "-a", "v0.1.0", first_sha, "-m", "v0.1.0")
        write(root / "docs" / "note.md", "change\n")
        git(root, "add", "docs/note.md")
        git(root, "commit", "-m", "docs: later")
        context = rt.build_context(
            root,
            version="0.1.0",
            ref="HEAD",
            skip_dry_run=True,
            skip_ci=True,
            skip_remote_tag=True,
        )
        self.assertFalse(context["ok"])
        self.assertIn("local_tag_exists_on_different_commit", context["errors"])

    def test_live_cli_dry_run_no_mutation(self) -> None:
        before = subprocess.run(
            ["git", "tag", "-l"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        ).stdout
        version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
        proc = subprocess.run(
            [
                sys.executable, str(SCRIPT),
                "--version", version,
                "--ref", "HEAD",
                "--skip-dry-run",
                "--skip-ci",
                "--skip-remote-tag",
                "--json",
            ],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            capture_output=True,
        )
        self.assertIn(proc.returncode, (0, 1), proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertIn("ok", payload)
        after = subprocess.run(
            ["git", "tag", "-l"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        ).stdout
        self.assertEqual(after, before)

    def _chain(self, length: int) -> tuple[Path, list[str]]:
        """Build a first-parent chain; returns shas newest first."""
        root, _ = self._repo()
        for i in range(length - 1):
            write(root / "touch.txt", f"{i}\n")
            git(root, "add", "touch.txt")
            git(root, "commit", "-m", f"change {i}")
        shas = git(root, "rev-list", "--first-parent", "HEAD").splitlines()
        return root, shas

    @staticmethod
    def _fake_ci(verdicts: dict[str, str]):
        def fake(root, sha, *, workflow, require_ci, wait=False, runner=None,
                 release_dispatch=False, trunk="main"):
            verdict = verdicts.get(sha, "NO_SIGNAL")
            passed = verdict == "GREEN"
            return {"pass": passed, "verdict": verdict, "advisory": False}
        return fake

    def test_candidate_skips_coalesced_head_to_newest_green_ancestor(self) -> None:
        rt = load()
        root, shas = self._chain(3)
        original = rt.ci_signal
        self.addCleanup(setattr, rt, "ci_signal", original)
        rt.ci_signal = self._fake_ci({shas[1]: "GREEN"})
        selection = rt.select_green_candidate(root, shas[0], workflow="ci.yml")
        self.assertTrue(selection["pass"])
        self.assertEqual(selection["sha"], shas[1])
        self.assertEqual([row["verdict"] for row in selection["scanned"]], ["NO_SIGNAL", "GREEN"])

    def test_candidate_tolerates_pending_newer_run_but_blocks_on_red(self) -> None:
        rt = load()
        root, shas = self._chain(3)
        original = rt.ci_signal
        self.addCleanup(setattr, rt, "ci_signal", original)
        rt.ci_signal = self._fake_ci({shas[0]: "PENDING", shas[1]: "GREEN"})
        self.assertEqual(rt.select_green_candidate(root, shas[0], workflow="ci.yml")["sha"], shas[1])
        rt.ci_signal = self._fake_ci({shas[0]: "PENDING", shas[1]: "RED", shas[2]: "GREEN"})
        blocked = rt.select_green_candidate(root, shas[0], workflow="ci.yml")
        self.assertFalse(blocked["pass"])
        self.assertIsNone(blocked["sha"])
        self.assertEqual(blocked["ci"]["verdict"], "RED")

    def test_candidate_window_is_bounded_and_refuses_when_no_green(self) -> None:
        rt = load()
        root, shas = self._chain(4)
        original = rt.ci_signal
        self.addCleanup(setattr, rt, "ci_signal", original)
        rt.ci_signal = self._fake_ci({shas[3]: "GREEN"})
        beyond = rt.select_green_candidate(root, shas[0], workflow="ci.yml", depth=3)
        self.assertFalse(beyond["pass"])
        self.assertEqual(beyond["ci"]["verdict"], "NO_SIGNAL")
        self.assertEqual(len(beyond["scanned"]), 3)
        within = rt.select_green_candidate(root, shas[0], workflow="ci.yml", depth=4)
        self.assertEqual(within["sha"], shas[3])

    def test_gated_context_never_tags_green_ancestor_for_uncovered_cut(self) -> None:
        rt = load()
        root, shas = self._chain(2)
        original = rt.ci_signal
        self.addCleanup(setattr, rt, "ci_signal", original)
        rt.ci_signal = self._fake_ci({shas[1]: "GREEN"})
        context = rt.build_context(
            root,
            version="0.1.0",
            ref="HEAD",
            trunk="master",
            require_ci=True,
            skip_dry_run=True,
            skip_remote_tag=True,
        )
        self.assertFalse(context["ok"])
        self.assertEqual(context["sha"], shas[0])
        self.assertIn("ci_not_green:NO_SIGNAL", context["errors"])
        self.assertFalse(context["checks"]["ci"]["pass"])
        self.assertEqual(context["cut_base"]["sha"], shas[1])
        self.assertNotIn("release_candidate", context)

    def test_gated_context_tags_cut_only_with_its_own_green_run(self) -> None:
        rt = load()
        root, shas = self._chain(2)
        original = rt.ci_signal
        self.addCleanup(setattr, rt, "ci_signal", original)
        rt.ci_signal = self._fake_ci({shas[0]: "GREEN", shas[1]: "GREEN"})
        context = rt.build_context(
            root,
            version="0.1.0",
            ref="HEAD",
            trunk="master",
            require_ci=True,
            skip_dry_run=True,
            skip_remote_tag=True,
        )
        self.assertTrue(context["ok"], context["errors"])
        self.assertEqual(context["sha"], shas[0])
        self.assertTrue(context["checks"]["ci"]["pass"])

    def test_gated_context_refuses_when_no_green_candidate(self) -> None:
        rt = load()
        root, shas = self._chain(2)
        original = rt.ci_signal
        self.addCleanup(setattr, rt, "ci_signal", original)
        rt.ci_signal = self._fake_ci({})
        context = rt.build_context(
            root,
            version="0.1.0",
            ref="HEAD",
            trunk="master",
            require_ci=True,
            skip_dry_run=True,
            skip_remote_tag=True,
        )
        self.assertFalse(context["ok"])
        self.assertIn("ci_not_green:NO_SIGNAL", context["errors"])
        self.assertEqual(context["cut_base"]["sha"], None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
