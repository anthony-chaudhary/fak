#!/usr/bin/env python3
"""Hermetic tests for the fleet registry-dir resolver.

Python twin of internal/accountprobe/regdir_test.go. Every rung the resolver surveys --
$FLEET_REG_DIR, $FLEET_STATE_DIR, LOCALAPPDATA, the temp root, the clone root -- is pinned
inside a temp dir, so no test here ever stats, let alone reads or writes, the operator's
own fleet registry. That pinning is not politeness: the whole defect under test is a
resolver that reaches a host dir it should not, and a test that reached the real one would
be a fact about the machine it ran on.
"""
from __future__ import annotations

import importlib
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT / "tools"))

import fleet_regdir  # noqa: E402


class RegFixture:
    """The three registries a real host can end up with, all inside one temp dir."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.user_reg = root / "localappdata" / "Fleet" / "registry"
        self.temp_reg = root / "temproot" / "Fleet" / "registry"
        self.clone_reg = root / "clone" / "tools" / "_registry"
        self.state_reg = root / "statedir" / "registry"


def write_registry(path: Path, *, ledger: bool) -> Path:
    """Lay down a registry dir. ledger=False reproduces the exact fork shape: sessions.json
    present, probe_ledger.jsonl absent, so no block is derivable and the dir reports a
    fleet with zero blocked seats."""
    path.mkdir(parents=True, exist_ok=True)
    (path / fleet_regdir.SESSIONS_FILE).write_text(
        '{"generated_utc":"2026-07-28T12:54:04Z","app_version":"0.43.0","sessions":[]}',
        encoding="utf-8")
    if ledger:
        (path / fleet_regdir.LEDGER_FILE).write_text(
            '{"ts":"2026-07-28T12:54:00Z","account":"acct-a","status":"AUTH"}\n',
            encoding="utf-8")
    return path


class RegDirTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.fx = RegFixture(Path(tmp.name))
        for d in ("localappdata", "temproot", "clone", "statedir"):
            (self.fx.root / d).mkdir()
        env = mock.patch.dict(os.environ, {
            "FLEET_REG_DIR": "",     # the unset case this defect is about
            "FLEET_STATE_DIR": "",   # no installed-service declaration either
            "LOCALAPPDATA": str(self.fx.root / "localappdata"),
            "TMP": str(self.fx.root / "temproot"),
            "TEMP": str(self.fx.root / "temproot"),
        })
        env.start()
        self.addCleanup(env.stop)
        # tempfile caches its answer in a module global on first use, so repointing TMP/TEMP
        # is not enough to move the temp rung -- pin the cache itself or the rung would name
        # the operator's real temp root.
        temproot = mock.patch.object(tempfile, "tempdir", str(self.fx.root / "temproot"))
        temproot.start()
        self.addCleanup(temproot.stop)

    def resolve(self):
        return fleet_regdir.resolve(str(self.fx.clone_reg))

    # -- the headline -------------------------------------------------------------------

    def test_prefers_ledger_bearing_registry(self) -> None:
        """With FLEET_REG_DIR unset and both registries on disk, the one that can derive a
        block wins. Before this resolver the clone-root dir won by being the hardcoded
        default, and it is precisely the one with no ledger."""
        write_registry(self.fx.user_reg, ledger=True)
        write_registry(self.fx.clone_reg, ledger=False)

        got = self.resolve()

        self.assertEqual(got.dir, str(self.fx.user_reg))
        self.assertEqual(got.rung, fleet_regdir.RUNG_USER)
        self.assertEqual(got.health, fleet_regdir.HEALTH_BLOCKS_KNOWN)
        self.assertTrue(got.blocks_derivable())

    def test_ignores_mtime(self) -> None:
        """Authority, not recency. Two writers run on their own schedules and each is the
        freshest for part of every minute; ordering by mtime would let the choice FLAP
        between ticks. Make the ledger-less clone dir the NEWEST and nothing moves."""
        write_registry(self.fx.user_reg, ledger=True)
        write_registry(self.fx.clone_reg, ledger=False)
        old = 1_600_000_000
        for f in self.fx.user_reg.iterdir():
            os.utime(f, (old, old))
        os.utime(self.fx.user_reg, (old, old))

        self.assertEqual(self.resolve().dir, str(self.fx.user_reg))

    # -- what the chosen dir may honestly say --------------------------------------------

    def test_registry_without_ledger_is_unknown_not_zero_blocks(self) -> None:
        """Absence is not neutral: a dir with sessions.json and no ledger beside it can
        derive no probe verdict, so its "zero blocked seats" means "cannot tell". Grading
        it unknown is what stops a headroom-weighted allocator from ranking a dead seat as
        the emptiest in the fleet."""
        write_registry(self.fx.clone_reg, ledger=False)

        got = self.resolve()

        self.assertEqual(got.dir, str(self.fx.clone_reg))
        self.assertEqual(got.health, fleet_regdir.HEALTH_BLOCKS_UNKNOWN)
        self.assertFalse(got.blocks_derivable())

    # -- the fork the operator has to see ------------------------------------------------

    def test_fork_is_observable(self) -> None:
        write_registry(self.fx.user_reg, ledger=True)
        write_registry(self.fx.clone_reg, ledger=False)

        got = self.resolve()
        note = got.fork_note()

        self.assertTrue(got.forked)
        self.assertIn("FORKED", note)
        self.assertIn(str(self.fx.user_reg), note)
        self.assertIn(str(self.fx.clone_reg), note)
        self.assertIn(fleet_regdir.HEALTH_BLOCKS_UNKNOWN, note)
        # Byte-parity with accountprobe.RegChoice.ForkNote: one grep finds both halves.
        self.assertTrue(note.startswith("accountprobe: fleet registry FORKED — reading "), note)

    def test_single_registry_reports_no_fork(self) -> None:
        write_registry(self.fx.user_reg, ledger=True)

        got = self.resolve()

        self.assertFalse(got.forked)
        self.assertEqual(got.fork_note(), "")

    def test_sites_deduplicate_same_path(self) -> None:
        """The ordinary production wiring -- FLEET_REG_DIR pointing AT the per-user Fleet
        registry -- is one site, not a phantom fork."""
        write_registry(self.fx.user_reg, ledger=True)

        with mock.patch.dict(os.environ, {"FLEET_REG_DIR": str(self.fx.user_reg)}):
            got = self.resolve()

        self.assertFalse(got.forked, got.sites)
        self.assertEqual(got.rung, fleet_regdir.RUNG_ENV)

    # -- declared rungs outrank discovery ------------------------------------------------

    def test_explicit_env_wins_even_with_no_state_under_it(self) -> None:
        """An operator (or the fleet launcher) naming the dir is the highest authority
        there is, and every existing wiring depends on FLEET_REG_DIR still winning
        outright -- including on a first boot where nothing is written yet."""
        write_registry(self.fx.user_reg, ledger=True)
        named = self.fx.root / "named"

        with mock.patch.dict(os.environ, {"FLEET_REG_DIR": str(named)}):
            got = self.resolve()

        self.assertEqual(got.dir, str(named))
        self.assertEqual(got.rung, fleet_regdir.RUNG_ENV)
        self.assertEqual(got.health, fleet_regdir.HEALTH_EMPTY)

    def test_state_dir_outranks_discovery(self) -> None:
        write_registry(self.fx.user_reg, ledger=True)

        with mock.patch.dict(os.environ, {"FLEET_STATE_DIR": str(self.fx.root / "statedir")}):
            got = self.resolve()

        self.assertEqual(got.dir, str(self.fx.state_reg))
        self.assertEqual(got.rung, fleet_regdir.RUNG_STATE)

    # -- the conservative floor ----------------------------------------------------------

    def test_falls_back_to_clone_root_when_nothing_exists(self) -> None:
        """A fresh checkout and CI behave exactly as they always have: nothing on disk
        anywhere means the legacy clone-root path, returned unchanged."""
        got = self.resolve()

        self.assertEqual(got.dir, str(self.fx.clone_reg))
        self.assertEqual(got.rung, fleet_regdir.RUNG_CLONE)
        self.assertEqual(got.health, fleet_regdir.HEALTH_EMPTY)
        self.assertFalse(got.forked)

    def test_existing_but_empty_dir_beats_a_nonexistent_one(self) -> None:
        """Rung 5: matching what the Go resolver settles on, so a fak process and a fleet
        script on one host converge instead of each inventing a dir."""
        self.fx.user_reg.mkdir(parents=True)

        got = self.resolve()

        self.assertEqual(got.dir, str(self.fx.user_reg))
        self.assertEqual(got.health, fleet_regdir.HEALTH_EMPTY)

    def test_default_clone_dir_is_file_relative(self) -> None:
        """The clone rung must not follow the cwd: a fleet script run from C:\\work\\job
        names the same tools/_registry it always has, which is what makes the legacy
        default stable rather than a function of who launched it."""
        self.assertEqual(fleet_regdir.default_clone_dir(),
                         os.path.join(str(ROOT / "tools"), "_registry"))


class ModuleWiringTest(unittest.TestCase):
    """The regression this file exists to prevent: a module that rebuilds the two-rung
    fallback by hand forks the host again the moment it runs with FLEET_REG_DIR unset."""

    # `os.environ.get("FLEET_REG_DIR")` with NO default is a plain env read and stays legal;
    # supplying a default (positional or via `or`) is the fallback that forks.
    FALLBACK = re.compile(r"""environ\.get\(\s*["']FLEET_REG_DIR["']\s*,"""
                          r"""|environ\.get\(\s*["']FLEET_REG_DIR["']\s*\)\s*or""")

    def test_no_tools_module_rebuilds_the_two_rung_fallback(self) -> None:
        offenders = []
        for path in sorted((ROOT / "tools").glob("*.py")):
            if path.name in ("fleet_regdir.py",) or path.name.endswith("_test.py"):
                continue
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if self.FALLBACK.search(line):
                    offenders.append(f"{path.name}:{n}: {line.strip()}")
        self.assertEqual(offenders, [], "these must resolve through fleet_regdir instead:\n"
                                        + "\n".join(offenders))

    def test_launch_admission_ledger_follows_the_resolver(self) -> None:
        """End-to-end on the lightest consumer: with the env unset and a ledger-bearing
        per-user registry on disk, the module-level default lands THERE, not in the clone."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            user_reg = root / "localappdata" / "Fleet" / "registry"
            write_registry(user_reg, ledger=True)
            with mock.patch.dict(os.environ, {
                "FLEET_REG_DIR": "", "FLEET_STATE_DIR": "",
                "LOCALAPPDATA": str(root / "localappdata"),
                "TMP": str(root / "temproot"), "TEMP": str(root / "temproot"),
            }), mock.patch.object(tempfile, "tempdir", str(root / "temproot")):
                import launch_admission
                reloaded = importlib.reload(launch_admission)
                try:
                    self.assertEqual(reloaded.DEFAULT_LEDGER,
                                     str(user_reg / "resume_ledger.jsonl"))
                finally:
                    importlib.reload(launch_admission)  # restore the host-resolved default


if __name__ == "__main__":
    unittest.main(verbosity=2)
