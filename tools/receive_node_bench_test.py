#!/usr/bin/env python3
"""Tests for receive_node_bench.py — importing a private benchmark-node bundle.

This tool untars an archive that arrived over the network from another machine,
so ``validate_archive`` is a security boundary, not a formality: without it a
crafted bundle could write outside the node directory (``../``, an absolute path,
a symlink pointing anywhere). Those refusals are the bulk of what is pinned here,
one test per escape shape, because a regression would be invisible until it was
exploited.

The rest pins the import contract around it: the newest matching bundle wins, an
existing node directory is never silently clobbered, and a turn-agent bundle
whose result JSON is missing or incomplete raises with a POINTED message rather
than importing a half-verified result.

All archives are built in a temp directory; nothing here touches the network,
tailscale, or the real experiments tree.
"""
import io
import json
import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import receive_node_bench as M  # noqa: E402


def tmpdir():
    return Path(tempfile.mkdtemp())


def make_tar(path, entries, extra_infos=()):
    """entries: {arcname: file body}; extra_infos: raw TarInfo objects to add."""
    with tarfile.open(path, "w:gz") as tar:
        for arcname, body in entries.items():
            data = body.encode("utf-8")
            info = tarfile.TarInfo(arcname)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        for info in extra_infos:
            tar.addfile(info)
    return path


def opened(path):
    return tarfile.open(path, "r:*")


class ValidateArchive(unittest.TestCase):
    def check_refusal(self, entries, extra_infos=(), needle=""):
        p = make_tar(tmpdir() / "b.tgz", entries, extra_infos)
        with opened(p) as tar:
            with self.assertRaises(ValueError) as cm:
                M.validate_archive(tar, "node-a")
        if needle:
            self.assertIn(needle, str(cm.exception))

    def test_a_well_formed_bundle_is_accepted(self):
        p = make_tar(tmpdir() / "b.tgz", {"node-a/kernel.json": "{}",
                                          "node-a/sub/batch.json": "{}"})
        with opened(p) as tar:
            self.assertEqual(len(M.validate_archive(tar, "node-a")), 2)

    def test_an_empty_archive_is_refused(self):
        self.check_refusal({}, needle="empty")

    def test_a_parent_traversal_member_is_refused(self):
        self.check_refusal({"node-a/../../evil.txt": "x"}, needle="unsafe")

    def test_an_absolute_member_path_is_refused(self):
        info = tarfile.TarInfo("/etc/cron.d/evil")
        info.size = 0
        self.check_refusal({}, extra_infos=[info])

    def test_a_member_outside_the_host_directory_is_refused(self):
        self.check_refusal({"other-node/kernel.json": "{}"}, needle="not under node-a/")

    def test_a_bare_top_level_file_is_refused(self):
        self.check_refusal({"kernel.json": "{}"}, needle="not under node-a/")

    def test_a_symlink_member_is_refused(self):
        link = tarfile.TarInfo("node-a/shortcut")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        self.check_refusal({"node-a/kernel.json": "{}"}, extra_infos=[link],
                           needle="unsupported archive member type")

    def test_the_host_name_is_matched_exactly_not_as_a_prefix(self):
        # `node-a2/` must not be accepted as `node-a/`.
        self.check_refusal({"node-a2/kernel.json": "{}"}, needle="not under node-a/")


class ExtractArchive(unittest.TestCase):
    def bundle(self):
        return make_tar(tmpdir() / "node-a-20260101-000000.tgz",
                        {"node-a/kernel.json": '{"ok": true}'})

    def test_extracts_into_the_named_host_directory(self):
        nodes = tmpdir() / "fleet-nodes"
        dest = M.extract_archive(self.bundle(), nodes, "node-a", replace=False)
        self.assertEqual(dest, nodes / "node-a")
        self.assertEqual(json.loads((dest / "kernel.json").read_text(encoding="utf-8")),
                         {"ok": True})

    def test_an_existing_node_dir_is_never_silently_overwritten(self):
        nodes = tmpdir() / "fleet-nodes"
        (nodes / "node-a").mkdir(parents=True)
        (nodes / "node-a" / "precious.json").write_text("keep me", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            M.extract_archive(self.bundle(), nodes, "node-a", replace=False)
        self.assertTrue((nodes / "node-a" / "precious.json").exists())

    def test_replace_clears_the_previous_import_first(self):
        nodes = tmpdir() / "fleet-nodes"
        (nodes / "node-a").mkdir(parents=True)
        (nodes / "node-a" / "stale.json").write_text("old", encoding="utf-8")
        dest = M.extract_archive(self.bundle(), nodes, "node-a", replace=True)
        self.assertFalse((dest / "stale.json").exists())
        self.assertTrue((dest / "kernel.json").exists())

    def test_a_hostile_bundle_writes_nothing(self):
        nodes = tmpdir() / "fleet-nodes"
        bad = make_tar(tmpdir() / "bad.tgz", {"node-a/../escape.txt": "x"})
        with self.assertRaises(ValueError):
            M.extract_archive(bad, nodes, "node-a", replace=False)
        self.assertFalse((nodes.parent / "escape.txt").exists())
        self.assertFalse((nodes / "node-a").exists())


class ArchiveCandidates(unittest.TestCase):
    def test_newest_matching_bundle_comes_first_and_others_are_ignored(self):
        inbox = tmpdir()
        for name, mtime in (("node-a-20260101-000000.tgz", 1000),
                            ("node-a-20260301-000000.tgz", 3000),
                            ("node-a-20260201-000000.tar.gz", 2000),
                            ("other-node-20260401-000000.tgz", 4000),
                            ("node-a-notes.txt", 5000)):
            p = inbox / name
            p.write_text("x", encoding="utf-8")
            os.utime(p, (mtime, mtime))
        got = [p.name for p in M.archive_candidates(inbox, "node-a")]
        self.assertEqual(got, ["node-a-20260301-000000.tgz",
                               "node-a-20260201-000000.tar.gz",
                               "node-a-20260101-000000.tgz"])

    def test_an_exact_host_named_bundle_also_matches(self):
        inbox = tmpdir()
        (inbox / "node-a.tgz").write_text("x", encoding="utf-8")
        self.assertEqual([p.name for p in M.archive_candidates(inbox, "node-a")],
                         ["node-a.tgz"])

    def test_an_empty_inbox_yields_nothing(self):
        self.assertEqual(M.archive_candidates(tmpdir(), "node-a"), [])


class TurnAgentBundle(unittest.TestCase):
    def node(self, points=None, with_result=True, stderr=None):
        node_dir = tmpdir() / "node-a"
        run_dir = node_dir / "turn-agent-fast"
        run_dir.mkdir(parents=True)
        (run_dir / "turn-agent-sweep-manifest.json").write_text(
            json.dumps({"profile": "fast"}), encoding="utf-8")
        if with_result:
            body = {"points": points if points is not None else
                    [{"turns": 4, "concurrency": 2, "reuse_total_ms": 1234.0,
                      "reuse_agent_turns_per_sec": 3.5}]}
            (run_dir / "turn-agent-fak-q8.json").write_text(json.dumps(body), encoding="utf-8")
        if stderr is not None:
            (run_dir / "turn-agent-fak-q8.err.txt").write_text(stderr, encoding="utf-8")
        return node_dir

    def test_detects_a_turn_agent_bundle_by_its_manifest(self):
        self.assertTrue(M.looks_like_turn_agent(self.node()))

    def test_a_broad_node_bundle_is_not_mistaken_for_a_turn_agent_one(self):
        node_dir = tmpdir() / "node-a"
        (node_dir).mkdir(parents=True)
        (node_dir / "kernel.json").write_text("{}", encoding="utf-8")
        self.assertFalse(M.looks_like_turn_agent(node_dir))

    def test_a_complete_sweep_verifies_and_reports_every_point(self):
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            M.verify_turn_agent(self.node())
        text = buf.getvalue()
        self.assertIn("turn-agent profile: fast (1 points)", text)
        self.assertIn("T=4 A=2", text)

    def test_no_manifest_at_all_is_refused(self):
        with self.assertRaises(FileNotFoundError):
            M.verify_turn_agent(tmpdir())

    def test_a_missing_result_surfaces_the_captured_stderr(self):
        node_dir = self.node(with_result=False, stderr="CUDA out of memory")
        with self.assertRaises(FileNotFoundError) as cm:
            M.verify_turn_agent(node_dir)
        self.assertIn("CUDA out of memory", str(cm.exception))

    def test_a_result_with_no_points_is_refused(self):
        with self.assertRaises(ValueError) as cm:
            M.verify_turn_agent(self.node(points=[]))
        self.assertIn("no points", str(cm.exception))

    def test_a_point_missing_fields_names_every_one_of_them(self):
        node_dir = self.node(points=[{"turns": 4, "concurrency": 2}])
        with self.assertRaises(ValueError) as cm:
            M.verify_turn_agent(node_dir)
        msg = str(cm.exception)
        self.assertIn("reuse_total_ms", msg)
        self.assertIn("reuse_agent_turns_per_sec", msg)


class TaildropTimeouts(unittest.TestCase):
    def test_the_waiting_receive_is_bounded_and_looser_than_the_plain_one(self):
        # #3477: a hung tailscale daemon must not wedge the caller forever, and
        # --wait legitimately needs the longer of the two ceilings.
        self.assertGreater(M._TAILDROP_WAIT_TIMEOUT_S, M._TAILDROP_TIMEOUT_S)
        self.assertGreater(M._TAILDROP_TIMEOUT_S, 0)


if __name__ == "__main__":
    unittest.main()
