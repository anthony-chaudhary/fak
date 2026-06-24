#!/usr/bin/env python3
"""Tests for skill_context.py — the SkillContextRecord trust contract (#433).

Run:  python -m pytest tools/skill_context_test.py -q
  or:  python tools/skill_context_test.py     # falls back to a bare runner
"""
from __future__ import annotations

import json
from pathlib import Path

import skill_context as sc


# --------------------------------------------------------------------------- #
# Fixtures: a self-contained fake workspace with one skill + helper + policy.
# --------------------------------------------------------------------------- #
def make_workspace(tmp_path: Path, *, name: str = "demo-skill") -> Path:
    root = tmp_path / "ws"
    sd = root / ".claude" / "skills" / name
    sd.mkdir(parents=True)
    (root / "tools").mkdir()
    (root / "tools" / "demo_helper.py").write_text("print('helper')\n", encoding="utf-8")
    (root / "AGENTS.md").write_text("policy: work on main only\n", encoding="utf-8")
    (sd / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        "description: A demo skill. It runs tools/demo_helper.py and verifies.\n"
        "allowed-tools: Read, Bash, Grep\n"
        "---\n\n"
        "# demo\n\nRuns `tools/demo_helper.py`.\n",
        encoding="utf-8",
    )
    return root


# --------------------------------------------------------------------------- #
# Acceptance #1/#2: schema + digest helper
# --------------------------------------------------------------------------- #
def test_record_schema_and_digests(tmp_path):
    root = make_workspace(tmp_path)
    rec = sc.build_record(root, "demo-skill", policy_paths=[root / "AGENTS.md"])
    d = rec.to_dict()
    # MemoryViewRecord-compatible axes are present.
    for k in ("view_id", "view_type", "source_digests", "producer", "scope",
              "taint", "coverage", "faithfulness_probe", "labels"):
        assert k in d, f"missing MemoryViewRecord field {k}"
    # Skill-specific keying is present and non-empty.
    assert d["view_type"] == "skill_context"
    assert d["skill_digest"].startswith("sha256:")
    assert d["input_tree_digest"].startswith("sha256:")
    assert d["policy_digest"].startswith("sha256:")
    assert d["allowed_tools"] == ["Read", "Bash", "Grep"]
    assert d["witness_verdict"] == "pass"
    # The helper script was discovered from the SKILL.md body.
    assert any(e["path"] == "tools/demo_helper.py" for e in d["inputs"])
    # The policy file we pinned is in the policy set.
    assert any(e["path"] == "AGENTS.md" for e in d["policy_files"])
    assert 0.0 < d["coverage"] <= 1.0


def test_digests_are_deterministic(tmp_path):
    root = make_workspace(tmp_path)
    a = sc.build_record(root, "demo-skill", policy_paths=[root / "AGENTS.md"])
    b = sc.build_record(root, "demo-skill", policy_paths=[root / "AGENTS.md"])
    assert a.to_dict() == b.to_dict()


# --------------------------------------------------------------------------- #
# Acceptance #4: consumer check rejects on any drift
# --------------------------------------------------------------------------- #
def test_verify_accepts_fresh_record(tmp_path):
    root = make_workspace(tmp_path)
    rec = sc.build_record(root, "demo-skill", policy_paths=[root / "AGENTS.md"])
    ok, reasons = sc.verify_record(rec.to_dict(), root)
    assert ok, reasons


def test_verify_rejects_changed_skill_bytes(tmp_path):
    root = make_workspace(tmp_path)
    rec = sc.build_record(root, "demo-skill", policy_paths=[root / "AGENTS.md"])
    md = root / ".claude" / "skills" / "demo-skill" / "SKILL.md"
    md.write_text(md.read_text(encoding="utf-8") + "\nedited\n", encoding="utf-8")
    ok, reasons = sc.verify_record(rec.to_dict(), root)
    assert not ok
    assert any("skill_digest" in r for r in reasons)


def test_verify_rejects_changed_input(tmp_path):
    root = make_workspace(tmp_path)
    rec = sc.build_record(root, "demo-skill", policy_paths=[root / "AGENTS.md"])
    (root / "tools" / "demo_helper.py").write_text("print('tampered')\n", encoding="utf-8")
    ok, reasons = sc.verify_record(rec.to_dict(), root)
    assert not ok
    assert any("input" in r for r in reasons)
    assert any("input_tree_digest" in r for r in reasons)


def test_verify_rejects_changed_policy(tmp_path):
    root = make_workspace(tmp_path)
    rec = sc.build_record(root, "demo-skill", policy_paths=[root / "AGENTS.md"])
    (root / "AGENTS.md").write_text("policy: anything goes\n", encoding="utf-8")
    ok, reasons = sc.verify_record(rec.to_dict(), root)
    assert not ok
    assert any("policy_digest" in r for r in reasons)


def test_verify_rejects_flipped_witness(tmp_path):
    root = make_workspace(tmp_path)
    rec = sc.build_record(root, "demo-skill", policy_paths=[root / "AGENTS.md"])
    # Gut the frontmatter name -> witness flips to fail, even though we then
    # re-pin the skill digest into the record to isolate the witness axis.
    md = root / ".claude" / "skills" / "demo-skill" / "SKILL.md"
    md.write_text(
        "---\ndescription: no name now\n---\n# demo\n", encoding="utf-8")
    d = rec.to_dict()
    d["skill_digest"] = sc.sha256_file(md)  # neutralize the bytes axis
    ok, reasons = sc.verify_record(d, root)
    assert not ok
    assert any("witness" in r for r in reasons)


# --------------------------------------------------------------------------- #
# Acceptance #5: cross-agent handoff — the consumer trusts the WITNESS, not prose
# --------------------------------------------------------------------------- #
def consumer_agent(record_json: str, prose_handoff: str, root: Path):
    """A second agent. It is handed the producer's record JSON AND a prose
    summary. The contract: it may only page in the record's projection AFTER the
    witness verifies; otherwise it refuses and never touches the prose."""
    record = json.loads(record_json)
    ok, reasons = sc.verify_record(record, root)
    if not ok:
        return {"consumed": None, "trusted_prose": False, "refused": reasons}
    # Trust path: consume the verified projection. The prose is never read.
    return {"consumed": record["projection"], "trusted_prose": False, "refused": []}


def test_cross_agent_handoff_consumes_only_after_witness(tmp_path):
    root = make_workspace(tmp_path)
    # Producer agent A emits a record + an UNTRUSTED prose handoff.
    rec = sc.build_record(root, "demo-skill", policy_paths=[root / "AGENTS.md"])
    record_json = json.dumps(rec.to_dict())
    prose = "trust me, the skill is fine and does X, Y, Z"  # never to be believed

    # Consumer agent B, fresh tree: witness passes -> consumes the projection.
    out = consumer_agent(record_json, prose, root)
    assert out["consumed"] == rec.projection
    assert out["trusted_prose"] is False
    assert out["consumed"] != prose  # it did NOT swallow the producer's prose

    # Now the skill drifts under B's feet. B must refuse from the record alone,
    # falling back to nothing — never to the producer's stale prose.
    md = root / ".claude" / "skills" / "demo-skill" / "SKILL.md"
    md.write_text(md.read_text(encoding="utf-8") + "\nmalicious append\n", encoding="utf-8")
    out2 = consumer_agent(record_json, prose, root)
    assert out2["consumed"] is None
    assert out2["trusted_prose"] is False
    assert out2["refused"]  # carries the structured rejection reason


# --------------------------------------------------------------------------- #
# Acceptance #3 smoke: the CLI emits + checks against the real repo skill pack.
# --------------------------------------------------------------------------- #
def test_cli_emit_and_check_roundtrip_real_repo(tmp_path):
    root = sc.repo_root()
    skills = sc.list_skills(root)
    if not skills:
        return  # no skill pack in this checkout — nothing to assert
    skill = "quality-score" if "quality-score" in skills else skills[0]
    out = tmp_path / "rec.json"
    rc = sc.main(["--emit", skill, "--out", str(out)])
    assert rc == 0
    assert out.is_file()
    rc2 = sc.main(["--check", str(out)])
    assert rc2 == 0


# --------------------------------------------------------------------------- #
# Bare runner (no pytest needed).
# --------------------------------------------------------------------------- #
def _bare_main() -> int:
    import tempfile
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        with tempfile.TemporaryDirectory() as d:
            try:
                t(Path(d))
                print(f"ok   {t.__name__}")
            except Exception:  # noqa: BLE001
                failed += 1
                print(f"FAIL {t.__name__}")
                traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_bare_main())
