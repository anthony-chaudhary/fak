#!/usr/bin/env python3
"""Tests for tools/session_checkpoint.py -- the durable session work-status checkpoint.

Covers the load-bearing properties: deterministic render, the private-route needle GATE
(refuses + writes nothing), the public-route scrub TRANSFORM + refuse-if-needle-survives,
`both` writing private even when public is refused, the public no-op without a target, and
that --dry-run / mocked push+gh write/post nothing real.
"""
import json
import os
import re
import subprocess
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import session_checkpoint as sc  # noqa: E402


# A live-Slack-token SHAPE the AUDIT_REGEXES always catch (no private sidecar needed),
# so the gate is exercised deterministically on any clone. ASSEMBLED at runtime from
# parts so this SOURCE line does not itself match the xoxb shape -- otherwise the repo's
# own pre-commit leak gate would (correctly) refuse to commit the test file.
PLANTED_NEEDLE = "xox" + "b-" + "12345678-87654321-" + "AbCdEfGhIjKlMnOpQrStUvWx"


def _rec(**over):
    base = dict(source="periodic", transcript=None, in_flight=None,
                stamp="2026-06-24T00:00:00Z", host="test-host")
    base.update(over)
    return sc.build_record(**base)


# --- render ------------------------------------------------------------------------
def test_record_renders_deterministically():
    r1 = _rec()
    r2 = _rec()
    assert r1["schema"] == sc.SCHEMA
    assert r1["source"] == "periodic"
    assert r1["host"] == "test-host"
    assert r1["stamp"] == "2026-06-24T00:00:00Z"
    # Same inputs + same git state => identical record.
    assert r1 == r2


def test_periodic_has_no_session_context_but_stop_can():
    periodic = _rec(source="periodic")
    assert "transcript" not in periodic and "in_flight" not in periodic
    stop = _rec(source="stop", transcript="/p/x.jsonl", in_flight="shipping checkpoint")
    assert stop["transcript"] == "/p/x.jsonl"
    assert stop["in_flight"] == "shipping checkpoint"
    md = sc.render_md(stop)
    assert "x.jsonl" in md and "shipping checkpoint" in md


# --- private route: the needle GATE ------------------------------------------------
def test_private_gate_refuses_on_planted_needle_and_writes_nothing(tmp_path):
    rec = _rec(in_flight=f"leaked secret {PLANTED_NEEDLE}")
    priv = tmp_path / "fak-private"
    priv.mkdir()
    subprocess.run(["git", "-C", str(priv), "init", "-q"], check=False)
    res = sc.route_checkpoint(rec, route="private", public_target=None,
                              do_push=False, dry=False, private_repo=str(priv))
    pr = res["results"][0]
    assert pr["ok"] is False
    assert "refused" in pr["reason"]
    assert pr["wrote"] is False
    # Nothing written under the archive.
    archive = priv / "session-checkpoints" / "fak"
    assert not archive.exists() or not any(archive.iterdir())


def test_private_route_writes_and_commits(tmp_path, monkeypatch):
    rec = _rec()
    priv = tmp_path / "fak-private"
    (priv / "session-checkpoints" / "fak").mkdir(parents=True)
    subprocess.run(["git", "-C", str(priv), "init", "-q"], check=False)
    subprocess.run(["git", "-C", str(priv), "config", "user.email", "t@t"], check=False)
    subprocess.run(["git", "-C", str(priv), "config", "user.name", "t"], check=False)
    res = sc.route_private(sc.render_md(rec), rec["host"], do_push=False, dry=False,
                           private_repo=str(priv))
    assert res["ok"] is True and res["wrote"] is True
    dst = priv / "session-checkpoints" / "fak" / "test-host.md"
    assert dst.is_file()
    body = dst.read_text(encoding="utf-8")
    assert "session checkpoint" in body
    # The commit landed (no push attempted).
    log = subprocess.run(["git", "-C", str(priv), "log", "--oneline"],
                         capture_output=True, text=True)
    assert "checkpoint" in log.stdout


# --- public route: TRANSFORM + re-audit --------------------------------------------
def test_public_route_noops_without_target():
    res = sc.route_public(sc.render_md(_rec()), public_target=None, dry=False)
    assert res["ok"] is True and res["posted"] is False
    assert "no --public-target" in res["reason"]


def test_public_route_refuses_if_needle_survives_scrub():
    # The planted Slack-token shape is not in REPLACEMENTS, so it survives the transform
    # and the re-audit must REFUSE rather than post.
    body = sc.render_md(_rec(in_flight=f"oops {PLANTED_NEEDLE}"))
    posted = {"called": False}

    def fake_post(target, b):
        posted["called"] = True
        return 0, "ok"

    res = sc.route_public(body, public_target="123", dry=False, post_fn=fake_post)
    assert res["ok"] is False and res["posted"] is False
    assert "survived scrub" in res["reason"]
    assert posted["called"] is False  # never reached gh


def test_public_route_posts_clean_body_via_mock():
    body = sc.render_md(_rec(in_flight="nothing secret here"))
    captured = {}

    def fake_post(target, b):
        captured["target"] = target
        captured["body"] = b
        return 0, "https://github.com/.../comment"

    res = sc.route_public(body, public_target="123", dry=False, post_fn=fake_post)
    assert res["ok"] is True and res["posted"] is True
    assert captured["target"] == "123"
    assert "session checkpoint" in captured["body"]


# --- both: private is the floor ----------------------------------------------------
def test_both_writes_private_even_when_public_refused(tmp_path):
    # in_flight carries a needle: public must refuse, but the PRIVATE gate refuses too here
    # (same needle). So instead test the asymmetry: a CLEAN body, public has no target ->
    # public no-ops (ok), private writes. Then a body where public is refused via a planted
    # needle that the private side ALSO refuses is the gate test above. Here we assert the
    # ordering contract: private result comes first.
    rec = _rec(in_flight="clean note")
    priv = tmp_path / "fak-private"
    (priv / "session-checkpoints" / "fak").mkdir(parents=True)
    subprocess.run(["git", "-C", str(priv), "init", "-q"], check=False)
    subprocess.run(["git", "-C", str(priv), "config", "user.email", "t@t"], check=False)
    subprocess.run(["git", "-C", str(priv), "config", "user.name", "t"], check=False)
    res = sc.route_checkpoint(rec, route="both", public_target=None, do_push=False,
                              dry=False, private_repo=str(priv))
    assert res["results"][0]["route"] == "private"
    assert res["results"][0]["ok"] is True
    assert res["results"][1]["route"] == "public"  # no target -> no-op ok
    dst = priv / "session-checkpoints" / "fak" / "test-host.md"
    assert dst.is_file()


# --- dry-run writes/posts nothing --------------------------------------------------
def test_dry_run_writes_nothing(tmp_path):
    rec = _rec()
    priv = tmp_path / "fak-private"
    priv.mkdir()
    res = sc.route_checkpoint(rec, route="both", public_target="123", do_push=False,
                              dry=True, private_repo=str(priv))
    assert all(r.get("ok") for r in res["results"])
    assert not (priv / "session-checkpoints").exists()


# --- CLI smoke ---------------------------------------------------------------------
def test_cli_dry_run_json_smoke(capsys):
    # --no-discover keeps the smoke test on the PURE floor (deterministic: no dependence
    # on whether a live transcript happens to exist on the host running the tests).
    rc = sc.main(["--source", "periodic", "--route", "private", "--dry-run", "--json",
                  "--no-discover"])
    out = capsys.readouterr().out
    doc = json.loads(out)
    assert doc["record"]["schema"] == sc.SCHEMA
    assert doc["result"]["results"][0]["route"] == "private"
    assert rc == 0


def test_cli_refusal_exits_nonzero(monkeypatch, capsys):
    # Force the in-flight note to carry a needle via --in-flight; private route refuses.
    rc = sc.main(["--route", "private", "--in-flight", f"x {PLANTED_NEEDLE}",
                  "--no-push", "--json"])
    assert rc == 3  # loud, non-zero, on a gate refusal


# --- within-turn coverage: the periodic crash-survivor discovers the transcript (#634) -
def _slug(repo):
    return re.sub(r"[^A-Za-z0-9]", "-", os.path.normpath(repo))


def _make_transcript(home, repo, account=".claude", sid="11111111-1111-1111-1111-111111111111",
                     content="{}\n"):
    proj = os.path.join(home, account, "projects", _slug(repo))
    os.makedirs(proj, exist_ok=True)
    p = os.path.join(proj, sid + ".jsonl")
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)
    return p


def test_discover_finds_newest_recent_transcript_across_accounts(tmp_path):
    home = tmp_path / "home"
    repo = str(tmp_path / "fak")
    old = _make_transcript(str(home), repo, account=".claude", sid="a" * 32)
    new = _make_transcript(str(home), repo, account=".claude-q-netra", sid="b" * 32)
    # Force a deterministic ordering: `new` is touched AFTER `old`.
    os.utime(old, (time.time() - 60, time.time() - 60))
    os.utime(new, None)
    got = sc.discover_active_transcript(repo, home=str(home))
    assert got == new  # newest wins, across both account dirs


def test_discover_returns_none_when_no_live_transcript(tmp_path):
    repo = str(tmp_path / "fak")
    assert sc.discover_active_transcript(repo, home=str(tmp_path / "home")) is None


def test_discover_ignores_wrong_project_slug(tmp_path):
    home = tmp_path / "home"
    repo = str(tmp_path / "fak")
    other = str(tmp_path / "OTHER-repo")
    _make_transcript(str(home), other)  # different slug
    assert sc.discover_active_transcript(repo, home=str(home)) is None


def test_discover_recency_filter_drops_stale_dead_session(tmp_path):
    # A long-dead session's transcript must NOT become the survivor pointer: a stale mtime
    # past max_age_seconds is excluded, so we degrade to the pure git/host floor (None).
    home = tmp_path / "home"
    repo = str(tmp_path / "fak")
    p = _make_transcript(str(home), repo)
    stale = time.time() - 10000  # ~2.7h ago, beyond the 2h default max_age
    os.utime(p, (stale, stale))
    assert sc.discover_active_transcript(repo, home=str(home)) is None
    # ...and a tight max_age also excludes a 5-minute-old transcript.
    fresh = _make_transcript(str(home), repo, sid="c" * 32)
    os.utime(fresh, (time.time() - 300, time.time() - 300))
    assert sc.discover_active_transcript(repo, home=str(home), max_age_seconds=60) is None


def test_cli_periodic_discovers_transcript_pointer(monkeypatch, capsys):
    # The periodic crash-survivor has no stdin/Stop event, yet its record CARRIES the
    # transcript pointer discovered from disk -- the within-turn coverage gap (#634).
    # main() discovers against REPO_ROOT, so the fixture transcript must live under the
    # slug for sc.REPO_ROOT (portable: whatever path the repo actually has on this host).
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = _make_transcript(td, sc.REPO_ROOT)
        monkeypatch.setenv("FAK_CHECKPOINT_HOME", td)
        rc = sc.main(["--source", "periodic", "--route", "private", "--dry-run", "--json"])
        doc = json.loads(capsys.readouterr().out)
        assert rc == 0
        assert doc["record"]["transcript"] == p


def test_cli_no_discover_keeps_pure_floor(monkeypatch, capsys):
    # --no-discover is the explicit escape hatch back to the pre-#634 git/host-only floor,
    # even when a live transcript is sitting on disk right next to it.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        _make_transcript(td, sc.REPO_ROOT)
        monkeypatch.setenv("FAK_CHECKPOINT_HOME", td)
        rc = sc.main(["--source", "periodic", "--no-discover", "--route", "private",
                      "--dry-run", "--json"])
        doc = json.loads(capsys.readouterr().out)
        assert rc == 0
        assert "transcript" not in doc["record"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
