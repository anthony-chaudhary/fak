#!/usr/bin/env python3
r"""session_checkpoint.py -- a durable, off-host record of WHERE a fak session's work
stands, so a local crash that can't resume nicely still leaves a recoverable trace on
GitHub.

WHY THIS EXISTS
---------------
The only durable GitHub push on a clean session Stop today is ``memory_sync.py`` -- it
mirrors agent *memory*, not the session's *work status*. And this box crashes for a known
hardware reason (the AMD RX 7600 driver faults Windows Terminal via GPU TDR; the agent
process survives and the transcript stays on disk, but nothing OFF the host records what
was in flight). This tool writes that missing record.

THE LOAD-BEARING LIMITATION (read before trusting this)
-------------------------------------------------------
A **Stop hook never fires on a crash.** Stop is a CLEAN end-of-turn event; a TDR terminal
kill / OOM / power loss never reaches it. So the Stop-hook writer (``--hook``) covers every
GRACEFUL boundary and gives ZERO coverage of a real crash. The crash-survivor is the
PERIODIC writer (``--source periodic``, driven by a scheduled task), which writes BEFORE
the crash. The two writers produce different-fidelity records on purpose:

  * periodic  -- runs in session 0 with NO stdin, so it has no Stop event to read. It
                 DISCOVERS the active transcript POINTER from disk (the newest
                 recently-touched ``.jsonl`` for this repo under
                 ``~/.claude*/projects/<slug>/``) and carries it in the record alongside
                 the git/host truth. Because the periodic tick is the ONLY writer that
                 survives a hard crash, giving it the pointer CLOSES THE WITHIN-TURN
                 COVERAGE GAP: a mid-LONG-TURN crash leaves a survivor record that points
                 at what was in flight, even though the Stop hook (which has the pointer
                 from its event JSON) only fires at turn-end (#634). The pointer is a PATH
                 only -- no transcript content is read, so there is no new leak surface
                 beyond the route gates that already handle a host path. Degrades to
                 git/host truth only when no live session exists (``--no-discover``).
                 This is the FLOOR that survives a crash.
  * stop      -- receives the Stop event JSON on stdin: adds the transcript pointer and a
                 one-line in-flight note. This is the CEILING, only at a clean boundary.

THE ROUTER (the core abstraction)
---------------------------------
A checkpoint is rendered ONCE, then a router decides where it lands:

  * private  (DEFAULT) -- raw record committed+pushed to the PRIVATE fak-private repo
              (sibling of memory_sync's agent-memory archive). GATED: a shape-regex /
              needle scan refuses to write if a public-leak secret appears (defense in
              depth -- keeps a stray pasted secret out of even the private history).
  * public   -- the record is run through the FULL scrub TRANSFORM (host paths / account
              names / needles rewritten), RE-AUDITED, and only on a clean audit posted to
              one pinned ``gh`` issue/gist IN PLACE (editable surface, no public-repo commit
              churn). A post-scrub needle => REFUSE, do not post. There is NO raw-to-public
              code path. Public is opt-in; the default crash-survivor tick stays private.
  * both     -- private first (the floor), then public; a public failure never sinks private.

FAIL-SOFT, NEVER BLOCK: a checkpoint failure must never block a session Stop or crash the
scheduled task. Push/post failures degrade to a warning; the local private commit is still
durable on the next sync.

This is a PURE READ-ONLY FOLD over git + the scrub primitives; it launches no worker and
writes only the one checkpoint file (private route) or one gh edit (public route).

CLI
---
    python tools/session_checkpoint.py --source periodic         # scheduled tick (route=private; discovers transcript ptr)
    python tools/session_checkpoint.py --hook                    # Stop-hook (reads event JSON on stdin)
    python tools/session_checkpoint.py --route both --public-target 123
    python tools/session_checkpoint.py --source periodic --dry-run --json
    python tools/session_checkpoint.py --source periodic --no-push
    python tools/session_checkpoint.py --source periodic --no-discover   # pure git/host floor (pre-#634)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import re
import subprocess
from dispatch_worker import install_no_window_subprocess_defaults
install_no_window_subprocess_defaults(subprocess)
import sys
import tempfile
import time
from typing import Any

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except (AttributeError, ValueError):
    pass

SCHEMA = "fak-session-checkpoint/1"

# --- repo geometry -----------------------------------------------------------------
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TOOLS_DIR)
# The private repo + the archive subtree the record lands in (sibling of memory_sync's
# ../fak-private/agent-memory/fak). One stable file PER HOST: newest overwrites, git
# history is the time series, no unbounded growth.
PRIVATE_REPO = os.path.normpath(os.path.join(REPO_ROOT, "..", "fak-private"))
PRIVATE_ARCHIVE_REL = os.path.join("session-checkpoints", "fak")


def _py() -> str:
    return sys.executable or "python"


# --- scrub primitives (imported, never reimplemented) ------------------------------
# We reuse scrub_public_copy's SOUND defenses: the shape regexes (always correct, no
# sidecar needed -> works headless on the scheduled tick) and the pulled-private real
# needles when present. This is exactly what audit_tree() does per file, minus the
# git-ls-files iteration (our record lives OUTSIDE the tracked public tree).
sys.path.insert(0, TOOLS_DIR)
try:
    import scrub_public_copy as _scrub  # type: ignore
except Exception:  # pragma: no cover - scrub tool always present in this repo
    _scrub = None  # type: ignore


def needle_hits(text: str, root: str = REPO_ROOT, *, secrets_only: bool = False
                ) -> list[dict[str, str]]:
    """Scan ``text`` for leak markers the way scrub_public_copy.audit_tree scans a file.

    Two kinds, treated differently by the two routes:
      * ``shape``  -- a true SECRET shape (live Slack/SA token). Must NEVER touch ANY repo,
                      public OR private. Always sound (no sidecar needed -> works headless).
      * ``needle`` -- a REDACTION needle (host/account/infra name). Allowed in the PRIVATE
                      repo (memory_sync already mirrors host-tagged data there); must be
                      scrubbed out only for PUBLIC. Present only once the private sidecar is
                      pulled.

    ``secrets_only=True`` (the PRIVATE gate) returns ONLY shape hits -- a private checkpoint
    legitimately carries the operator's own host/account names. The PUBLIC gate uses the full
    set (shapes + redaction needles)."""
    if _scrub is None:
        return []
    hits: list[dict[str, str]] = []
    for rx, label in _scrub.AUDIT_REGEXES:
        if rx.search(text):
            hits.append({"needle": label, "kind": "shape"})
    if secrets_only:
        return hits
    priv = _scrub.load_private_needles(root)
    real = sorted({n for n in (priv.get("export_audit_needles") or []) if n}) if priv else []
    low = text.lower()
    for needle in real:
        if needle.lower() in low:
            hits.append({"needle": needle, "kind": "needle"})
    return hits


def scrub_transform(text: str, root: str = REPO_ROOT) -> str:
    """Apply scrub_public_copy's REPLACEMENTS to ``text`` for the PUBLIC route.

    On THIS public clone the committed REPLACEMENTS are de-fanged (both sides are
    placeholders), so this is a literal-form pass; the real rewrites live in the private
    repo's copy of the tool. Either way the public route ALSO hard-gates on needle_hits()
    AFTER this transform, so a value the de-fanged pass can't rewrite is REFUSED, never
    posted. Best-effort: if the scrub module is unavailable the gate alone protects us."""
    if _scrub is None:
        return text
    out = text
    for triple in getattr(_scrub, "REPLACEMENTS", []):
        old, new = triple[0], triple[1]
        if old and old != new:
            out = out.replace(old, new)
    for triple in getattr(_scrub, "CASE_INSENSITIVE_REPLACEMENTS", []):
        old, new = triple[0], triple[1]
        if old and old.lower() != new.lower():
            out = re.sub(re.escape(old), new, out, flags=re.IGNORECASE)
    return out


# --- git fold ----------------------------------------------------------------------
def _git(repo: str, *args: str) -> tuple[int, str]:
    try:
        r = subprocess.run(["git", "-C", repo, *args], capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
    except OSError as exc:
        return 1, str(exc)
    return r.returncode, (r.stdout + r.stderr).strip()


def git_status(repo: str = REPO_ROOT) -> dict[str, Any]:
    """Branch, HEAD sha, dirty path NAMES (never content), count. Pure read-only."""
    _, branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    _, head = _git(repo, "rev-parse", "--short", "HEAD")
    _, subject = _git(repo, "log", "-1", "--pretty=%s")
    rc, porc = _git(repo, "status", "--porcelain")
    dirty: list[str] = []
    if rc == 0:
        for line in porc.splitlines():
            line = line.rstrip()
            if not line:
                continue
            # "XY <path>"; keep the path NAME only (no content, no diff).
            dirty.append(line[3:] if len(line) > 3 else line)
    return {
        "branch": branch or "?",
        "head": head or "?",
        "head_subject": subject or "",
        "dirty_paths": dirty,
        "dirty_count": len(dirty),
    }


def discover_active_transcript(repo: str, *, home: str | None = None,
                               max_age_seconds: int = 7200) -> str | None:
    """Find the NEWEST recently-active Claude Code transcript POINTER for ``repo``, so the
    periodic crash-survivor can carry the same pointer the Stop hook gets from its event
    JSON -- closing the within-turn coverage gap (#634).

    Claude Code writes one append-only ``.jsonl`` per session under
    ``<home>/.claude*/projects/<slug(cwd)>/<sid>.jsonl``, where ``slug(cwd)`` collapses every
    non-alphanumeric in the cwd to ``-`` (so ``C:\\work\\fak`` -> ``C--work-fak``). We glob
    EVERY account dir (``.claude`` + ``.claude-*``), keep only transcripts touched within
    ``max_age_seconds`` (a long-dead session's transcript must NOT become the survivor
    pointer), and return the newest by mtime.

    Returns the PATH only (no transcript content is ever read), so there is no new leak
    surface -- a host path is exactly what the PRIVATE route already permits and the PUBLIC
    route's scrub transform already rewrites. Degrades to ``None`` when no live transcript
    exists, so the caller falls back to git/host truth only and the FLOOR is preserved.

    ``home`` defaults to ``$FAK_CHECKPOINT_HOME`` then ``~``; the env override is the
    testability seam (and helps an operator whose claude config lives off the default path)."""
    h = home or os.environ.get("FAK_CHECKPOINT_HOME") or os.path.expanduser("~")
    slug = re.sub(r"[^A-Za-z0-9]", "-", os.path.normpath(repo))
    now = time.time()
    candidates: list[str] = []
    for p in glob.glob(os.path.join(h, ".claude*", "projects", slug, "*.jsonl")):
        try:
            if now - os.path.getmtime(p) <= max_age_seconds:
                candidates.append(p)
        except OSError:
            continue
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _git_commit_push(archive: str, do_push: bool, *, message: str) -> int:
    """Stage + commit ``archive`` in its repo; optionally push. Fail-soft (returns 1 on a
    real git error, 0 otherwise). Mirrors memory_sync._git_commit_push exactly -- the same
    proven idiom -- but with a checkpoint commit message. Kept self-contained so this
    PUBLIC-tree tool never cross-imports the PRIVATE repo's module."""
    repo = archive
    while repo and not os.path.isdir(os.path.join(repo, ".git")):
        parent = os.path.dirname(repo)
        if parent == repo:
            break
        repo = parent
    if not os.path.isdir(os.path.join(repo, ".git")):
        print(f"  commit: no git repo found above {archive}; skipped", file=sys.stderr)
        return 1
    rel = os.path.relpath(archive, repo)
    rc, out = _git(repo, "add", "--", rel)
    if rc != 0:
        print(f"  git add failed: {out}", file=sys.stderr)
        return 1
    rc, _ = _git(repo, "diff", "--cached", "--quiet", "--", rel)
    if rc == 0:
        print("  commit: nothing staged (checkpoint unchanged)")
        return 0
    rc, out = _git(repo, "commit", "-s", "-m", message, "--", rel)
    if rc != 0:
        print(f"  git commit failed: {out}", file=sys.stderr)
        return 1
    print(f"  committed: {out.splitlines()[0] if out else 'ok'}")
    if do_push:
        rc, out = _git(repo, "push", "origin", "HEAD")
        if rc != 0:
            print(f"  git push failed (commit is local; will push next run): {out}",
                  file=sys.stderr)
            return 1
        print("  pushed to origin")
    return 0


# --- record ------------------------------------------------------------------------
def build_record(*, source: str, transcript: str | None, in_flight: str | None,
                 stamp: str, host: str, repo: str = REPO_ROOT) -> dict[str, Any]:
    """Render ONE checkpoint record. Deterministic given (source, transcript, in_flight,
    stamp, host) + the repo's git state. ``stamp`` is injected (never read from the clock
    here) so callers control it and tests stay deterministic."""
    gs = git_status(repo)
    rec: dict[str, Any] = {
        "schema": SCHEMA,
        "source": source,            # stop | periodic -- tells a reader the fidelity
        "stamp": stamp,
        "host": host,
        "repo_root": repo,
        **gs,
    }
    # Session-context fields exist ONLY for the stop writer; the periodic tick can't have them.
    if transcript:
        rec["transcript"] = transcript
    if in_flight:
        rec["in_flight"] = in_flight
    return rec


def render_md(rec: dict[str, Any]) -> str:
    """Human-readable checkpoint doc (one stable file per host; overwritten each tick)."""
    lines = [
        f"# session checkpoint — {rec.get('host', '?')}",
        "",
        f"- schema: `{rec.get('schema')}`",
        f"- source: **{rec.get('source')}**  (periodic = crash-survivor floor; stop = clean-end ceiling)",
        f"- stamp: {rec.get('stamp')}",
        f"- branch: `{rec.get('branch')}`  @ `{rec.get('head')}`",
        f"- last commit: {rec.get('head_subject', '')}",
        f"- dirty files: {rec.get('dirty_count', 0)}",
    ]
    if rec.get("transcript"):
        lines.append(f"- transcript: `{rec['transcript']}`")
    if rec.get("in_flight"):
        lines.append(f"- in-flight: {rec['in_flight']}")
    dirty = rec.get("dirty_paths") or []
    if dirty:
        lines += ["", "## uncommitted (names only)", ""]
        lines += [f"- `{p}`" for p in dirty[:200]]
        if len(dirty) > 200:
            lines.append(f"- … and {len(dirty) - 200} more")
    lines += ["", "<!-- pointer + status, not a content backup. Resume via the transcript above. -->", ""]
    return "\n".join(lines)


# --- the router --------------------------------------------------------------------
def route_private(body: str, host: str, *, do_push: bool, dry: bool,
                  private_repo: str = PRIVATE_REPO) -> dict[str, Any]:
    """private route: GATE on hard SECRETS only (the private repo may hold the operator's
    own host/account names), then write+commit+push the raw record."""
    hits = needle_hits(body, REPO_ROOT, secrets_only=True)
    if hits:
        return {"route": "private", "ok": False, "reason": "needle-gate refused",
                "hits": hits, "wrote": False}
    archive = os.path.join(private_repo, PRIVATE_ARCHIVE_REL)
    dst = os.path.join(archive, f"{host}.md")
    if dry:
        return {"route": "private", "ok": True, "wrote": False, "dry_run": True, "path": dst}
    if not os.path.isdir(private_repo):
        return {"route": "private", "ok": False, "reason": f"private repo not found: {private_repo}",
                "wrote": False}
    os.makedirs(archive, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        f.write(body)
    rc = _git_commit_push(archive, do_push,
                          message="chore(checkpoint): session work-status checkpoint (fak)")
    return {"route": "private", "ok": rc == 0, "wrote": True, "path": dst,
            "pushed": do_push and rc == 0}


def _gh_post(target: str, body: str) -> tuple[int, str]:
    """Update one pinned gh issue/gist IN PLACE with the scrubbed body. A pure-int
    ``target`` is an issue number (comment); anything else is treated as a gist id."""
    if target.isdigit():
        cmd = ["gh", "issue", "comment", target, "--body", body]
    else:
        # gist edit needs a file; write the body to a temp file and pass it.
        tf = tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8")
        try:
            tf.write(body)
            tf.close()
            cmd = ["gh", "gist", "edit", target, tf.name]
            r = subprocess.run(cmd, capture_output=True, text=True)
            return r.returncode, (r.stdout + r.stderr).strip()
        finally:
            try:
                os.unlink(tf.name)
            except OSError:
                pass
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


def route_public(body: str, *, public_target: str | None, dry: bool,
                 post_fn=_gh_post) -> dict[str, Any]:
    """public route: TRANSFORM (scrub) -> RE-AUDIT -> refuse if a needle survives ->
    else post to one gh issue/gist in place. NO raw-to-public path."""
    if not public_target:
        return {"route": "public", "ok": True, "posted": False,
                "reason": "no --public-target set; public route is a no-op"}
    scrubbed = scrub_transform(body, REPO_ROOT)
    hits = needle_hits(scrubbed, REPO_ROOT)
    if hits:
        return {"route": "public", "ok": False, "posted": False,
                "reason": "needle survived scrub; refused to post", "hits": hits}
    if dry:
        return {"route": "public", "ok": True, "posted": False, "dry_run": True,
                "target": public_target, "scrubbed_preview": scrubbed[:400]}
    rc, out = post_fn(public_target, scrubbed)
    if rc != 0:
        return {"route": "public", "ok": False, "posted": False,
                "reason": f"gh post failed: {out}", "target": public_target}
    return {"route": "public", "ok": True, "posted": True, "target": public_target}


def route_checkpoint(rec: dict[str, Any], *, route: str, public_target: str | None,
                     do_push: bool, dry: bool, post_fn=_gh_post,
                     private_repo: str = PRIVATE_REPO) -> dict[str, Any]:
    """Dispatch the rendered record to private / public / both. Private is the FLOOR and is
    always attempted first in ``both`` mode, so a public failure never sinks the private
    write. Default route is private (the periodic tick never auto-promotes)."""
    body = render_md(rec)
    out: dict[str, Any] = {"route": route, "results": []}
    if route in ("private", "both"):
        out["results"].append(route_private(body, rec["host"], do_push=do_push,
                                             dry=dry, private_repo=private_repo))
    if route in ("public", "both"):
        out["results"].append(route_public(body, public_target=public_target,
                                            dry=dry, post_fn=post_fn))
    out["ok"] = all(r.get("ok") for r in out["results"]) if out["results"] else False
    return out


# --- CLI -----------------------------------------------------------------------------
def _now_iso() -> str:
    # Wall clock is read ONLY here at the CLI edge (not in build_record), so the fold stays
    # deterministic and testable. importlib keeps the lint-flagged direct Date use local.
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_stop_event() -> dict[str, Any]:
    """Stop-hook mode: the harness writes the Stop event JSON to stdin. We pull
    transcript_path from it; everything else degrades cleanly if absent."""
    try:
        raw = sys.stdin.read()
    except Exception:
        return {}
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except ValueError:
        return {}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Durable session work-status checkpoint to GitHub.")
    ap.add_argument("--source", choices=["stop", "periodic"], default="periodic",
                    help="stop = clean-end (session context from the event); "
                         "periodic = crash-survivor (git/host truth + discovered transcript pointer)")
    ap.add_argument("--hook", action="store_true",
                    help="Stop-hook mode: read the Stop event JSON on stdin (implies --source stop)")
    ap.add_argument("--route", choices=["private", "public", "both"], default="private")
    ap.add_argument("--public-target", default=os.environ.get("FAK_CHECKPOINT_PUBLIC_TARGET"),
                    help="gh issue number or gist id for the public route")
    ap.add_argument("--transcript", default=None,
                    help="session transcript path (explicit override; else stop reads the event, "
                         "periodic DISCOVERS it from disk)")
    ap.add_argument("--in-flight", default=None, help="one-line in-flight note (stop mode)")
    ap.add_argument("--no-discover", action="store_true",
                    help="periodic source: do NOT auto-discover the active transcript pointer "
                         "(stay pure git/host truth -- the pre-#634 floor)")
    ap.add_argument("--no-push", action="store_true", help="commit locally but do not push")
    ap.add_argument("--dry-run", action="store_true", help="render + route decisions, write/post nothing")
    ap.add_argument("--json", action="store_true", help="machine-readable result")
    args = ap.parse_args(argv)

    source = "stop" if args.hook else args.source
    transcript = args.transcript
    in_flight = args.in_flight
    if args.hook:
        ev = _read_stop_event()
        transcript = transcript or ev.get("transcript_path") or ev.get("transcript")
        # The Stop event may carry the active goal/condition; surface it as the in-flight note.
        in_flight = in_flight or ev.get("goal") or ev.get("stop_hook_condition")
    elif source == "periodic" and not args.no_discover and not transcript:
        # Close the within-turn coverage gap (#634): the periodic crash-survivor has no
        # stdin/Stop event, so DISCOVER the active transcript pointer from disk. Now a
        # mid-LONG-TURN crash leaves a survivor record that points at what was in flight,
        # even though the Stop hook (which carries the pointer) only fires at turn-end.
        # Degrades to None (pure git/host floor) when no live session exists.
        discovered = discover_active_transcript(REPO_ROOT)
        if discovered:
            transcript = discovered

    rec = build_record(source=source, transcript=transcript, in_flight=in_flight,
                       stamp=_now_iso(), host=platform.node())
    result = route_checkpoint(rec, route=args.route, public_target=args.public_target,
                              do_push=not args.no_push, dry=args.dry_run)

    if args.json:
        print(json.dumps({"record": rec, "result": result}, indent=2))
    else:
        print(f"checkpoint [{source}] -> route={args.route}  ok={result['ok']}")
        for r in result["results"]:
            tail = r.get("path") or r.get("target") or r.get("reason") or ""
            print(f"  {r['route']:8} ok={r.get('ok')}  {tail}")
    # Fail-SOFT: never non-zero on a routing miss (must not block Stop / crash the task).
    # Exit non-zero ONLY on a needle GATE refusal -- an operator wants to see that loudly.
    refused = any(("refused" in (r.get("reason") or "")) for r in result["results"])
    return 3 if refused else 0


if __name__ == "__main__":
    raise SystemExit(main())
