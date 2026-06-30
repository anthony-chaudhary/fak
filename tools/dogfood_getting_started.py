#!/usr/bin/env python3
"""dogfood_getting_started.py — spin up isolated agent workers that actually
RUN the getting-started docs, end to end, and collect a strict report per lane.

This is the durable harness behind the /dogfood-getting-started skill. It exists
because "the docs build clean" is a claim that rots silently: a flag is renamed,
a release stops shipping assets (#80), an installer points at a doc table that
no longer exists (#89) — and nothing fails until a brand-new user hits it. The
only honest check is to BE that user, in a clean room, on every cadence.

Design decisions that are load-bearing (learned the hard way):

  * The base snapshot is `git archive HEAD`, NOT a copy of the working tree.
    The working tree is frequently mid-merge in this repo (snapshot-refresh
    landings leave UU/DU files with `<<<<<<<` conflict markers in real .go
    sources). Copying that gives every lane a FALSE build break. HEAD is what a
    user who `git clone`s main actually gets, and it always compiles.

  * The specific docs UNDER TEST are then overlaid from the working tree, so the
    about-to-ship edits (a new tutorial, a changed GETTING-STARTED) are what the
    workers follow — the point is to catch doc bugs BEFORE they are published.
    By default we overlay exactly the getting-started docs that differ from HEAD.

  * Each lane is its own directory with no `.git` — a worker physically cannot
    pollute or commit to the real repo, and parallel `go build`s don't collide.

  * Workers are real, separate `opencode` processes (an OUTSIDE agent), launched
    through git-bash so the long multiline prompt is passed as one literal arg
    with no cmd.exe / .cmd quoting hell. An outside agent catches doc bugs that
    the doc's own author papers over with context.

Subcommands:
  setup    build the verified base snapshot + assemble one isolated dir per lane
  launch   start an opencode worker in each lane (blocking, per-lane timeout)
  collect  gather each lane's REPORT.json into one dated audit
  run      setup -> launch -> collect

Pure stdlib. Run from the repo root (or pass --repo).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import io
import json
import os
import shutil
import subprocess
from dispatch_worker import install_no_window_subprocess_defaults
import sys
import tarfile
import threading
from pathlib import Path
install_no_window_subprocess_defaults(subprocess)

LANES = ("A", "B", "C", "D")
DEFAULT_MODEL = "zai-coding-plan/glm-5.2"
# The getting-started doc/script surface whose working-tree edits we overlay
# onto the HEAD base (so the about-to-ship version is what gets dogfooded).
GETTING_STARTED_DOCS = (
    "fak/GETTING-STARTED.md",
    "docs/fak/tutorial.md",
    "docs/fak/README.md",
    "INSTALL.md",
    "install.sh",
    "fak/cmd/simpledemo/README.md",
)
# Prebuilt artifacts that must never seed a lane — workers build their own.
ARTIFACTS = ("fak/fak", "fak/fak.exe", "fak/cmd/simpledemo/simpledemo",
             "fak/cmd/fakchat/fakchat")


def _utc_today() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")


def _run(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


def _git_archive_head(repo: Path, dest: Path) -> int:
    """Extract a pristine HEAD snapshot into dest. Returns file count."""
    dest.mkdir(parents=True, exist_ok=True)
    raw = subprocess.run(
        ["git", "-C", str(repo), "archive", "--format=tar", "HEAD"],
        check=True, stdout=subprocess.PIPE).stdout
    with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
        # filter="data" is the safe extractor (py3.12+) and silences the 3.14
        # deprecation; this is our own HEAD tree, not an untrusted archive.
        try:
            tf.extractall(dest, filter="data")
        except TypeError:  # filter kw unavailable on <3.12
            tf.extractall(dest)  # noqa: S202
    return sum(1 for _ in dest.rglob("*") if _.is_file())


def _changed_getting_started_docs(repo: Path) -> list[str]:
    """Which getting-started docs differ from HEAD (staged or unstaged)?"""
    out = subprocess.run(
        ["git", "-C", str(repo), "diff", "--name-only", "HEAD", "--",
         *GETTING_STARTED_DOCS],
        check=False, stdout=subprocess.PIPE, text=True).stdout
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def _overlay(repo: Path, base: Path, rels: list[str]) -> list[str]:
    done = []
    for rel in rels:
        src = repo / rel
        if not src.is_file():
            continue
        dst = base / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        done.append(rel)
    return done


def _strip_artifacts(base: Path) -> None:
    for rel in ARTIFACTS:
        p = base / rel
        if p.exists():
            p.unlink()


def _lane_dir(work: Path, lane: str) -> Path:
    return work / f"lane{lane}"


def _prompt_for(lane: str, skill_dir: Path) -> Path:
    p = skill_dir / "lanes" / f"lane{lane}.txt"
    if not p.is_file():
        raise SystemExit(f"missing lane prompt template: {p}")
    return p


def cmd_setup(a) -> None:
    repo = Path(a.repo).resolve()
    work = Path(a.workdir).resolve()
    skill_dir = Path(a.skill_dir).resolve()
    base = work / "base"
    if base.exists():
        shutil.rmtree(base)
    n = _git_archive_head(repo, base)
    print(f"base: archived HEAD -> {base} ({n} files)")

    overlay = a.overlay or _changed_getting_started_docs(repo)
    if overlay:
        did = _overlay(repo, base, overlay)
        print(f"base: overlaid {len(did)} working-tree doc(s) under test: {', '.join(did)}")
    else:
        print("base: no getting-started docs differ from HEAD (testing committed docs as-is)")
    _strip_artifacts(base)

    for lane in a.lanes:
        d = _lane_dir(work, lane)
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(base, d)
        shutil.copy2(_prompt_for(lane, skill_dir), d / "PROMPT.txt")
        # Witness the isolation invariants that make a lane trustworthy.
        store = d / "fak/internal/blob/store.go"
        markers = 0
        if store.is_file():
            markers = store.read_text(encoding="utf-8", errors="ignore").count("\n<<<<<<<")
        assert not (d / ".git").exists(), f"lane{lane} leaked .git"
        assert not (d / "fak/fak.exe").exists(), f"lane{lane} leaked prebuilt binary"
        assert markers == 0, f"lane{lane} carries {markers} conflict markers (bad base!)"
        print(f"lane{lane}: {sum(1 for _ in d.rglob('*') if _.is_file())} files, "
              f"clean (no .git / no binary / 0 conflict markers)")
    print(f"\nsetup complete. launch with:\n  python {sys.argv[0]} launch "
          f"--workdir {work} --lanes {','.join(a.lanes)}")


def _bash_path() -> str:
    for c in ("bash", r"C:\Program Files\Git\bin\bash.exe",
              r"C:\Program Files\Git\usr\bin\bash.exe"):
        if shutil.which(c) or os.path.exists(c):
            return c
    raise SystemExit("git-bash not found; needed to launch opencode workers")


def _launch_one(lane: str, work: Path, model: str, timeout: int) -> dict:
    d = _lane_dir(work, lane)
    log = d / "opencode-stdout.log"
    runner = d / "run.sh"
    # A tiny git-bash runner: read the prompt as one literal arg (command
    # substitution is NOT re-scanned, so backticks/`$(...)` in the prompt are
    # safe), then exec opencode headless with permissions pre-approved (the lane
    # is a throwaway copy — the worker MUST be able to build/serve/curl freely).
    runner.write_text(
        "#!/usr/bin/env bash\n"
        f'cd "{d.as_posix()}"\n'
        "m=$(cat PROMPT.txt)\n"
        f'exec opencode run --dangerously-skip-permissions -m "{model}" "$m"\n',
        encoding="utf-8",
    )
    bash = _bash_path()
    with log.open("w") as fh:
        fh.write(f"# lane{lane} worker, model={model}, timeout={timeout}s\n")
        fh.flush()
        proc = subprocess.Popen([bash, str(runner)], stdout=fh,
                                stderr=subprocess.STDOUT, cwd=str(d))
        try:
            rc = proc.wait(timeout=timeout)
            status = f"exit={rc}"
        except subprocess.TimeoutExpired:
            proc.kill()
            status = f"TIMEOUT after {timeout}s"
    rep = d / "REPORT.json"
    print(f"lane{lane}: {status}; REPORT.json={'yes' if rep.is_file() else 'MISSING'}")
    return {"lane": lane, "status": status, "report": rep.is_file()}


def cmd_launch(a) -> None:
    work = Path(a.workdir).resolve()
    threads, results = [], []
    for lane in a.lanes:
        t = threading.Thread(
            target=lambda L=lane: results.append(
                _launch_one(L, work, a.model, a.timeout)))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    miss = [r["lane"] for r in results if not r["report"]]
    if miss:
        print(f"\nWARNING: lanes with no REPORT.json: {', '.join(miss)} "
              f"(read lane<X>/opencode-stdout.log for what happened)")


def cmd_collect(a) -> None:
    work = Path(a.workdir).resolve()
    audits = Path(a.audits_dir).resolve()
    audits.mkdir(parents=True, exist_ok=True)
    raw = {}
    for lane in a.lanes:
        d = _lane_dir(work, lane)
        rep = d / "REPORT.json"
        if rep.is_file():
            try:
                raw[lane] = json.loads(rep.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                raw[lane] = {"_unparseable": True,
                             "_tail": (d / "opencode-stdout.log").read_text(
                                 encoding="utf-8", errors="ignore")[-1500:]}
        else:
            raw[lane] = {"_missing": True,
                         "_tail": (d / "opencode-stdout.log").read_text(
                             encoding="utf-8", errors="ignore")[-1500:] if (d / "opencode-stdout.log").is_file() else ""}
    date = _utc_today()
    out_json = audits / f"dogfood-getting-started-{date}.json"
    out_json.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [f"# Dogfood: getting-started, {date}", "",
             "Isolated `opencode` workers ran the getting-started docs end to end "
             "in clean-room checkouts. One lane per doc surface.", ""]
    bug_total = 0
    for lane in a.lanes:
        r = raw[lane]
        lines.append(f"## Lane {lane}")
        if r.get("_missing") or r.get("_unparseable"):
            lines.append(f"- **no usable REPORT.json** ({'missing' if r.get('_missing') else 'unparseable'}); "
                         "see tail in the JSON sidecar.")
            lines.append("")
            continue
        lines.append(f"- overall: **{r.get('overall','?')}** — {r.get('summary','').strip()}")
        bugs = r.get("doc_bugs", []) or []
        bug_total += len(bugs)
        for b in bugs:
            lines.append(f"  - [{b.get('severity','?')}] `{b.get('doc','?')}` "
                         f"({b.get('location','?')}): {b.get('problem','')} "
                         f"— fix: {b.get('suggested_fix','')}")
        lines.append("")
    lines.insert(3, f"**{bug_total} doc finding(s)** across {len(a.lanes)} lanes. "
                    "Triage next: dedup vs the open-issue backlog, fix cheap doc bugs "
                    "in-tree, file new tickets for the rest.\n")
    out_md = audits / f"dogfood-getting-started-{date}.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"collected -> {out_md}")
    print(f"            {out_json}")
    print("\n" + "\n".join(lines))


def cmd_run(a) -> None:
    cmd_setup(a)
    cmd_launch(a)
    cmd_collect(a)


def _default_workdir() -> str:
    import tempfile
    return str(Path(tempfile.gettempdir()) / "fak-dogfood")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--workdir", default=_default_workdir())
    ap.add_argument("--skill-dir",
                    default=".claude/skills/dogfood-getting-started")
    ap.add_argument("--audits-dir", default="docs/_audits")
    ap.add_argument("--lanes", default=",".join(LANES),
                    type=lambda s: [x.strip().upper() for x in s.split(",") if x.strip()])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--overlay", nargs="*", default=None,
                    help="docs to overlay from the working tree onto the HEAD base "
                         "(default: getting-started docs that differ from HEAD)")
    sub = ap.add_subparsers(dest="sub", required=True)
    for name, fn in (("setup", cmd_setup), ("launch", cmd_launch),
                     ("collect", cmd_collect), ("run", cmd_run)):
        sp = sub.add_parser(name)
        sp.set_defaults(_fn=fn)
    a = ap.parse_args(argv)
    a._fn(a)


if __name__ == "__main__":
    main()
