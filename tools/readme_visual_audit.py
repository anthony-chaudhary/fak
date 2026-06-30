#!/usr/bin/env python3
"""README visual-density auditor — every README earns at least one diagram.

A wall of prose is the easiest doc to skip and the hardest to trust. The front
``README.md`` already leads with rendered charts, a hero reel, and ASCII
before/after blocks; the *rest* of the tree's READMEs (``examples/*``,
``docs/*``, ``experiments/*``, the skills index) were mostly text-only. This is
the checking layer that makes "a visual by default" a property of the tree
rather than a thing a human happens to add: it folds every git-tracked
``README.md`` and reports one typed verdict per file plus an ``ok`` bit the
control-pane reads first.

It is the visual-surface sibling of ``readme_freshness_audit.py`` (which checks
the front page's *facts*); this one checks that every README carries a
*diagram-class* visual a reader can see at a glance. Read-only by construction:
it never edits a README; it only checks it.

What counts as a visual (any one is enough):

  mermaid        a fenced ```mermaid block                 (GitHub renders it inline)
  diagram image  an embedded ![alt](…​.svg|.png|.gif|…)     (a chart/diagram, NOT a badge)
  ascii diagram  a fenced block with box-drawing / arrow    (renders identically everywhere)
                 or bar/block-chart lines (│┌┐└┘├┤┬┴┼─,
                 ▶→↓, -->, +--, and bar glyphs █▉▌░▒▓▁▄▇)

A lone shields.io / Colab *badge* does NOT count — a badge is a link decoration,
not a diagram, and the goal is a picture that explains the page. Markdown
*tables* also do not count: nearly every README already has one, so they make a
poor bar. The verdict per README is OK (has a visual) or FAIL (text-only).
``ok`` is False iff any tracked README is text-only.

Run from the repo ROOT (``python tools/readme_visual_audit.py``); pure
filesystem + ``git ls-files``, no ``dos`` subprocess. The companion process is
the ``/refresh-readme`` family and the "more visuals" pass, which read this
audit's FAILs as their work-list.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from dispatch_worker import install_no_window_subprocess_defaults
install_no_window_subprocess_defaults(subprocess)
from pathlib import Path
from typing import Any

SCHEMA = "fleet-readme-visual-audit/1"

# Tracked README globs we audit. Vendored/generated caches are excluded below so
# the glob fallback (when git is unavailable) matches the git-tracked set.
README_GLOBS = ("**/README.md", "**/readme.md")

# Path fragments that mark a generated / vendored / cache README we do not own.
EXCLUDE_FRAGMENTS = (
    "/.pytest_cache/", "/_registry/", "/node_modules/", "/.git/",
    "/vendor/", "/testdata/",
)

# Box-drawing + arrow glyphs that signal a hand-drawn ASCII diagram. Distinct
# from the `|` and `-` a Markdown TABLE uses, so a table never trips this.
# The second cluster is bar/block-chart glyphs (a scorecard's at-a-glance chart
# is a visual too) — also absent from any normal table or prose.
_BOX_GLYPHS = (
    "│┌┐└┘├┤┬┴┼─━┃┏┓┗┛┣┫┳┻╋═║╔╗╚╝╠╣╦╩╬▶◀▲▼►◄→←↑↓↳↴↦⟶⟵⇒⇐⇨"
    "█▉▊▋▌▍▎▏▕▁▂▃▄▅▆▇░▒▓▬■"
)
# ASCII arrow / connector tokens that also signal a diagram inside a code fence.
_ASCII_ARROWS = ("-->", "==>", "<--", "<==", "+--", "--+", "|--", "--|", ".->", "o--")

# A fenced code block: opening ``` (with optional info string) … closing ```.
# group("info") is the language tag; group("body") is the fenced content.
_FENCE_RE = re.compile(
    r"```(?P<info>[^\n`]*)\n(?P<body>.*?)```", re.DOTALL)

# An embedded image: ![alt](target). Capture the target so we can classify it.
_IMG_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<target>[^)\s]+)")

# Image targets that are decoration, not a diagram, so they do NOT satisfy the bar.
_BADGE_MARKERS = (
    "shields.io", "img.shields", "badge", "colab.research.google.com/assets",
    "/badges/", "buymeacoffee", "ko-fi", "/actions/workflows/",
)
_IMG_EXT_RE = re.compile(r"\.(svg|png|jpe?g|gif|webp|avif)(?:[?#].*)?$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Pure detectors: each takes already-read README text and returns a bool/detail.
# This is the testable seam — tests pass fixture strings, no disk needed.
# ---------------------------------------------------------------------------

def _fences(text: str) -> list[tuple[str, str]]:
    """Every fenced code block as (info_string, body)."""
    return [(m.group("info").strip(), m.group("body"))
            for m in _FENCE_RE.finditer(text)]


def has_mermaid(text: str) -> bool:
    """True if any fenced block is tagged ```mermaid (GitHub renders it inline)."""
    return any(info.lower().startswith("mermaid") for info, _ in _fences(text))


def diagram_images(text: str) -> list[str]:
    """Embedded image targets that are diagrams (a chart/figure), not badges."""
    out: list[str] = []
    for m in _IMG_RE.finditer(text):
        target = m.group("target")
        low = target.lower()
        if any(b in low for b in _BADGE_MARKERS):
            continue
        if _IMG_EXT_RE.search(low) or low.startswith(("visuals/", "../visuals/", "../../visuals/")):
            out.append(target)
    return out


def ascii_diagram_blocks(text: str) -> int:
    """Count fenced blocks that read as a hand-drawn ASCII/box diagram.

    A block qualifies when >= 2 of its lines carry a box-drawing glyph or an
    ASCII arrow token. The two-line floor keeps a stray arrow in a shell snippet
    (``foo | bar``, ``a --> b`` in prose) from masquerading as a diagram.
    """
    qualifying = 0
    for _info, body in _fences(text):
        hits = 0
        for line in body.splitlines():
            if any(g in line for g in _BOX_GLYPHS) or any(a in line for a in _ASCII_ARROWS):
                hits += 1
                if hits >= 2:
                    qualifying += 1
                    break
    return qualifying


def audit_one(text: str) -> dict[str, Any]:
    """Classify one README's visual surface into a {kinds, has_visual} verdict."""
    mermaid = has_mermaid(text)
    imgs = diagram_images(text)
    ascii_n = ascii_diagram_blocks(text)
    kinds: list[str] = []
    if mermaid:
        kinds.append("mermaid")
    if imgs:
        kinds.append(f"image×{len(imgs)}")
    if ascii_n:
        kinds.append(f"ascii×{ascii_n}")
    return {"has_visual": bool(kinds), "kinds": kinds,
            "mermaid": mermaid, "images": imgs, "ascii_blocks": ascii_n}


# ---------------------------------------------------------------------------
# Wiring: enumerate tracked READMEs, fold per-file verdicts into the payload.
# ---------------------------------------------------------------------------

def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


def _excluded(rel: str) -> bool:
    slug = "/" + rel.replace("\\", "/")
    return any(frag in slug for frag in EXCLUDE_FRAGMENTS)


def list_readmes(root: Path) -> list[str]:
    """Tracked README paths (POSIX-relative), git first, glob as fallback."""
    try:
        cp = subprocess.run(
            ["git", "-C", str(root), "ls-files", "*README.md", "*readme.md"],
            capture_output=True, text=True, check=False)
        if cp.returncode == 0 and cp.stdout.strip():
            rels = [ln.strip() for ln in cp.stdout.splitlines() if ln.strip()]
            return sorted(r for r in rels if not _excluded(r))
    except OSError:
        pass
    rels = set()
    for pat in README_GLOBS:
        for p in root.glob(pat):
            rels.add(p.relative_to(root).as_posix())
    return sorted(r for r in rels if not _excluded(r))


def collect(workspace: Path) -> dict[str, Any]:
    root = workspace.resolve()
    rels = list_readmes(root)
    if not rels:
        return build_payload(workspace=str(root), checks=[],
                             error="no tracked README.md found (run from repo ROOT)")
    checks: list[dict[str, Any]] = []
    for rel in rels:
        try:
            text = (root / rel).read_text(encoding="utf-8")
        except OSError as exc:
            checks.append({"check": rel, "status": "FAIL",
                           "detail": f"cannot read: {exc}"})
            continue
        v = audit_one(text)
        if v["has_visual"]:
            checks.append({"check": rel, "status": "OK",
                           "detail": "has " + ", ".join(v["kinds"])})
        else:
            checks.append({"check": rel, "status": "FAIL",
                           "detail": "text-only: no mermaid, diagram image, or ASCII diagram"})
    return build_payload(workspace=str(root), checks=checks)


# ---------------------------------------------------------------------------
# Grader: fold the per-README list into the standard control-pane payload.
# ---------------------------------------------------------------------------

def build_payload(*, workspace: str, checks: list[dict[str, Any]],
                  error: str | None = None) -> dict[str, Any]:
    counts = {"OK": 0, "WARN": 0, "FAIL": 0, "ADVISORY": 0}
    for c in checks:
        counts[c["status"]] = counts.get(c["status"], 0) + 1
    fails = [c for c in checks if c["status"] == "FAIL"]
    total = len(checks)

    if error:
        ok, verdict, finding = False, "AUDIT_ERROR", "tooling_error"
        reason = error
        next_action = "run from the repo ROOT so `git ls-files *README.md` resolves, then re-run"
    elif fails:
        ok, verdict, finding = False, "ACTION", "readmes_text_only"
        names = ", ".join(c["check"] for c in fails[:6])
        more = "" if len(fails) <= 6 else f" (+{len(fails) - 6} more)"
        reason = f"{len(fails)}/{total} README(s) are text-only: {names}{more}"
        next_action = ("add one diagram-class visual to each FAIL — a ```mermaid block "
                       "(GitHub-rendered pages), an embedded visuals/*.png, or a fenced "
                       "ASCII diagram (Jekyll-served docs/*) — accurate to the page, no new claims")
    else:
        ok, verdict, finding = True, "OK", "all_readmes_visual"
        reason = f"all {total} tracked README(s) carry a diagram-class visual"
        next_action = "no action; re-run after adding a README or stripping a diagram"

    return {
        "schema": SCHEMA,
        "ok": ok,
        "verdict": verdict,
        "finding": finding,
        "reason": reason,
        "next_action": next_action,
        "workspace": workspace,
        "counts": counts,
        "checks": checks,
    }


def render(payload: dict[str, Any]) -> str:
    counts = payload.get("counts") or {}
    total = sum(counts.get(k, 0) for k in ("OK", "WARN", "FAIL", "ADVISORY"))
    lines = [
        f"readme-visual audit: {payload.get('verdict')} ({payload.get('finding')})",
        f"readmes: {counts.get('OK', 0)}/{total} have a visual · fail={counts.get('FAIL', 0)}",
        f"next: {payload.get('next_action')}",
    ]
    for c in payload.get("checks", []):
        mark = {"OK": "  ok ", "FAIL": " FAIL"}.get(c["status"], "  ?  ")
        lines.append(f"{mark}  {c['check']:<48} {c['detail']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="README visual-density auditor (read-only): every README earns a diagram.")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args(argv)

    workspace = Path(args.workspace).resolve() if args.workspace else repo_root()
    payload = collect(workspace)

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(render(payload))

    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
