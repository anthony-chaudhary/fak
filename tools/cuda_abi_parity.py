#!/usr/bin/env python3
"""CUDA ABI parity — the GPU-free static cross-check of the CUDA seam.

The CUDA backend is three files that have to agree on one flat C ABI:

    internal/compute/cuda_backend.h   the prototypes (`fcuda_*`)  — the typed seam
    internal/compute/cuda_kernels.cu  the definitions (`extern "C" … fcuda_*`)
    internal/compute/cuda.go          the cgo call sites (`C.fcuda_*`)

When they DISAGREE — a header prototype with no kernel definition, a `C.fcuda_…`
call the header never declared, a kernel that exports an `fcuda_*` symbol the header
forgot to expose — the failure surfaces only at the nvcc link / cgo build, which on
this project happens on a **remote GPU node** (the win32 dev host has no CUDA toolkit
and the GPU quota is walled). That is a slow, multi-host round trip to catch a typo.

This checker closes that loop on the laptop: it reads the three files as TEXT (no nvcc,
no GPU, no cgo) and reports every mismatch, so the whole class of "I renamed the kernel
but not the prototype" / "the binding calls a symbol the header doesn't declare" is
caught in milliseconds, locally, before the push. It is the local feedback loop the
remote-GPU dev process was missing — and a regression sentinel so the seam can't silently
drift as the kernel set grows.

It also checks the header as if it were parsed standalone by a strict host compiler:
portable fixed-width / size types must bring their own standard header in
cuda_backend.h, not arrive accidentally through nvcc or a platform-specific transitive
include. That is the cheap sentinel for the uint8_t-class CUDA portability bug.

Defect classes:
  HARD (a real build/link break the checker promotes to a local failure):
    - prototype with no definition   — declared in the header, never defined in the .cu
    - call with no prototype         — `C.fcuda_x` in cuda.go the header never declares
    - definition with no prototype   — `extern "C" … fcuda_x` in the .cu the header omits
    - header uses a standard type without including its defining header
  SOFT (dead/standby ABI surface — informational, never fails a gate):
    - prototype defined but never called from cuda.go (a utility kept for a future path
      or a test-only entry; listed with a reason in UNCALLED_OK so it stays honest)

Deterministic + read-only by construction: two clones of the same commit report the same
thing, and it edits nothing. Run from the repo ROOT::

    python tools/cuda_abi_parity.py            # human report
    python tools/cuda_abi_parity.py --json     # machine payload
    python tools/cuda_abi_parity.py --check     # exit non-zero on any HARD mismatch (CI/gate)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

SCHEMA = "fak-cuda-abi-parity/1"

HEADER = "internal/compute/cuda_backend.h"
KERNELS = "internal/compute/cuda_kernels.cu"
BINDING = "internal/compute/cuda.go"

# A CUDA-seam symbol: the `fcuda_` prefix plus a tail that MAY carry an uppercase
# suffix (fcuda_f32_to_f16_T). A prefix-only `[a-z0-9_]+` class silently truncates the
# `_T` and would mis-pair the symbol — the first correctness trap of this checker.
SYM = r"fcuda_[A-Za-z0-9_]+"
# A header PROTOTYPE / call: a `fcuda_name(` (name immediately followed by `(`, modulo
# whitespace). Comments are stripped first (see strip_comments), so this matches real
# declarations, not prose mentions.
_DECL_RE = re.compile(rf"\b({SYM})\s*\(")
# A .cu DEFINITION: an `extern "C" … fcuda_name(` — the exported symbol with a body. The
# return type/signature may wrap, but the `extern "C"` lead and the `fcuda_name(` sit on
# the same line in this codebase; the lead is what separates a definition from a call-site
# (a bare `fcuda_free(x)` inside a function body has no `extern "C"`). DOTALL so a wrapped
# `extern "C"\n void fcuda_x(` still pairs, while the `[^;{]*?` guard stops the non-greedy
# run from leaping across a prior statement into an unrelated symbol.
_DEF_RE = re.compile(rf'extern\s+"C"\s+[^;{{}}]*?\b({SYM})\s*\(', re.DOTALL)
# A cgo CALL: `C.fcuda_name(` in cuda.go.
_CALL_RE = re.compile(rf"\bC\.({SYM})\b")
_INCLUDE_RE = re.compile(r'(?m)^\s*#\s*include\s*[<"]([^>"]+)[>"]')

# The tiny standalone-header portability floor this checker enforces without needing a
# compiler. It deliberately covers only C standard types used by cuda_backend.h today:
# adding broader semantic lint here would turn a zero-false-positive gate into style lint.
HEADER_TYPE_INCLUDES: dict[str, str] = {
    "size_t": "stddef.h",
    "int8_t": "stdint.h",
    "uint8_t": "stdint.h",
    "int16_t": "stdint.h",
    "uint16_t": "stdint.h",
    "int32_t": "stdint.h",
    "uint32_t": "stdint.h",
    "int64_t": "stdint.h",
    "uint64_t": "stdint.h",
}

# Prototypes intentionally not called from cuda.go today — each with a reason, so an
# uncalled symbol is an honest documented standby, never silent dead weight. Keeps the
# SOFT list from being a place bit-rot hides.
UNCALLED_OK: dict[str, str] = {
    "fcuda_sync": "device-wide sync utility; the live path fences via Read/Argmax (fcuda_d2h / "
                  "fcuda_argmax_f32), so the explicit barrier is kept for diagnostics / future use",
}


# ---------------------------------------------------------------------------
# Pure core (the testable part): given the three file texts, compute the verdict.
# ---------------------------------------------------------------------------

def strip_comments(text: str) -> str:
    """Remove C/C++/Go comments, preserving newlines and string literals verbatim.

    The header and the .cu name `fcuda_*` symbols heavily in PROSE comments (the header's
    block comments mention fcuda_d2h / fcuda_argmax_f32; the .cu documents the seam in
    /* … */ banners). A naive 'extract every fcuda_ token' would count a comment-mention as
    a declaration, and a call-site inside a function body as a definition — a false PASS the
    moment a symbol exists ONLY in prose. Stripping comments makes the extraction match real
    CODE, not documentation — the un-gameable form of the parse.

    String / char literals are copied VERBATIM (not removed): the definition seam is
    `extern "C"` — collapsing the "C" literal would erase the very token that distinguishes
    a definition from a call-site. The literal is still SCANNED THROUGH so a `//` or `/*`
    inside a string (e.g. a URL) cannot spuriously open a comment. Newlines are preserved so
    the `extern "C" … fcuda_x(` window the definition regex matches stays line-aligned.
    """
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "*":                       # /* block comment */
            j = text.find("*/", i + 2)
            j = n if j == -1 else j + 2
            out.append("\n" * text.count("\n", i, j))      # keep the line count intact
            i = j
        elif c == "/" and nxt == "/":                     # // line comment
            j = text.find("\n", i)
            i = n if j == -1 else j
        elif c == '"' or c == "'":                        # string / char literal — copy verbatim
            out.append(c)
            j = i + 1
            while j < n and text[j] != c:
                out.append(text[j])
                if text[j] == "\\" and j + 1 < n:
                    out.append(text[j + 1])
                    j += 2
                else:
                    j += 1
            if j < n:
                out.append(text[j])                        # the closing quote
            i = j + 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def header_decls(text: str) -> set[str]:
    """The `fcuda_*` prototypes the header declares (name + `(`), comments stripped."""
    return {m.group(1) for m in _DECL_RE.finditer(strip_comments(text))}


def kernel_defs(text: str) -> set[str]:
    """The `fcuda_*` symbols the .cu exports with `extern "C"` + a body, comments stripped."""
    return {m.group(1) for m in _DEF_RE.finditer(strip_comments(text))}


def binding_calls(text: str) -> set[str]:
    """The `C.fcuda_*` symbols cuda.go calls through cgo, comments stripped."""
    return {m.group(1) for m in _CALL_RE.finditer(strip_comments(text))}


def header_includes(text: str) -> set[str]:
    """The direct includes named by the header, comments stripped."""
    return {m.group(1) for m in _INCLUDE_RE.finditer(strip_comments(text))}


def header_portability(text: str) -> dict[str, Any]:
    """Check that standard C types used by the header include their own standard header.

    This is the deterministic form of "parse cuda_backend.h alone": if the header names
    `uint8_t`, `int8_t`, or `size_t`, it must directly include the header that defines
    that type. A transitive include from WSL's nvcc, libc, or another source file cannot
    satisfy this check, so the datacenter-toolchain portability failure is caught before
    a GPU VM is provisioned.
    """
    stripped = strip_comments(text)
    includes = header_includes(stripped)
    used: dict[str, list[str]] = {}
    for typ, header in HEADER_TYPE_INCLUDES.items():
        if re.search(rf"\b{re.escape(typ)}\b", stripped):
            used.setdefault(header, []).append(typ)

    missing = {
        header: sorted(types)
        for header, types in sorted(used.items())
        if header not in includes
    }
    hard = [
        f"standalone header parse would fail: {HEADER} uses {', '.join(types)} but does "
        f"not include <{header}> directly — do not rely on nvcc/platform transitive includes"
        for header, types in missing.items()
    ]
    return {
        "includes": sorted(includes),
        "required": {header: sorted(types) for header, types in sorted(used.items())},
        "missing_includes": missing,
        "hard": hard,
    }


def parity(decls: set[str], defs: set[str], calls: set[str]) -> dict[str, Any]:
    """Cross the three symbol sets into HARD mismatches + SOFT standby notes.

    The ABI is the header's declared set. Every declared symbol must be defined in the
    .cu (else a call links to nothing); every call must be declared (else cgo can't see
    it); every exported definition must be declared (else the header is lying about the
    seam). Uncalled-but-declared is SOFT — a present, working entry no live path uses yet.
    """
    undefined = sorted(decls - defs)                 # HARD: prototype, no kernel
    undeclared_calls = sorted(calls - decls)         # HARD: call, no prototype
    undeclared_defs = sorted(defs - decls)           # HARD: kernel exported, header omits it
    uncalled = sorted(decls - calls)                 # SOFT: declared+(usually)defined, unused in .go

    hard: list[str] = []
    for s in undefined:
        hard.append(f"prototype with no definition: {s}() is declared in {HEADER} but never "
                    f"defined (extern \"C\") in {KERNELS} — a call would link to nothing")
    for s in undeclared_calls:
        hard.append(f"call with no prototype: cuda.go calls C.{s}() but {HEADER} declares no "
                    f"such symbol — the cgo build cannot see it")
    for s in undeclared_defs:
        hard.append(f"definition with no prototype: {KERNELS} exports {s}() (extern \"C\") but "
                    f"{HEADER} declares no prototype — the seam is undocumented / unreachable")

    soft: list[str] = []
    for s in uncalled:
        if s in UNCALLED_OK:
            soft.append(f"{s}() declared but not called from cuda.go — OK: {UNCALLED_OK[s]}")
        else:
            soft.append(f"{s}() declared in {HEADER} but never called from cuda.go — dead ABI "
                        "surface? add a caller, drop the prototype, or list it in UNCALLED_OK with a reason")
    return {
        "undefined": undefined,
        "undeclared_calls": undeclared_calls,
        "undeclared_defs": undeclared_defs,
        "uncalled": uncalled,
        "hard": hard,
        "soft": soft,
    }


def build_payload(*, workspace: str, decls: set[str], defs: set[str], calls: set[str],
                  header_text: str | None = None, error: str | None = None) -> dict[str, Any]:
    if error:
        return {"schema": SCHEMA, "ok": False, "verdict": "AUDIT_ERROR",
                "reason": error, "workspace": workspace, "corpus": {}}
    p = parity(decls, defs, calls)
    hp = header_portability(header_text) if header_text is not None else {
        "includes": [], "required": {}, "missing_includes": {}, "hard": [],
    }
    hard = list(p["hard"]) + list(hp["hard"])
    n_hard = len(hard)
    n_soft = len([s for s in p["soft"] if " — OK: " not in s])  # documented-OK don't count as advisory
    corpus = {
        "n_symbols": len(decls | defs | calls),
        "n_declared": len(decls),
        "n_defined": len(defs),
        "n_called": len(calls),
        "hard_mismatches": n_hard,
        "soft_signals": n_soft,
        "header_portability": hp,
        "undefined": p["undefined"],
        "undeclared_calls": p["undeclared_calls"],
        "undeclared_defs": p["undeclared_defs"],
        "uncalled": p["uncalled"],
        "hard": hard,
        "soft": p["soft"],
    }
    ok = n_hard == 0
    if ok:
        verdict, reason = "OK", (
            f"CUDA header is standalone-portable and ABI is in parity: {len(decls)} prototypes, "
            f"all defined in {KERNELS} and declared for every C.fcuda_* call in cuda.go "
            f"({n_soft} standby symbol(s) advisory)")
    else:
        verdict, reason = "ACTION", (
            f"{n_hard} CUDA seam issue(s) — the header / kernels / cgo binding disagree, or "
            "cuda_backend.h is not standalone-portable under a strict host compiler")
    return {"schema": SCHEMA, "ok": ok, "verdict": verdict, "reason": reason,
            "workspace": workspace, "corpus": corpus}


# ---------------------------------------------------------------------------
# Thin disk shell.
# ---------------------------------------------------------------------------

def repo_root(start: Path | None = None) -> Path:
    return (start or Path(__file__)).resolve().parent.parent


def _read(root: Path, rel: str) -> str:
    try:
        return (root / rel).read_text(encoding="utf-8")
    except OSError:
        return ""


def collect(root: Path) -> dict[str, Any]:
    root = root.resolve()
    htext, ktext, gtext = _read(root, HEADER), _read(root, KERNELS), _read(root, BINDING)
    missing = [rel for rel, t in ((HEADER, htext), (KERNELS, ktext), (BINDING, gtext)) if not t]
    if missing:
        return build_payload(workspace=str(root), decls=set(), defs=set(), calls=set(),
                             error=f"cannot read CUDA seam file(s): {', '.join(missing)} "
                                   "(run from the repo ROOT)")
    return build_payload(workspace=str(root), decls=header_decls(htext),
                         defs=kernel_defs(ktext), calls=binding_calls(gtext),
                         header_text=htext)


def render(payload: dict[str, Any]) -> str:
    c = payload.get("corpus") or {}
    lines = [
        f"cuda-abi-parity: {payload.get('verdict')}",
        f"  {payload.get('reason')}",
        "",
        (f"symbols: {c.get('n_declared', 0)} declared · {c.get('n_defined', 0)} defined · "
         f"{c.get('n_called', 0)} called   |   HARD {c.get('hard_mismatches', 0)} · "
         f"advisory {c.get('soft_signals', 0)}"),
    ]
    if c.get("hard"):
        lines.append("")
        lines.append("HARD mismatches (would break the GPU build / strict header parse):")
        for h in c["hard"]:
            lines.append(f"  x {h}")
    if c.get("soft"):
        lines.append("")
        lines.append("advisory (standby / dead-surface):")
        for s in c["soft"]:
            lines.append(f"  . {s}")
    if not c.get("hard"):
        lines.append("")
        lines.append("OK - header <-> kernels <-> cgo binding agree on every fcuda_* symbol.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CUDA ABI parity static cross-check (read-only).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero on any HARD mismatch (for a gate / CI / make target)")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass
    root = Path(args.workspace).resolve() if args.workspace else repo_root()
    payload = collect(root)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(render(payload))
    if args.check:
        return 0 if payload.get("ok") else 1
    return 0 if payload.get("verdict") != "AUDIT_ERROR" else 2


if __name__ == "__main__":
    raise SystemExit(main())
