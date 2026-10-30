#!/usr/bin/env python3
"""CUDA ABI parity — the GPU-free static cross-check of the CUDA seam.

The CUDA backend has three source sets that must agree on one flat C ABI:

    internal/compute/cuda_backend.h   the base prototypes (`fcuda_*`)
    internal/compute/cuda_kv_precision.h, cuda_mxfp8_cast.h  modular prototypes
    internal/compute/cuda_kernels.cu  the base definitions (`extern "C" … fcuda_*`)
    internal/compute/cuda_nccl.cu     the NCCL definitions (`extern "C" … fcuda_*`)
    internal/compute/cuda_nccl_pg.cu  the multi-process NCCL PG definitions (`extern "C" … fcuda_*`)
    internal/compute/cuda*.go         the cgo call sites (`C.fcuda_*`) — their `/* … */`
                                     preambles may also carry `extern … fcuda_*(…);`
                                     prototypes (the device-work shim declares its Go-facing
                                     ABI next to the call sites, not in an ABI header)

When they DISAGREE — a header or preamble prototype with no kernel definition, a
`C.fcuda_…` call neither the headers nor its cgo preamble declared, a kernel that exports
an `fcuda_*` symbol the seam forgot to declare — the failure surfaces only at the nvcc
link / cgo build, which on
this project happens on a **remote GPU node** (the win32 dev host has no CUDA toolkit
and the GPU quota is walled). That is a slow, multi-host round trip to catch a typo.

This checker closes that loop on the laptop: it reads the seam and the roots' local
quoted includes as TEXT (no nvcc, no GPU, no cgo) and reports every mismatch, so the
whole class of "I renamed the kernel
but not the prototype" / "the binding calls a symbol the header doesn't declare" is
caught in milliseconds, locally, before the push. It is the local feedback loop the
remote-GPU dev process was missing — and a regression sentinel so the seam can't silently
drift as the kernel set grows.

It also checks the header as if it were parsed standalone by a strict host compiler:
portable fixed-width / size types must bring their own standard header in
each ABI header, not arrive accidentally through nvcc or a platform-specific transitive
include. That is the cheap sentinel for the uint8_t-class CUDA portability bug.

This is global symbol-set parity, not a C/C++ parser: signature compatibility,
duplicate definitions, and each cgo preamble's visibility still require compilation.

Defect classes:
  HARD (a real build/link break the checker promotes to a local failure):
    - prototype with no definition   — declared in a header or cgo preamble, never defined in the .cu
    - call with no prototype         — `C.fcuda_x` in cuda.go that neither the headers nor its
                                       cgo preamble declares
    - definition with no prototype   — `extern "C" … fcuda_x` in the .cu the seam omits
    - header uses a standard type without including its defining header
  Documented exceptions (advisory, never fail a gate — each must carry a reason):
    - UNCALLED_OK                    — standby prototypes declared but never called from cuda.go
    - UNDECLARED_DEF_OK              — intentionally unbound C-side exports (shim plumbing /
                                       diagnostics) that no Go path binds today

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
# Explicit ABI roots, not every header found under compute/. The typed KV header is
# included by cgo consumers; MXFP8 is a standalone API included by its native source.
HEADERS = (HEADER, "internal/compute/cuda_kv_precision.h", "internal/compute/cuda_mxfp8_cast.h")
HEADERS_LABEL = ", ".join(HEADERS)
KERNELS = ("internal/compute/cuda_kernels.cu", "internal/compute/cuda_nccl.cu", "internal/compute/cuda_nccl_pg.cu")
KERNELS_LABEL = ", ".join(KERNELS)
BINDINGS = (
    "internal/compute/cuda.go",
    "internal/compute/cuda_collective.go",
    "internal/compute/cuda_collective_pg.go",
    "internal/compute/cuda_qwen4exp_ple_hash_cuda.go",
    "internal/compute/cuda_routed_expert_gemv.go",
)
BINDINGS_LABEL = ", ".join(BINDINGS)

# A CUDA-seam symbol: the `fcuda_` prefix plus a tail that MAY carry an uppercase
# suffix (fcuda_f32_to_f16_T). A prefix-only `[a-z0-9_]+` class silently truncates the
# `_T` and would mis-pair the symbol — the first correctness trap of this checker.
SYM = r"fcuda_[A-Za-z0-9_]+"
# A header PROTOTYPE / call: a `fcuda_name(` (name immediately followed by `(`, modulo
# whitespace). Comments are stripped first (see strip_comments), so this matches real
# declarations, not prose mentions.
_DECL_RE = re.compile(rf"\b({SYM})\s*\(")
# A .cu DEFINITION: an `extern "C" … fcuda_name(` — the exported symbol with a body. The
# return type/signature may wrap; the lead separates a definition from a call-site
# (a bare `fcuda_free(x)` inside a function body has no `extern "C"`). DOTALL so a wrapped
# `extern "C"\n void fcuda_x(` still pairs, while the `[^;{]*?` guard stops the non-greedy
# run from leaping across a prior statement into an unrelated symbol. Requiring the
# body also excludes prototypes in headers reached through quoted includes.
_DEF_RE = re.compile(rf'extern\s+"C"\s+[^;{{}}]*?\b({SYM})\s*\([^;{{}}]*\)\s*\{{', re.DOTALL)
# A cgo CALL: `C.fcuda_name(` in cuda.go.
_CALL_RE = re.compile(rf"\bC\.({SYM})\b")
# A cgo PREAMBLE PROTOTYPE: `extern <type> fcuda_name(…);` — a plain extern declaration
# (NOT `extern "C"`, which marks a .cu definition) ending in `;` (not a `{` body). The
# device-work shim (cuda/kernel_profile.cuh) declares its Go-facing prototypes in cuda.go's
# preamble instead of an ABI header; cgo compiles that preamble as the declaration surface
# for the translation unit, so these are real prototypes the linker sees.
_PREAMBLE_DECL_RE = re.compile(rf"\bextern\s+(?:const\s+)?[A-Za-z_]\w*\s*\**\s*({SYM})\s*\([^;{{}}]*\)\s*;")
# The cgo preamble itself: the /* … */ block directly preceding import "C". Go treats it as
# a comment, so it must be recovered from the RAW text before comment stripping.
_CGO_PREAMBLE_RE = re.compile(r'/\*(.*)\*/\s*import\s+"C"', re.DOTALL)
_INCLUDE_RE = re.compile(r'(?m)^\s*#\s*include\s*[<"]([^>"]+)[>"]')
_QUOTED_INCLUDE_RE = re.compile(r'(?m)^[ \t]*#[ \t]*include[ \t]*"([^"\r\n]+)"')

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
    "fcuda_set_device": "C-side helper used by the device-specific allocation/copy helpers; Go calls "
                        "the higher-level fcuda_*_on wrappers, not cudaSetDevice directly",
    "fcuda_nccl_world": "diagnostic/world query kept with the NCCL ABI; Go gates Caps().Collective "
                        "from cudaNCCLWorld after fcuda_nccl_init succeeds",
}

# Exported `extern "C"` definitions the seam intentionally does NOT bind from Go — each with
# a reason, mirroring UNCALLED_OK. They stay visible in the advisory report; any NEW
# unbound export still fails the gate until someone documents why it is unbound.
UNDECLARED_DEF_OK: dict[str, str] = {
    "fcuda_dw_site_name": "diagnostic site-name lookup kept beside the Go-bound fcuda_dw_stage_name in the "
                          "device-work shim (cuda/kernel_profile.cuh); no Go path binds it today",
    "fcuda_dw_note_unobserved": "C-side hook the shim's graph-launch wrapper (fcuda_dw_graph_launch) calls to mark "
                                "the counters incomplete; internal plumbing, never bound from Go",
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


def binding_preamble_decls(text: str) -> set[str]:
    """The `extern … fcuda_*(…);` prototypes a binding's cgo preamble declares.

    The preamble is a Go block comment, so strip_comments would erase it: pull the raw
    block out first, then strip the // line comments that may sit inside it (string
    literals stay verbatim, as elsewhere in this checker).
    """
    decls: set[str] = set()
    for m in _CGO_PREAMBLE_RE.finditer(text):
        body = strip_comments(m.group(1))
        decls.update(x.group(1) for x in _PREAMBLE_DECL_RE.finditer(body))
    return decls


def header_includes(text: str) -> set[str]:
    """The direct includes named by the header, comments stripped."""
    return {m.group(1) for m in _INCLUDE_RE.finditer(strip_comments(text))}


def header_portability(text: str, label: str = HEADER) -> dict[str, Any]:
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
        f"standalone header parse would fail: {label} uses {', '.join(types)} but does "
        f"not include <{header}> directly — do not rely on nvcc/platform transitive includes"
        for header, types in missing.items()
    ]
    return {
        "includes": sorted(includes),
        "required": {header: sorted(types) for header, types in sorted(used.items())},
        "missing_includes": missing,
        "hard": hard,
    }


def parity(decls: set[str], defs: set[str], calls: set[str],
           preamble_decls: set[str] | None = None) -> dict[str, Any]:
    """Cross the three symbol sets into HARD mismatches + SOFT standby notes.

    The ABI is the declared set: the headers' prototypes plus any `extern … fcuda_*(…);`
    prototypes a binding's cgo preamble carries (the device-work shim pattern). Every
    declared symbol must be defined in the .cu (else a call links to nothing); every call
    must be declared somewhere cgo can see it (else the build breaks); every exported
    definition must be declared (else the seam is lying about its surface) — except the
    documented UNDECLARED_DEF_OK exceptions. Uncalled-but-declared is SOFT — a present,
    working entry no live path uses yet.
    """
    preamble_decls = preamble_decls or set()
    undefined = sorted(decls - defs)                 # HARD: prototype, no kernel
    undeclared_calls = sorted(calls - decls)         # HARD: call, no prototype anywhere
    undeclared_defs = sorted(defs - decls)           # HARD: kernel exported, seam omits it
    uncalled = sorted(decls - calls)                 # SOFT: declared+(usually)defined, unused in .go

    hard: list[str] = []
    for s in undefined:
        where = (f"a cgo preamble ({BINDINGS_LABEL})" if s in preamble_decls
                 else f"the ABI header closure ({HEADERS_LABEL})")
        hard.append(f"prototype with no definition: {s}() is declared in {where} but never "
                    f"defined (extern \"C\") in {KERNELS_LABEL} — a call would link to nothing")
    for s in undeclared_calls:
        hard.append(f"call with no prototype: {BINDINGS_LABEL} calls C.{s}() but neither the ABI header closure "
                    f"({HEADERS_LABEL}) nor any cgo preamble declares it — the cgo build cannot see it")
    soft: list[str] = []
    for s in undeclared_defs:
        if s in UNDECLARED_DEF_OK:
            soft.append(f"{s}() exported by {KERNELS_LABEL} with no prototype — OK: {UNDECLARED_DEF_OK[s]}")
        else:
            hard.append(f"definition with no prototype: {KERNELS_LABEL} exports {s}() (extern \"C\") but "
                        "the ABI header closure and every cgo preamble declare no prototype — the seam is "
                        "undocumented / unreachable")
    for s in uncalled:
        if s in UNCALLED_OK:
            soft.append(f"{s}() declared but not called from {BINDINGS_LABEL} — OK: {UNCALLED_OK[s]}")
        else:
            soft.append(f"{s}() declared in the ABI header closure but never called from {BINDINGS_LABEL} — dead ABI "
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
                  header_text: str | None = None, header_texts: dict[str, str] | None = None,
                  preamble_decls: set[str] | None = None,
                  error: str | None = None) -> dict[str, Any]:
    if error:
        return {"schema": SCHEMA, "ok": False, "verdict": "AUDIT_ERROR",
                "reason": error, "workspace": workspace, "corpus": {}}
    preamble_decls = preamble_decls or set()
    all_decls = decls | preamble_decls
    p = parity(all_decls, defs, calls, preamble_decls=preamble_decls - decls)
    hp = header_portability(header_text) if header_text is not None else {
        "includes": [], "required": {}, "missing_includes": {}, "hard": [],
    }
    headers = {rel: header_portability(text, rel)
               for rel, text in (header_texts or {}).items()}
    hard = list(p["hard"]) + (list(h for report in headers.values() for h in report["hard"])
                              if headers else list(hp["hard"]))
    n_hard = len(hard)
    n_soft = len([s for s in p["soft"] if " — OK: " not in s])  # documented-OK don't count as advisory
    corpus = {
        "n_symbols": len(all_decls | defs | calls),
        "n_declared": len(all_decls),
        "preamble_decls": sorted(preamble_decls),
        "n_defined": len(defs),
        "n_called": len(calls),
        "hard_mismatches": n_hard,
        "soft_signals": n_soft,
        "header_portability": hp,
        "headers": headers,
        "undefined": p["undefined"],
        "undeclared_calls": p["undeclared_calls"],
        "undeclared_defs": p["undeclared_defs"],
        "uncalled": p["uncalled"],
        "hard": hard,
        "soft": p["soft"],
    }
    ok = n_hard == 0
    if ok:
        reason = (
            f"CUDA ABI headers are standalone-portable and symbol sets are in parity: {len(all_decls)} prototypes, "
            f"all defined in {KERNELS_LABEL} and declared for every C.fcuda_* call in {BINDINGS_LABEL} "
            f"({n_soft} standby symbol(s) advisory)")
        if preamble_decls - decls:
            reason += f"; {len(preamble_decls - decls)} prototype(s) carried in cgo preambles"
        verdict, reason = "OK", reason
    else:
        verdict, reason = "ACTION", (
            f"{n_hard} CUDA seam issue(s) — the header / kernels / cgo binding disagree, or "
            "an ABI header is not standalone-portable under a strict host compiler")
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


def _read_sources(root: Path, rels: tuple[str, ...], *, headers_only: bool = False
                  ) -> tuple[dict[str, str], list[str]]:
    """Read only the roots' repository-local quoted-include closure.

    This is source discovery, not a preprocessor: system headers and conditional
    compilation are not interpreted. Canonical paths bound the traversal to the
    repository and deduplicate cycles; root and include order keep it deterministic.
    A missing or escaping include must fail the audit, never yield partial parity.
    """
    pending = [root / rel for rel in reversed(rels)]
    roots = set(pending)
    seen: set[Path] = set()
    texts: dict[str, str] = {}
    errors: list[str] = []
    while pending:
        source = pending.pop()
        rel = source.relative_to(root).as_posix()
        try:
            path = source.resolve(strict=True)
            if not path.is_relative_to(root):
                errors.append(f"{rel} (quoted include escapes repository)")
                continue
            if headers_only and (source.suffix != ".h" or path.suffix != ".h"):
                errors.append(f"{rel} (ABI header closure requires .h files, not source bodies)")
                continue
            # Open the original spelling before deduplication: normalization must
            # not hide an unreadable alias such as missing/../leaf.cu.
            text = source.read_text(encoding="utf-8")
            if not text and source in roots:
                errors.append(rel)
            if path in seen:
                continue
            seen.add(path)
        except (OSError, UnicodeError, RuntimeError, ValueError):
            errors.append(rel)
            continue
        stripped = strip_comments(text)
        texts[path.relative_to(root).as_posix()] = stripped
        includes: list[Path] = []
        for match in _QUOTED_INCLUDE_RE.finditer(stripped):
            name = match.group(1)
            if Path(name).is_absolute():
                errors.append(f'{rel}: "{name}" (quoted include must be repository-relative)')
            else:
                includes.append(source.parent / name)
        pending.extend(reversed(includes))
    return texts, errors


def _read_kernel_sources(root: Path) -> tuple[str, list[str]]:
    texts, errors = _read_sources(root, KERNELS)
    return "\n".join(texts.values()), errors


def collect(root: Path) -> dict[str, Any]:
    root = root.resolve()
    htexts, hmissing = _read_sources(root, HEADERS, headers_only=True)
    ktext, kmissing = _read_kernel_sources(root)
    gtexts: dict[str, str] = {}
    gmissing: list[str] = []
    for rel in BINDINGS:
        text = _read(root, rel)
        if not text:
            gmissing.append(rel)
        gtexts[rel] = text
    missing = hmissing + kmissing + gmissing
    if missing:
        return build_payload(workspace=str(root), decls=set(), defs=set(), calls=set(),
                             error=f"cannot read CUDA seam file(s): {', '.join(missing)} "
                                   "(run from the repo ROOT)")
    gtext = "\n".join(gtexts.values())
    return build_payload(workspace=str(root), decls=header_decls("\n".join(htexts.values())),
                         defs=kernel_defs(ktext), calls=binding_calls(gtext),
                         preamble_decls={s for text in gtexts.values() for s in binding_preamble_decls(text)},
                         header_text=htexts.get(HEADER), header_texts=htexts)


def render(payload: dict[str, Any]) -> str:
    c = payload.get("corpus") or {}
    lines = [
        f"cuda-abi-parity: {payload.get('verdict')}",
        f"  {payload.get('reason')}",
        "",
        (f"symbols: {c.get('n_declared', 0)} declared · {c.get('n_defined', 0)} defined · "
         f"{c.get('n_called', 0)} called · {len(c.get('preamble_decls') or [])} in preambles   |   "
         f"HARD {c.get('hard_mismatches', 0)} · advisory {c.get('soft_signals', 0)}"),
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
