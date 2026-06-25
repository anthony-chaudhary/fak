#!/usr/bin/env python3
"""CUDA-dev-process scorecard — the measuring stick for the CUDA development LOOP.

The sibling scorecards grade a *surface* (``code_quality`` the Go module, ``docs`` the
doc set, ``agent_readiness`` the agent front door). This one grades a *process*: how
hard is it to develop a CUDA kernel in fak — to change ``cuda_kernels.cu`` /
``cuda_backend.h`` / ``cuda.go``, build it, prove it correct, and not have it silently
rot? That loop is unusually painful here for a structural reason: the canonical dev host
is win32 with **no CUDA toolkit and a walled GPU quota**, so the edit→build→validate loop
spans a remote GPU node (WSL / DGX / GCP), and a mistake in the cgo seam or the kernels is
caught late, by hand, on another machine. The number below is how much of that pain has
been engineered away.

It scores the git-tracked tree on fourteen mechanical KPIs across the five stages of the
loop, folds them into a weighted score + an A–F grade, and counts **process-debt**: the
total of concrete, re-derivable defects that make the CUDA loop slower or less safe. Each
is a defect you fix by *adding the affordance the loop is missing* — a local static gate, a
wired task-runner verb, an automatic CI compile check, a one-command witness aggregator, a
consolidated dev guide — never by writing prose.

  AUTHOR   — change a kernel without flying blind (local, GPU-free feedback)
    local_static_check     a one-command GPU-free local check exists AND is wired into the
                           task runner (cuda_abi_parity.py + a `make cuda-check` that runs
                           it + a scripts/ci.ps1 mirror for the no-toolchain Windows host)
    abi_parity             the header / kernels / cgo-binding ABI is in parity — computed
                           LIVE by the real tools/cuda_abi_parity.py parser over the
                           workspace, so a stub checker cannot earn this
    cpuref_parity_coverage every device op has a cpuref-parity witness test (the cosine-vs-
                           reference loop a kernel dev actually lives in); a composite op is
                           covered by the full-forward witness, a heavy op by its own test
  BUILD    — reproducible, one-command, portable across the host/arch matrix
    build_portable         the build scripts cover the host matrix (WSL · datacenter/DGX ·
                           GCP DLVM · native Windows) and the arch matrix (sm_80/89/90/100)
                           in an EXECUTABLE context, not just a comment
    toolchain_pinned       the CUDA toolchain is pinned (setup_cuda_wsl.sh) + the arch
                           override is documented
    task_runner            the cuda build/test/accept verbs are reachable from the Makefile
                           and DELEGATE to the real scripts (not named no-ops)
  VALIDATE — prove correctness; automate the proof
    witness_coverage       every recorded cosine/equality floor in cuda.go names an on-disk
                           acceptance witness (the recorded→measured honesty made checkable)
    witness_aggregator     ONE command runs every GPU witness and emits one verdict
                           (cuda_acceptance.sh), SKIP-is-not-PASS, covering each floor family
    floor_honesty          every recorded floor carries the "do not read a pass from this
                           value alone" caveat — the sentinel that keeps a green local build
                           from being mistaken for a passed GPU gate
  GATE     — catch a regression automatically, on every push
    cgo_typecheck_gate     an AUTOMATIC (push/PR, hosted, no-toolkit) job runs
                           `go vet -tags cuda` — the cheapest check of the cgo seam, which
                           the toolkit-free header was deliberately engineered to allow
    nvcc_compile_gate      an AUTOMATIC job nvcc-compiles the kernels and links
                           `go build -tags cuda` (against the -devel image's stub libs; no GPU)
    pure_go_guard          an AUTOMATIC gate asserts the DEFAULT build stays pure-Go (no cgo
                           leak) — today enforced only in the manual self-hosted job
  ONBOARD  — find the loop
    dev_guide              a single CUDA-dev guide doc that names the REAL on-disk commands
                           and scripts (a doc that names an absent command DROPS the score)
    entrypoint_surfaced    the guide is linked from AGENTS.md / EXTENDING.md and resolves

The headline metric is **process-debt**: the count of concrete HARD defects above. Driving
it down means a kernel dev gets a local gate in milliseconds, an automatic CI check on every
push, a one-command correctness proof, and a documented loop — instead of a multi-host round
trip to discover a typo. The companion process re-runs this, retires the worst-first defect
by adding the missing affordance, and re-runs to prove the drop (``--compare``).

Deterministic + read-only by construction: it reads the tree on disk (so a `git archive
HEAD` clone of a commit scores identically — the determinism witness in the test asserts
this) and edits nothing. Run from the repo ROOT::

    python tools/cuda_dev_scorecard.py                 # human scorecard
    python tools/cuda_dev_scorecard.py --json           # machine payload (corpus.process_debt)
    python tools/cuda_dev_scorecard.py --markdown        # the committed snapshot body
    python tools/cuda_dev_scorecard.py --compare base.json   # prove the 2x process-debt drop
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import cuda_abi_parity  # the real ABI parser — abi_parity runs IT, not a re-implementation

SCHEMA = "fak-cuda-dev-scorecard/1"
GENERATED_SNAPSHOT = "docs/CUDA-DEV-SCORECARD.md"

# ---- the CUDA seam + dev-process files the KPIs cross-check ----------------------------
HEADER = cuda_abi_parity.HEADER
KERNELS = cuda_abi_parity.KERNELS
BINDING = cuda_abi_parity.BINDING
BUILD_SH = "internal/compute/build_cuda.sh"
BUILD_PS1 = "tools/build_cuda_windows.ps1"
SETUP_SH = "internal/compute/setup_cuda_wsl.sh"
ABI_TOOL = "tools/cuda_abi_parity.py"
AGGREGATOR = "tools/cuda_acceptance.sh"
DEV_GUIDE = "docs/cuda-dev.md"
MAKEFILE = "Makefile"
CI_PS1 = "scripts/ci.ps1"
AGENTS = "AGENTS.md"
EXTENDING = "EXTENDING.md"
WORKFLOW_DIR = ".github/workflows"

# The device ops that have a cpuref counterpart, grouped into witness families. Each family
# is (label, member methods, name keywords a covering test must carry, forward_covered).
# A "forward_covered" family is exercised by the full multi-layer forward witness
# (TestCUDA…Forward…) rather than an isolated test — RMSNorm/RoPE/SwiGLU/Add are composed
# inside one forward pass and proven by its end-to-end cosine, so requiring an isolated test
# for each would false-fail a genuinely-covered op.
CUDA_OP_FAMILIES: list[tuple[str, list[str], list[str], bool]] = [
    ("MatMul (decode GEMV)", ["MatMul"], ["MatMul"], False),
    ("BatchedMatMul (prefill GEMM)", ["BatchedMatMul"], ["BatchedMatMul"], False),
    ("elementwise/norm (RMSNorm·RoPE·SwiGLU·Add)",
     ["RMSNorm", "RoPE", "SwiGLU", "AddInPlace", "AddBias"], ["RMSNorm", "RoPE", "SwiGLU"], True),
    ("Attention (flash)", ["Attention"], ["Attention", "Flash"], False),
    ("Argmax", ["Argmax"], ["Argmax"], False),
    ("GLM-DSA (sparse-attend + index-select)", ["DSASparseAttend", "DSAIndexSelect"],
     ["Dsa", "DSA"], False),
    ("AWQ (4-bit GEMV/GEMM)", ["AWQMatMul", "AWQBatchedMatMul"], ["AWQ", "Awq"], False),
]
_FORWARD_RE = re.compile(r"\bTestCUDA\w*Forward\w*", re.IGNORECASE)
_CUDA_TEST_RE = re.compile(r"\bfunc\s+(Test|Benchmark)(CUDA\w+)\s*\(")

# A recorded floor / equality gate constant in cuda.go.
_FLOOR_RE = re.compile(r"\b(cuda[A-Za-z0-9]*(?:CosineMin|Exact))\b")
_FLOOR_DECL_RE = re.compile(r"^\s*(?:const\s+)?(cuda[A-Za-z0-9]*(?:CosineMin|Exact))\s*(?:=|float|bool)")
# The honesty caveat (wording varies — "do not read a pass" / "… — or a speedup — from this/these value").
_CAVEAT_RE = re.compile(r"do not read a pass", re.IGNORECASE)
_SCRIPT_REF_RE = re.compile(r"tools/[A-Za-z0-9_./-]+\.sh")

GROUPS = ("author", "build", "validate", "gate", "onboard")
KPI_GROUP: dict[str, str] = {
    "local_static_check": "author", "abi_parity": "author", "cpuref_parity_coverage": "author",
    "build_portable": "build", "toolchain_pinned": "build", "task_runner": "build",
    "witness_coverage": "validate", "witness_aggregator": "validate", "floor_honesty": "validate",
    "cgo_typecheck_gate": "gate", "nvcc_compile_gate": "gate", "pure_go_guard": "gate",
    "dev_guide": "onboard", "entrypoint_surfaced": "onboard",
}
# Weights sum to exactly 1.0 (a regression test asserts both the sum and weight-set == KPI-set).
# Tilted toward VALIDATE + GATE (the daily loop + the automatic safety net), away from a doc.
KPI_WEIGHTS: dict[str, float] = {
    # author (0.24)
    "local_static_check": 0.09,
    "abi_parity": 0.09,
    "cpuref_parity_coverage": 0.06,
    # build (0.16)
    "build_portable": 0.08,
    "toolchain_pinned": 0.04,
    "task_runner": 0.04,
    # validate (0.30)
    "witness_coverage": 0.10,
    "witness_aggregator": 0.10,
    "floor_honesty": 0.10,
    # gate (0.24)
    "cgo_typecheck_gate": 0.10,
    "nvcc_compile_gate": 0.08,
    "pure_go_guard": 0.06,
    # onboard (0.06)
    "dev_guide": 0.04,
    "entrypoint_surfaced": 0.02,
}

_LINK_RE = re.compile(r"\[(?P<text>[^\]]+)\]\((?P<target>[^)]+)\)")


# ---------------------------------------------------------------------------
# Small pure helpers.
# ---------------------------------------------------------------------------

def _clamp(score: float) -> int:
    return int(max(0, min(100, round(score))))


def grade_letter(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def _has(text: str | None, *tokens: str) -> bool:
    if not text:
        return False
    low = text.lower()
    return any(t.lower() in low for t in tokens)


# ---------------------------------------------------------------------------
# Per-KPI pure checks. Each returns
#   {kpi, group, score (0-100 int), detail, defects: [str], soft: [str]}
# defects = HARD units of process-debt; soft = score-only nudges.
# ---------------------------------------------------------------------------

def kpi_local_static_check(tool_present: bool, make_wired: bool, ps1_wired: bool) -> dict[str, Any]:
    """The GPU-free local feedback loop: the ABI checker exists AND a one-command verb runs
    it (`make cuda-check`) AND the no-toolchain Windows host (scripts/ci.ps1) mirrors it.
    Each missing piece is one unit — without it a kernel dev edits the cgo seam blind until a
    remote GPU build, the exact friction this whole scorecard exists to retire."""
    defects: list[str] = []
    if not tool_present:
        defects.append(f"no {ABI_TOOL} — the GPU-free static ABI cross-check a dev runs locally")
    if not make_wired:
        defects.append(f"no `make cuda-check` target whose recipe runs {ABI_TOOL} --check "
                       "(the one-command local gate)")
    if not ps1_wired:
        defects.append(f"{CI_PS1} does not mirror cuda-check — the canonical Windows dev host "
                       "(CGO_ENABLED=0, no nvcc) gets no local cuda feedback")
    return {"kpi": "local_static_check", "group": "author",
            "score": _clamp(100 - 30 * len(defects)),
            "detail": (f"{len(defects)} missing local-check affordance(s)" if defects
                       else "ABI checker + make cuda-check + ci.ps1 mirror all present"),
            "defects": defects, "soft": []}


def kpi_abi_parity(parity_payload: dict[str, Any]) -> dict[str, Any]:
    """The header ↔ kernels ↔ cgo-binding ABI must agree. Computed LIVE by the real
    tools/cuda_abi_parity.py parser over the workspace (not re-implemented here), so a
    stub checker can't earn the points and a comment-only symbol can't false-pass. HARD per
    mismatch — each would break the nvcc link / cgo build on a GPU node."""
    corpus = parity_payload.get("corpus") or {}
    hard = corpus.get("hard") or []
    soft_n = corpus.get("soft_signals", 0)
    if parity_payload.get("verdict") == "AUDIT_ERROR":
        return {"kpi": "abi_parity", "group": "author", "score": 0,
                "detail": f"ABI parser could not read the seam: {parity_payload.get('reason')}",
                "defects": [f"CUDA seam unreadable: {parity_payload.get('reason')}"], "soft": []}
    return {"kpi": "abi_parity", "group": "author",
            "score": _clamp(100 - 25 * len(hard)),
            "detail": (f"{len(hard)} ABI mismatch(es)" if hard
                       else f"{corpus.get('n_declared', 0)} prototypes in full parity "
                            f"({soft_n} standby advisory)"),
            "defects": list(hard), "soft": []}


def kpi_cpuref_parity_coverage(uncovered: list[str]) -> dict[str, Any]:
    """Every device op family has a cpuref-parity witness test — the cosine-vs-reference
    loop a kernel dev lives in ('I changed the kernel, did I break the cosine?'). A family
    with no parity test is a silent gap: a regression there can't be caught against the
    Reference. ``uncovered`` is the list of family labels with no covering -tags cuda test."""
    defects = [f"device op family with no cpuref-parity witness test: {lbl} — a kernel change "
               "here can't be checked against the cpu Reference's cosine" for lbl in uncovered]
    covered = len(CUDA_OP_FAMILIES) - len(uncovered)
    return {"kpi": "cpuref_parity_coverage", "group": "author",
            "score": _clamp(100 * covered / max(1, len(CUDA_OP_FAMILIES))),
            "detail": f"{covered}/{len(CUDA_OP_FAMILIES)} device op families have a cpuref-parity witness",
            "defects": defects, "soft": []}


def kpi_build_portable(host_missing: list[str], arch_executable: bool,
                       arch_named: list[str]) -> dict[str, Any]:
    """The build must be reproducible across the host matrix (WSL · datacenter/DGX · GCP
    DLVM · native Windows) and the arch matrix (sm_80/89/90/100), with the arch in an
    EXECUTABLE context (the -arch="$ARCH" nvcc line + the FAK_CUDA_ARCH override), not just a
    comment. Each missing host platform is one unit; an arch list that never reaches nvcc is
    one unit (a paste-to-pass guard)."""
    defects = [f"build scripts don't cover the {h} host path" for h in host_missing]
    if not arch_executable:
        defects.append("the GPU arch is not in an executable context (no FAK_CUDA_ARCH override + "
                       '-arch="$ARCH" nvcc line) — the arch matrix is a comment, not a build path')
    soft = ([f"advertised arch not named in build_cuda.sh: {a}" for a in arch_named]
            if arch_named else [])
    return {"kpi": "build_portable", "group": "build",
            "score": _clamp(100 - 22 * len(defects) - 5 * len(soft)),
            "detail": (f"{len(defects)} portability gap(s)" if defects
                       else "host matrix (WSL · GPU server · cloud · native Windows) + executable arch override covered"),
            "defects": defects, "soft": soft}


def kpi_toolchain_pinned(version_pinned: bool, arch_override: bool) -> dict[str, Any]:
    """The CUDA toolchain version is pinned (setup_cuda_wsl.sh) and the arch override is
    documented — the reproducibility floor so 'works on my WSL 12.6' isn't a surprise on a
    datacenter image. Each missing piece is one unit."""
    defects: list[str] = []
    if not version_pinned:
        defects.append(f"no pinned CUDA toolchain version in {SETUP_SH} (e.g. cuda-nvcc=12.6) — "
                       "the WSL toolkit is unpinned, so two devs can build against different nvcc")
    if not arch_override:
        defects.append(f"no documented arch override (FAK_CUDA_ARCH) in {BUILD_SH}")
    return {"kpi": "toolchain_pinned", "group": "build",
            "score": _clamp(100 - 50 * len(defects)),
            "detail": ("CUDA version pinned + arch override documented" if not defects
                       else f"{len(defects)} reproducibility gap(s)"),
            "defects": defects, "soft": []}


def kpi_task_runner(missing_verbs: list[str]) -> dict[str, Any]:
    """The cuda build/test/accept verbs must be reachable from the Makefile and actually
    DELEGATE to the real scripts (cuda-build → build_cuda.sh, cuda-accept → cuda_acceptance.sh),
    not exist as named no-ops. Today the cuda build is reachable only by knowing the bare
    script path — no discoverable verb. ``missing_verbs`` is the list of absent/undelegating
    targets (collapsed to one unit: the runner has no cuda verbs)."""
    defects: list[str] = []
    if missing_verbs:
        defects.append("the Makefile exposes no cuda build/test/accept verb that delegates to the "
                       f"real scripts (missing/undelegating: {', '.join(missing_verbs)}) — a dev has "
                       "to know `bash internal/compute/build_cuda.sh` by heart")
    return {"kpi": "task_runner", "group": "build",
            "score": 100 if not defects else 30,
            "detail": ("cuda-build/cuda-test/cuda-accept delegate to the real scripts" if not defects
                       else f"{len(missing_verbs)} cuda verb(s) missing/undelegating"),
            "defects": defects, "soft": []}


def kpi_witness_coverage(floors_missing_script: list[str]) -> dict[str, Any]:
    """Every recorded cosine/equality floor in cuda.go must name an acceptance witness that
    exists on disk — the recorded→measured honesty made checkable, and a regression sentinel
    so a NEW floor added without a measurement harness fails. ``floors_missing_script`` lists
    floor constants whose named witness script is absent."""
    defects = [f"recorded floor {c} names a witness script that is not on disk — a floor with no "
               "measurement harness can never be proven on a GPU" for c in floors_missing_script]
    return {"kpi": "witness_coverage", "group": "validate",
            "score": _clamp(100 - 25 * len(defects)),
            "detail": (f"{len(defects)} floor(s) name a missing witness script" if defects
                       else "every recorded floor names an on-disk acceptance witness"),
            "defects": defects, "soft": []}


def kpi_witness_aggregator(present: bool, families_unreferenced: list[str]) -> dict[str, Any]:
    """ONE command (cuda_acceptance.sh) must run every GPU witness and emit one verdict, with
    SKIP-is-not-PASS, covering each floor family. Today 'is the GPU path green?' means running
    6 scripts by hand and eyeballing 6 logs. Absent = HARD; a family the aggregator forgets to
    run is one unit (a partial aggregator reads as full coverage)."""
    defects: list[str] = []
    if not present:
        defects.append(f"no {AGGREGATOR} — no single command runs every GPU witness with one "
                       "verdict; 'is the cuda path green?' is 6 manual scripts + 6 logs today")
    else:
        for fam in families_unreferenced:
            defects.append(f"{AGGREGATOR} does not run the {fam} witness — a partial aggregator "
                           "reads as full coverage")
    return {"kpi": "witness_aggregator", "group": "validate",
            "score": _clamp((0 if not present else 100) - 18 * len(families_unreferenced)),
            "detail": (f"{AGGREGATOR} runs every witness with one verdict" if present and not defects
                       else (f"{AGGREGATOR} missing" if not present
                             else f"{len(families_unreferenced)} witness family(ies) not aggregated")),
            "defects": defects, "soft": []}


def kpi_floor_honesty(floors_without_caveat: list[str]) -> dict[str, Any]:
    """Every recorded floor must carry the 'do not read a pass from this value alone' caveat —
    the discipline that keeps a green local build from being mistaken for a passed GPU gate.
    A regression sentinel: a NEW floor added without the caveat is the defect. ``floors_without
    _caveat`` lists offending constants."""
    defects = [f"recorded floor {c} has no 'do not read a pass from this value alone' caveat — "
               "the value could be mistaken for a measured pass" for c in floors_without_caveat]
    return {"kpi": "floor_honesty", "group": "validate",
            "score": _clamp(100 - 25 * len(defects)),
            "detail": (f"{len(defects)} floor(s) missing the recorded-not-measured caveat" if defects
                       else "every recorded floor carries the recorded-not-measured caveat"),
            "defects": defects, "soft": []}


def kpi_cgo_typecheck_gate(present: bool) -> dict[str, Any]:
    """The cheapest, highest-leverage automatic gate: a push/PR job on a HOSTED runner (no CUDA
    toolkit) that runs `go vet -tags cuda` — the toolkit-free header was deliberately engineered
    so a plain compiler type-checks the cgo seam. Today nothing in CI runs it, so a peer can
    break the header↔binding agreement and CI stays green until a manual GPU run. HARD if absent
    / only on workflow_dispatch / only self-hosted."""
    defects: list[str] = []
    if not present:
        defects.append("no automatic (push/PR, hosted-runner, no-toolkit) job runs "
                       "`go vet -tags cuda` — the cheapest check of the cgo seam is unguarded; "
                       "a header↔binding break is invisible until a manual GPU run")
    return {"kpi": "cgo_typecheck_gate", "group": "gate",
            "score": 100 if present else 0,
            "detail": ("`go vet -tags cuda` runs automatically on every push/PR" if present
                       else "no automatic cgo-typecheck gate"),
            "defects": defects, "soft": []}


def kpi_nvcc_compile_gate(present: bool) -> dict[str, Any]:
    """An automatic job that nvcc-compiles the kernels and links `go build -tags cuda` against
    the -devel image's cudart/cublas STUB libs (no GPU, no libcuda, no run step) — catches a
    kernel that compiles on WSL but not a datacenter toolchain (the documented uint8_t trap),
    and a cgo link break. HARD if no automatic job compiles the cuda variant."""
    defects: list[str] = []
    if not present:
        defects.append("no automatic job nvcc-compiles cuda_kernels.cu + links `go build -tags "
                       "cuda` — a kernel/binding break is caught only on a manual self-hosted GPU run")
    return {"kpi": "nvcc_compile_gate", "group": "gate",
            "score": 100 if present else 0,
            "detail": ("an automatic job compiles + links the cuda variant (no GPU)" if present
                       else "no automatic nvcc/link gate"),
            "defects": defects, "soft": []}


def kpi_pure_go_guard(present: bool) -> dict[str, Any]:
    """An AUTOMATIC gate asserting the DEFAULT build stays pure-Go (no cgo file leaks into the
    shipped binary) — the DIRECTION rule-1 invariant. Today it lives only inside the MANUAL
    self-hosted windows-cuda.yml job, so on push nothing asserts a peer didn't add a cgo file
    to the default build. HARD if not enforced by an automatic job."""
    defects: list[str] = []
    if not present:
        defects.append("the pure-Go (no-cgo-leak) guard runs only in the manual self-hosted job — "
                       "no automatic push/PR gate asserts the default build stays pure-Go")
    return {"kpi": "pure_go_guard", "group": "gate",
            "score": 100 if present else 20,
            "detail": ("an automatic gate asserts the default build is pure-Go" if present
                       else "pure-Go guard only in the manual job"),
            "defects": defects, "soft": []}


def kpi_dev_guide(present: bool, dead_refs: list[str], missing_sections: list[str]) -> dict[str, Any]:
    """A single CUDA-dev guide doc — but graded on CONTENT, not mere presence: it must name the
    REAL on-disk commands/scripts (a named command that doesn't exist DROPS the score) and cover
    the loop (setup → build → test → validate → add-a-kernel). Absent = HARD; each dead artifact
    reference or missing loop section is one unit."""
    defects: list[str] = []
    if not present:
        defects.append(f"no {DEV_GUIDE} — the consolidated CUDA dev loop lives scattered across "
                       "script headers + Go doc-comments; a new dev has no single entry point")
        return {"kpi": "dev_guide", "group": "onboard", "score": 0,
                "detail": f"no {DEV_GUIDE}", "defects": defects, "soft": []}
    for r in dead_refs:
        defects.append(f"{DEV_GUIDE} names an artifact that does not exist: {r} (doc-rot — a dev "
                       "who pastes it hits a missing-file error)")
    for s in missing_sections:
        defects.append(f"{DEV_GUIDE} is missing the '{s}' loop stage")
    return {"kpi": "dev_guide", "group": "onboard",
            "score": _clamp(100 - 18 * len(defects)),
            "detail": (f"{len(defects)} content gap(s) in the dev guide" if defects
                       else "dev guide present, names real artifacts, covers the loop"),
            "defects": defects, "soft": []}


def kpi_entrypoint_surfaced(linked: bool, target_resolves: bool) -> dict[str, Any]:
    """The dev guide must be linked from an orientation doc (AGENTS.md / EXTENDING.md) AND the
    link target must resolve — a dev finds the loop from the front door, not by spelunking. A
    dangling link earns nothing."""
    defects: list[str] = []
    if not linked:
        defects.append(f"{DEV_GUIDE} is not linked from {AGENTS} or {EXTENDING} — a dev can't find "
                       "the cuda loop from the orientation path")
    elif not target_resolves:
        defects.append(f"the cuda-dev link in {AGENTS}/{EXTENDING} is dangling ({DEV_GUIDE} absent)")
    return {"kpi": "entrypoint_surfaced", "group": "onboard",
            "score": 100 if (linked and target_resolves) else 30,
            "detail": ("the dev guide is linked from the orientation path and resolves" if linked and target_resolves
                       else "dev guide not surfaced from AGENTS.md / EXTENDING.md"),
            "defects": defects, "soft": []}


# ---------------------------------------------------------------------------
# Fold: KPIs -> composite score, grade, process-debt, control-pane payload.
# ---------------------------------------------------------------------------

def build_payload(*, workspace: str, kpis: list[dict[str, Any]],
                  error: str | None = None) -> dict[str, Any]:
    if error:
        return {"schema": SCHEMA, "ok": False, "verdict": "AUDIT_ERROR",
                "finding": "tooling_error", "reason": error,
                "next_action": "fix the read (run from repo ROOT), then re-run",
                "workspace": workspace, "corpus": {}, "kpis": []}
    by_name = {k["kpi"]: k for k in kpis}
    score = round(sum(KPI_WEIGHTS[n] * by_name[n]["score"] for n in KPI_WEIGHTS if n in by_name), 1)
    process_debt = sum(len(k["defects"]) for k in kpis)
    n_soft = sum(len(k["soft"]) for k in kpis)
    grade = grade_letter(score)

    debt_by_group = {g: 0 for g in GROUPS}
    score_by_group = {g: 0.0 for g in GROUPS}
    wsum_by_group = {g: 0.0 for g in GROUPS}
    for k in kpis:
        debt_by_group[k["group"]] += len(k["defects"])
        w = KPI_WEIGHTS.get(k["kpi"], 0.0)
        score_by_group[k["group"]] += w * k["score"]
        wsum_by_group[k["group"]] += w
    group_scores = {g: (round(score_by_group[g] / wsum_by_group[g], 1) if wsum_by_group[g] else 0.0)
                    for g in GROUPS}

    breakdown = sorted(
        ({"kpi": k["kpi"], "group": k["group"], "score": k["score"],
          "debt": len(k["defects"]), "detail": k["detail"]} for k in kpis),
        key=lambda x: (-x["debt"], x["score"]))

    corpus = {
        "score": score, "grade": grade, "process_debt": process_debt, "soft_signals": n_soft,
        "group_scores": group_scores, "debt_by_group": debt_by_group,
        "kpi_scores": {k["kpi"]: k["score"] for k in kpis},
        "debt_by_kpi": {k["kpi"]: len(k["defects"]) for k in kpis},
        "breakdown": breakdown,
    }

    gs = group_scores
    standing = (f"author {gs['author']:.0f} · build {gs['build']:.0f} · validate {gs['validate']:.0f} "
                f"· gate {gs['gate']:.0f} · onboard {gs['onboard']:.0f}")
    if process_debt == 0:
        ok, verdict, finding = True, "OK", "cuda_dev_ready"
        reason = (f"CUDA dev loop fully affordanced: score {score}/100 (grade {grade}), zero "
                  f"process-debt across {len(kpis)} KPIs ({standing}; {n_soft} advisory)")
        next_action = ("hold the line; re-run after a change to a CUDA seam file, a build script, "
                       "an acceptance witness, or a CI gate")
    else:
        ok, verdict, finding = False, "ACTION", "process_debt"
        worst = breakdown[0]
        reason = (f"{process_debt} unit(s) of CUDA dev-process-debt; score {score}/100 (grade {grade}); "
                  f"heaviest: {worst['kpi']} ({worst['debt']} defect(s)); standing {standing}")
        next_action = ("retire process-debt worst-first (see corpus.breakdown + per-KPI defects): "
                       "wire the local static gate, the cuda task-runner verbs, the automatic CI "
                       "compile gate, the witness aggregator, and the dev guide; re-run to prove the drop")

    return {"schema": SCHEMA, "ok": ok, "verdict": verdict, "finding": finding,
            "reason": reason, "next_action": next_action, "workspace": workspace,
            "corpus": corpus, "kpis": kpis}


# ---------------------------------------------------------------------------
# Disk gathering (the impure shell around the pure KPIs).
# ---------------------------------------------------------------------------

def repo_root(start: Path | None = None) -> Path:
    return (start or Path(__file__)).resolve().parent.parent


def _safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _floor_blocks(cuda_go: str) -> list[tuple[str, str]]:
    """Each recorded floor constant paired with the comment/context window above its
    declaration (≈40 lines), so witness_coverage and floor_honesty can read the script
    reference and the caveat that the doc-comment block carries — including a shared
    `const ( … )` block where two members sit under one comment."""
    lines = cuda_go.splitlines()
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for i, ln in enumerate(lines):
        m = _FLOOR_DECL_RE.match(ln)
        if not m:
            continue
        name = m.group(1)
        if name in seen:
            continue
        seen.add(name)
        window = "\n".join(lines[max(0, i - 40):i + 1])
        out.append((name, window))
    return out


def _strip_yaml_comments(text: str) -> str:
    """Drop whole-line YAML comments so a banner comment that *mentions* a token (e.g. a
    comment explaining that windows-cuda.yml is `self-hosted`, or describing the nvcc step)
    cannot satisfy — or wrongly exclude — a gate check. Only full-line `#` comments are
    removed; a `run:` step's shell body (where the real `nvcc` / `go vet -tags cuda` /
    `CgoFiles` tokens live) is untouched, so the cross-check sees CODE, not documentation."""
    return "\n".join("" if ln.lstrip().startswith("#") else ln for ln in text.splitlines())


def _auto_hosted_cuda_workflows(root: Path) -> list[str]:
    """The comment-stripped texts of every workflow that is AUTOMATICALLY triggered (push /
    pull_request) AND runs on a hosted (not self-hosted) runner — the only place an automatic
    gate can live. A workflow_dispatch-only or self-hosted-only file (windows-cuda.yml) is
    excluded: it never runs on a push, so it cannot guard the trunk. Comments are stripped
    first so a prose mention of 'self-hosted' (describing another workflow) doesn't exclude an
    actually-hosted job, and a prose mention of 'nvcc' can't fake a gate."""
    wf_dir = root / WORKFLOW_DIR
    out: list[str] = []
    if not wf_dir.is_dir():
        return out
    for f in sorted(wf_dir.glob("*.yml")) + sorted(wf_dir.glob("*.yaml")):
        text = _strip_yaml_comments(_safe_read(f))
        if not text.strip():
            continue
        auto = bool(re.search(r"(?m)^\s*pull_request:", text) or re.search(r"(?m)^\s*push:", text))
        self_hosted = "self-hosted" in text
        if auto and not self_hosted:
            out.append(text)
    return out


def gather(root: Path) -> list[dict[str, Any]]:
    root = root.resolve()

    def present(rel: str) -> bool:
        return (root / rel).exists()

    cuda_go = _safe_read(root / BINDING)
    build_sh = _safe_read(root / BUILD_SH)
    setup_sh = _safe_read(root / SETUP_SH)
    makefile = _safe_read(root / MAKEFILE)
    ci_ps1 = _safe_read(root / CI_PS1)
    agents = _safe_read(root / AGENTS)
    extending = _safe_read(root / EXTENDING)
    guide = _safe_read(root / DEV_GUIDE)

    # --- author ---
    tool_present = present(ABI_TOOL)
    make_cuda_check = bool(re.search(r"(?m)^cuda-check:", makefile)) and _has(makefile, "cuda_abi_parity")
    ps1_cuda_check = _has(ci_ps1, "cuda_abi_parity")
    # abi_parity: run the REAL parser over the workspace seam.
    parity_payload = cuda_abi_parity.collect(root)
    # cpuref parity coverage: collect TestCUDA* names from the //go:build cuda test files.
    test_names: set[str] = set()
    has_forward = False
    for sub in ("internal/compute", "internal/model"):
        d = root / sub
        if not d.is_dir():
            continue
        for tf in sorted(d.glob("*_test.go")):
            txt = _safe_read(tf)
            if "//go:build cuda" not in txt:
                continue
            for mm in _CUDA_TEST_RE.finditer(txt):
                test_names.add(mm.group(2))
            if _FORWARD_RE.search(txt):
                has_forward = True
    joined_tests = " ".join(test_names)
    uncovered: list[str] = []
    for label, _members, keywords, forward_covered in CUDA_OP_FAMILIES:
        covered = any(kw.lower() in joined_tests.lower() for kw in keywords)
        if not covered and forward_covered and has_forward:
            covered = True
        if not covered:
            uncovered.append(label)

    # --- build ---
    host_missing: list[str] = []
    if not _has(build_sh, "wsl"):
        host_missing.append("WSL")
    if not _has(build_sh, "/usr/local/cuda", "dgx", "datacenter"):
        host_missing.append("datacenter/DGX")
    if not _has(build_sh, "gcp", "dlvm", "deep-learning"):
        host_missing.append("GCP DLVM")
    if not present(BUILD_PS1):
        host_missing.append("native Windows")
    arch_executable = _has(build_sh, "FAK_CUDA_ARCH") and bool(re.search(r'-arch="?\$?\{?ARCH', build_sh))
    arch_named = [a for a in ("sm_80", "sm_89", "sm_90", "sm_100") if not _has(build_sh, a)]
    version_pinned = bool(re.search(r"cuda-nvcc\s*=\s*\d", setup_sh)) or bool(re.search(r"\b12\.\d\b", setup_sh)) and _has(setup_sh, "nvcc")
    arch_override = _has(build_sh, "FAK_CUDA_ARCH")
    # task_runner: targets that exist AND delegate to the real scripts.
    missing_verbs: list[str] = []
    for verb, script in (("cuda-build", "build_cuda.sh"), ("cuda-test", "build_cuda.sh"),
                         ("cuda-accept", "cuda_acceptance.sh")):
        body_ok = bool(re.search(rf"(?m)^{verb}:", makefile))
        # the recipe (the line[s] after the target) must reference the real script.
        if body_ok:
            block = makefile.split(f"\n{verb}:", 1)[-1].split("\n\n", 1)[0]
            body_ok = script in block
        if not body_ok:
            missing_verbs.append(verb)

    # --- validate ---
    floor_blocks = _floor_blocks(cuda_go)
    floors_missing_script: list[str] = []
    floors_without_caveat: list[str] = []
    for name, window in floor_blocks:
        scripts = _SCRIPT_REF_RE.findall(window)
        if not scripts or not any((root / s).exists() for s in scripts):
            floors_missing_script.append(name)
        if not _CAVEAT_RE.search(window):
            floors_without_caveat.append(name)
    aggregator_present = present(AGGREGATOR)
    agg_text = _safe_read(root / AGGREGATOR) if aggregator_present else ""
    families_unreferenced: list[str] = []
    if aggregator_present:
        for tag, needles in (("fp16 (#484)", ["run_484"]), ("quant (#485)", ["run_485"]),
                             ("flash-attn (#486)", ["run_486"]), ("async (#482)", ["run_482"]),
                             ("graph (#483)", ["run_483"]), ("GLM-DSA", ["dgx_glm_gpu_witness", "GLMMoeDsa", "glm_dsa"])):
            if not _has(agg_text, *needles):
                families_unreferenced.append(tag)

    # --- gate ---
    auto_wfs = _auto_hosted_cuda_workflows(root)
    cgo_typecheck = any(_has(t, "tags cuda") and _has(t, "vet") for t in auto_wfs)
    nvcc_gate = any(_has(t, "nvcc") and _has(t, "cuda_kernels.cu") and _has(t, "build", "-tags cuda")
                    for t in auto_wfs)
    pure_go = any(_has(t, "cgofiles") for t in auto_wfs)

    # --- onboard ---
    guide_present = present(DEV_GUIDE)
    dead_refs: list[str] = []
    missing_sections: list[str] = []
    if guide_present:
        for ref in set(re.findall(r"(?:tools/|internal/compute/|scripts/)[A-Za-z0-9_./-]+\.(?:sh|ps1|py|go)", guide)):
            if not (root / ref).exists():
                dead_refs.append(ref)
        for cmd in ("make cuda-check", "make cuda-accept"):
            if not _has(guide, cmd):
                dead_refs.append(f"command not named: {cmd}")
        for sec in ("setup", "build", "test", "validate", "add"):
            if not _has(guide, sec):
                missing_sections.append(sec)
    guide_linked = (_has(agents, DEV_GUIDE) or _has(extending, DEV_GUIDE) or
                    _has(agents, "cuda-dev") or _has(extending, "cuda-dev"))
    target_resolves = guide_present

    return [
        kpi_local_static_check(tool_present, make_cuda_check, ps1_cuda_check),
        kpi_abi_parity(parity_payload),
        kpi_cpuref_parity_coverage(uncovered),
        kpi_build_portable(host_missing, arch_executable, arch_named),
        kpi_toolchain_pinned(bool(version_pinned), arch_override),
        kpi_task_runner(missing_verbs),
        kpi_witness_coverage(floors_missing_script),
        kpi_witness_aggregator(aggregator_present, families_unreferenced),
        kpi_floor_honesty(floors_without_caveat),
        kpi_cgo_typecheck_gate(cgo_typecheck),
        kpi_nvcc_compile_gate(nvcc_gate),
        kpi_pure_go_guard(pure_go),
        kpi_dev_guide(guide_present, dead_refs, missing_sections),
        kpi_entrypoint_surfaced(guide_linked, target_resolves),
    ]


def collect(workspace: Path) -> dict[str, Any]:
    root = workspace.resolve()
    if not (root / BINDING).exists():
        return build_payload(workspace=str(root), kpis=[],
                             error=f"no {BINDING} under {root} — run from the fak repo ROOT")
    return build_payload(workspace=str(root), kpis=gather(root))


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def render(payload: dict[str, Any]) -> str:
    c = payload.get("corpus") or {}
    gs = c.get("group_scores") or {}
    lines = [
        f"cuda-dev-scorecard: {payload.get('verdict')} ({payload.get('finding')})",
        f"  {payload.get('reason')}",
        "",
        (f"score {c.get('score', 0)}/100 (grade {c.get('grade', '?')}) "
         f"· PROCESS-DEBT {c.get('process_debt', 0)} · {c.get('soft_signals', 0)} advisory"),
        ("dev loop:  " + "  ·  ".join(f"{g} {gs.get(g, 0):.0f}" for g in GROUPS)),
        ("debt by stage: " + "  ".join(
            f"{g}:{(c.get('debt_by_group') or {}).get(g, 0)}" for g in GROUPS)),
        "",
        "per-KPI (worst first):",
        f"  {'score':>5} {'debt':>4}  {'stage':<9} {'kpi':<24} detail",
    ]
    for b in c.get("breakdown", []):
        lines.append(f"  {b['score']:>5} {b['debt']:>4}  {b['group']:<9} {b['kpi']:<24} {b['detail']}")
    lines.append("")
    lines.append("process-debt work-list:")
    any_defect = False
    for k in sorted(payload.get("kpis", []), key=lambda x: -len(x["defects"])):
        if not k["defects"]:
            continue
        any_defect = True
        lines.append(f"  {k['kpi']} ({len(k['defects'])}):")
        for it in k["defects"][:12]:
            lines.append(f"      - {it}")
    if not any_defect:
        lines.append("  (none — zero process-debt)")
    lines.append("")
    lines.append(f"next: {payload.get('next_action')}")
    return "\n".join(lines)


def render_markdown(payload: dict[str, Any], *, stamp: str | None = None) -> str:
    c = payload.get("corpus") or {}
    gs = c.get("group_scores") or {}
    out: list[str] = []
    out.append("---")
    out.append('title: "fak CUDA-dev-process scorecard — the process-debt measuring stick"')
    out.append('description: "fak\'s deterministic CUDA-dev-process scorecard: KPIs across the five '
               'stages of the kernel development loop — author, build, validate, gate, onboard — '
               'folded into a composite score and the headline process-debt metric, re-derived from '
               'the git-tracked tree."')
    out.append("---")
    out.append("")
    out.append("# CUDA-dev-process scorecard — how hard is it to develop a kernel in fak")
    out.append("")
    if stamp:
        out.append(f"<!-- cuda-dev-scorecard: {stamp} · process: tools/cuda_dev_scorecard.py -->")
        out.append("")
    out.append("This grades the **CUDA development loop**: changing `cuda_kernels.cu` / "
               "`cuda_backend.h` / `cuda.go`, building it, proving it correct, and keeping it from "
               "rotting — a loop made unusually painful because the canonical dev host has no CUDA "
               "toolkit and a walled GPU, so the loop spans a remote GPU node. Every number is "
               "re-derived from the git-tracked tree by `tools/cuda_dev_scorecard.py` — no hand-entry. "
               "The headline metric is **process-debt**: the count of concrete, mechanical defects "
               "that make the loop slower or less safe — a missing local gate, no automatic CI compile "
               "check, no one-command witness, no dev guide.")
    out.append("")
    out.append("> Regenerate: `python tools/cuda_dev_scorecard.py --markdown --stamp DATE > docs/CUDA-DEV-SCORECARD.md`")
    out.append("")
    out.append("## Headline")
    out.append("")
    out.append("| Metric | Value |")
    out.append("|---|---|")
    out.append(f"| **Process-debt (total HARD defects)** | **{c.get('process_debt', 0)}** |")
    out.append(f"| Composite score | {c.get('score', 0)}/100 (grade {c.get('grade', '?')}) |")
    out.append("| Dev loop | " + " · ".join(f"{g} {gs.get(g, 0):.0f}" for g in GROUPS) + " |")
    out.append(f"| Advisory (soft) signals | {c.get('soft_signals', 0)} |")
    g = c.get("debt_by_group", {})
    out.append("| Debt by stage | " + " · ".join(f"{gg}:{g.get(gg, 0)}" for gg in GROUPS) + " |")
    out.append("")
    out.append("## The five stages of the kernel loop")
    out.append("")
    out.append(f"{len(payload.get('kpis', []))} KPIs, each 0–100, grouped by the loop stage they "
               "gate. `debt` = units of HARD process-debt.")
    out.append("")
    out.append("| Stage | KPI | Score | Debt | Detail |")
    out.append("|---|---|---:|:--:|---|")
    for b in c.get("breakdown", []):
        out.append(f"| {b['group']} | `{b['kpi']}` | {b['score']} | {b['debt']} | {b['detail']} |")
    out.append("")
    out.append("## Process-debt work-list")
    out.append("")
    any_defect = False
    for k in sorted(payload.get("kpis", []), key=lambda x: -len(x["defects"])):
        if not k["defects"]:
            continue
        any_defect = True
        out.append(f"### `{k['kpi']}` ({k['group']}) — {len(k['defects'])} defect(s), score {k['score']}")
        for it in k["defects"]:
            out.append(f"- {it}")
        out.append("")
    if not any_defect:
        out.append("No process-debt: the CUDA dev loop has a local gate, automatic CI coverage, a "
                   "one-command witness, and a documented path. 🎉")
        out.append("")
    return "\n".join(out)


def render_compare(baseline: dict[str, Any], current: dict[str, Any]) -> str:
    b = baseline.get("corpus") or {}
    cur = current.get("corpus") or {}
    bd, cd = b.get("process_debt", 0), cur.get("process_debt", 0)
    bo, co = b.get("score", 0), cur.get("score", 0)
    ratio = "∞ (zero)" if cd == 0 else f"{bd / cd:.1f}×"
    lines = [
        f"process-debt: {bd} -> {cd}   ({ratio} fewer defects)",
        f"score:        {bo}/100 -> {co}/100   (+{round(co - bo, 1)})",
    ]
    for gp in GROUPS:
        gb = (b.get("debt_by_group") or {}).get(gp, 0)
        gc = (cur.get("debt_by_group") or {}).get(gp, 0)
        lines.append(f"  {gp:<9} {gb} -> {gc}")
    target = max(0, bd // 2)
    if cd <= target:
        lines.append(f"VERDICT: >=2x process-debt reduction achieved ({bd} -> {cd}).")
    else:
        lines.append(f"VERDICT: not yet 2x — need process-debt <= {target} (now {cd}).")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CUDA-dev-process scorecard (read-only).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--markdown", action="store_true", help="emit the snapshot markdown body")
    ap.add_argument("--stamp", default="", help="date stamp for the markdown header")
    ap.add_argument("--compare", default="", metavar="BASELINE.json",
                    help="print the process-debt delta vs a prior baseline JSON")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass
    workspace = Path(args.workspace).resolve() if args.workspace else repo_root()
    payload = collect(workspace)

    if args.compare:
        try:
            baseline = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        except OSError as exc:
            print(f"error: cannot read baseline {args.compare}: {exc}", file=sys.stderr)
            return 2
        print(render_compare(baseline, payload))
    elif args.json:
        print(json.dumps(payload, indent=2))
    elif args.markdown:
        print(render_markdown(payload, stamp=args.stamp or None))
    else:
        print(render(payload))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
