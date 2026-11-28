#!/usr/bin/env python3
"""Core constants, binding context, contract overlays, lane picker, hold ledgers, and collision guards for issue_resolve_dispatch."""
from __future__ import annotations

import contextlib
import datetime as dt
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile  # noqa: F401
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import account_probe  # noqa: E402, F401
import dispatch_preflight  # noqa: E402
import dispatch_worker  # noqa: E402
import fleet_trend  # noqa: E402, F401
import issue_closure_audit  # noqa: E402, F401
import issue_dispatch  # noqa: E402
import issue_lane_router  # noqa: E402
import issue_worker_prompt  # noqa: E402, F401
import lane_core  # noqa: E402
import lane_yield  # noqa: E402
import tier_launch  # noqa: E402, F401
import worker_worktree  # noqa: E402, F401

_ACTIVE_IRD: list[Any] = []


@contextlib.contextmanager
def bind_ird(mod_ns: Any):
    _ACTIVE_IRD.append(mod_ns)
    try:
        yield mod_ns
    finally:
        _ACTIVE_IRD.pop()


def ird_attr(name: str, fallback: Any) -> Any:
    """Resolve an attribute through the active issue_resolve_dispatch facade when bound."""
    if _ACTIVE_IRD:
        return getattr(_ACTIVE_IRD[-1], name, fallback)
    mod = sys.modules.get("issue_resolve_dispatch")
    return getattr(mod, name, fallback) if mod is not None else fallback


no_window_creationflags = dispatch_worker.no_window_creationflags

SCHEMA = "fleet-issue-resolve-dispatch/1"
RUNS_DIRNAME = ".dispatch-runs"
WAVE_SIDECAR_SUFFIX = ".wave"
ACCOUNT_SIDECAR_SUFFIX = ".account"
LEASE_SIDECAR_SUFFIX = ".lease"
BASE_SHA_SIDECAR_SUFFIX = ".basesha"
WITNESS_SIDECAR_SUFFIX = ".witness"
LEASE_ID_PREFIX = "resolve-"
LEASE_TTL_MARGIN_S = 600
_LOG_ISSUE_RE = re.compile(r"resolve-(\d+)-")
_REPAIR_LOG_RE = re.compile(r"repair-(\d+)-")
_ANY_WORKER_LOG_RE = re.compile(r"(?:resolve|repair)-(\d+)-")
_SPAWN_LANE_RE = re.compile(r"\blane=(\S+)")
_WITNESS_STAMP_RE = re.compile(r"^resolve-(\d+)-(\d{8})-(\d{6})\.witness$")
_RID_RE = re.compile(r"^RID-[A-Z0-9]+$")
DEFAULT_WORKER_TIMEOUT_S = dispatch_worker.DEFAULT_TIMEOUT_S
DEFAULT_FIRST_EDIT_DEADLINE_S = 300
DEFAULT_SPAWN_PROBE_S = 5.0
DEFAULT_REALLOC_CEILING = 2
DEFAULT_SEAT_ADAPTIVE = True
DEFAULT_SEAT_CEILING = 20
DEFAULT_SEAT_RAMP_DELTA = 2
DEFAULT_SPAWN_BURST = 1
SPAWN_BURST_HARD_CEILING = 4
SPAWN_BURST_ENV = "FAK_DISPATCH_SPAWN_BURST"
DEFAULT_SPAWN_BURST_STAGGER_S = 15.0
SPAWN_BURST_STAGGER_ENV = "FAK_DISPATCH_SPAWN_BURST_STAGGER_S"
SPAWN_BURST_STAGGER_MAX_S = 120.0
LOOP_ID_PREFIX = "issue-resolve-dispatch"
DEFAULT_ISSUE_CONTRACT_MIN_SCORE = 100
DEFAULT_CONTRACT_SCAN = 8
DEFAULT_CONTRACT_HOLD_TTL_H = 24
_CONTRACT_HOLD_LEDGER = "contract-holds.jsonl"
DEFAULT_MULTI_LANE_HOLD_TTL_H = 24
_MULTI_LANE_HOLD_LEDGER = "multi-lane-holds.jsonl"
DEFAULT_COLLISION_HOLD_TTL_H = 3
_COLLISION_HOLD_LEDGER = "collision-holds.jsonl"
DEFAULT_REBLOCK_STREAK_HOLD_N = 2
DEFAULT_REBLOCK_STREAK_HOLD_TTL_H = 5
REBLOCK_STREAK_FIRST_LOSS_TTL_H = 7.0
REBLOCK_STREAK_MIN_TTL_MARGIN_H = 1.5
LOW_YIELD_MAX_EXCLUDES = 5
LOW_YIELD_MIN_SESSIONS = 2
_CONTRACT_FORCED_LEDGER = "contract-forced-bypasses.jsonl"
DEFAULT_REPAIR_BATCH = 5
DEFAULT_REPAIR_COOLDOWN_MIN = 360
_REPAIR_ATTEMPT_LEDGER = "repair-attempts.jsonl"
SAME_ISSUE_WIP_LOOKBACK_MIN = 24 * 60
SAME_ISSUE_WIP_ORIENTATION_DOCS = frozenset({
    "AGENTS.md", "CLAUDE.md", "README.md", "INDEX.md",
    "CONTRIBUTING.md", "llms.txt", "llms-full.txt",
})
ACTIVE_GUARD_LIVELOCK_MIN_COUNT = 10
ACTIVE_GUARD_LIVELOCK_ESCAPE_ROWS = 50
_AUDIT_LOG_RE = re.compile(r"audit log\s*:\s*(.+?\.jsonl)")
ACTIVE_COMPACT_RUNAWAY_MIN_COUNT = 3
ACTIVE_COMPACT_RUNAWAY_MIN_PAST_K = 20.0
ACTIVE_COMPACT_RUNAWAY_SHED_K = 5.0
_CTX_BUDGET_RE = re.compile(r"\bctx:(\d+(?:\.\d+)?)k/(\d+(?:\.\d+)?)k")
_DIST_PAST_COMPACT_RE = re.compile(r"\bdist:(\d+(?:\.\d+)?)k-past-compact")
_COMPACT_FIELD_RE = re.compile(r"\bcompact=(\S+)")
REPAIR_LANE = "contract-repair"
REPAIR_ISSUES_SIDECAR_SUFFIX = ".issues"
CONTRACT_OVERLAY_DIRNAME = "contract-overlays"

BACKENDS = ("claude", "opencode", "codex")
_BACKEND_PRODUCT = {"claude": "claude", "opencode": "opencode", "codex": "codex"}
OPENCODE_PROMPT_NOTICE = "Resolve GitHub issue # from the attached dispatch prompt."
OPENCODE_PROMPT_FILE_SUFFIX = ".prompt.txt"
CLAUDE_PROMPT_FILE_SUFFIX = ".prompt.txt"
CLAUDE_WORKER_MODEL = "claude-opus-5"
CLAUDE_WORKER_EFFORT = "xhigh"

REFUSAL_CLASS_LANE_LEASE = "lane_lease"
REFUSAL_CLASS_WORKTREE_COTENANCY = "worktree_cotenancy"
_REFUSAL_CLASSES = {
    "LANE_LEASE_HELD": REFUSAL_CLASS_LANE_LEASE,
    "SAME_ISSUE_WIP": REFUSAL_CLASS_WORKTREE_COTENANCY,
    "DIRTY_PATH_COLLISION": REFUSAL_CLASS_WORKTREE_COTENANCY,
}

ACCEPTANCE_GATE_HOLD_REASON = "acceptance-gate-unresolved"
_ACCEPTANCE_GATE_SECTION = re.compile(
    r"^[ \t]*#{1,6}[ \t]*acceptance[ \t]+gate[ \t\r]*\n"
    r"(.*?)(?=^[ \t]*#{1,6}[ \t]|\Z)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL)
_GATE_UNRESOLVED_TOKEN = re.compile(
    r"^(?:unknown|unresolved|tbd|to be determined|\?+)(?:\s*(?:$|[-–—:,;.]))")
_GATE_OPERATOR_INPUT = re.compile(
    r"\b(?:needs?|requires?|awaiting|pending|blocked on)\s+(?:an?\s+)?"
    r"operator\s+(?:input|decision|answer|review|sign-?off)\b")

_EDIT_EVENT_RE = re.compile(r"(?i)(apply_patch|write_file|edit_file|create_file|git\s+apply|set-content|add-content)")
_TEST_EVENT_RE = re.compile(r"(?i)(go\s+test|make\s+(?:ci|test)|test\.ps1|python\s+tools/\S+_test\.py|fak\s+validate)")
_COMMIT_EVENT_RE = re.compile(r"(?i)(fak\s+commit|git\s+commit)")
_PLANNING_EVENT_RE = re.compile(r"(?i)(update_plan|ultracode|workflow|capabilities|tools_search|web_search)")
_TOOL_EVENT_RE = re.compile(r"(?i)(tool_call|exec_command|apply_patch|write_file|update_plan|mcp__|functions\.)")
_TYPED_ESCALATION_RE = re.compile(r"(?i)ESCALATION_NEED:\s*(missing-path|missing-acceptance|missing-capability|unsafe-scope)")

DISPATCH_PRIORITY_WEIGHT = {"priority/P0": 1000, "priority/P1": 400, "priority/P2": 150}
_PREREQ_MARKER = re.compile(
    r"(?im)\b(?:depends[ -]?on|blocked[ -]?by)\b[:\s]*"
    r"((?:#\d+(?:\s*(?:,|and|&)\s*)?)+)")
_ISSUE_REF = re.compile(r"#(\d+)")

_DETACHED_PROCESS = 0x00000008
_CREATE_NO_WINDOW = 0x08000000
EARLY_EXIT_TAIL_CHARS = 8192

_CAP_BANNER_RE = re.compile(
    r"hit your[\w\s-]*limit|reached your[\w\s-]*limit"
    r"|usage limit reached|(?:weekly|session|\d+[\s-]?hour)\s+limit\s+reached"
    r"|limit\s+exhausted|rate[_ -]limited|HTTP\s+429",
    re.IGNORECASE)
_CODEX_RESET_RE = re.compile(
    r"try again at\s+([A-Za-z]{3,9}\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\s+\d{1,2}(?::\d{2})?\s*(?:am|pm))",
    re.IGNORECASE)
_RESET_AT_RE = re.compile(
    r"(?:will\s+)?reset\s+at\s+(\d{4}-\d{2}-\d{2}(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?(?:Z|[+-]\d{2}:?\d{2})?)",
    re.IGNORECASE)
_CAP_WEEKLY_RE = re.compile(
    r"weekly[\w/\s]*limit|limit[\w\s]*for the week|7[\s-]?day\s+limit", re.IGNORECASE)
_CAP_SESSION_RE = re.compile(r"session\s+limit", re.IGNORECASE)
_CAP_FIVE_HOUR_RE = re.compile(r"\b5[\s-]?hour(?:ly)?\s+limit\b", re.IGNORECASE)
_CAP_USAGE_RE = re.compile(
    r"hit your[\w\s-]*limit|reached your[\w\s-]*limit|usage limit"
    r"|limit\s+reached|limit\s+exhausted|quota", re.IGNORECASE)
_CAP_RESET_RE = re.compile(r"resets\s+(.+?)\s*\(America/Los_Angeles\)", re.IGNORECASE)
_WILL_RESET_LA_RE = re.compile(
    r"resets?\s+(?:at|on)\s+(.+?)\s*\(America/Los_Angeles\)", re.IGNORECASE)
_WILL_RESET_RE = re.compile(r"resets?\s+(?:at|on)\s+([^\r\n.;]+)", re.IGNORECASE)
_CAP_RESET_FALLBACK_RE = re.compile(r"resets\s+([^\r\n]+)", re.IGNORECASE)
_ANNOUNCED_WAIT_RE = re.compile(
    r"announced_wait\s*[=:≈~]?\s*(\d+h(?:\d+m)?(?:\d+s)?|\d+m(?:\d+s)?|\d+s)",
    re.IGNORECASE)
_ANNOUNCED_WAIT_DUR_RE = re.compile(
    r"^\s*(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?\s*$", re.IGNORECASE)
_SESSION_HOLD_MAX_MIN = 90
_ROLLING_HOLD_MAX_MIN = 5 * 60 + 15
_WEEKLY_FALLBACK_MIN = 6 * 60
_MONTHS = {m: i for i, m in enumerate(
    ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"),
    start=1)}
_WEEKDAYS = {d: i for i, d in enumerate(
    ("mon", "tue", "wed", "thu", "fri", "sat", "sun"))}
_PT_TO_UTC = dt.timedelta(hours=7)
_CAP_TAIL_BYTES = 4096
_AUTH_GAP_RE = re.compile(
    r"API_KEY is empty"
    r"|no ChatGPT subscription auth\.json"
    r"|no auth\.json was resolved"
    r"|Run `(?:codex|claude) login`",
    re.IGNORECASE)

_STUB_LOG_MAX_BYTES = 512
_BACKEND_DEAD_STREAK = 3
_BACKEND_REPROBE_MIN = 30
_BACKEND_AUTHGAP_REPROBE_MIN = 360

LEASE_REFUSED_RC = 3
LEASE_TIMEOUT_RC = 124
LEASE_DRAIN_TIMEOUT_S = 5
_LEASE_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]{0,199}$")
_LEASE_SESSION_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")
_LANE_TREE_CACHE: dict[str, dict[str, list[str]]] = {}

_SAME_ISSUE_WIP_RE = re.compile(
    r"\b(uncommitted|working[- ]tree|working tree changes|left .* working[- ]tree|"
    r"worktree changes|left .* worktree|not committed|without a commit|"
    r"no commit|commit blocked)\b",
    re.IGNORECASE,
)
_SAME_ISSUE_WIP_CLAIMS = frozenset({"CLAIM_NO_COMMIT", "CLAIM_UNWITNESSED"})

CLAIM_WITNESSED = "CLAIM_WITNESSED"
CLAIM_UNWITNESSED = "CLAIM_UNWITNESSED"
CLAIM_NO_COMMIT = "CLAIM_NO_COMMIT"
_WITNESS_OK = "diff-witnessed"

NO_COMMIT_SELF_MODIFY = "self_modify"
NO_COMMIT_POLICY_BLOCK = "policy_block"
NO_COMMIT_AUTH_WALL = "auth_wall"
NO_COMMIT_OFF_TRUNK = "off_trunk"
NO_COMMIT_BANNER_NOOP = "banner_noop"
NO_COMMIT_PREVIEW_CONFIRM = "preview_confirm_feedback"
NO_COMMIT_MISSING_LOG = "missing_log_artifact"
NO_COMMIT_RESTART_EXHAUSTED = "restart_exhausted"
NO_COMMIT_UNKNOWN = "unknown"
NO_COMMIT_CLEAN_EXIT = "clean_exit_no_commit"
NO_COMMIT_DIED_BEFORE_EPILOGUE = "died_before_epilogue"
NO_COMMIT_GUARD_SPAWN_FAILED = "guard_child_spawn_failed"

_GUARD_EPILOGUE_MARKER = "── guard · "
_GUARD_CHILD_SPAWN_FAILED_MARKER = "fak guard: could not run"
_NOOP_BANNER_RE = re.compile(r">\s*build\s*[·:]", re.IGNORECASE)
_GLM_WALL_RE = re.compile(r"Limit Exhausted|limit will reset at|usage limit reached",
                          re.IGNORECASE)
_MISSING_API_KEY_RE = re.compile(
    r"Missing environment variable:\s*`?[A-Z0-9_]*API_KEY`?",
    re.IGNORECASE)
_PREVIEW_CONFIRM_FEEDBACK_RE = re.compile(
    r"(REQUIRE_WITNESS\s*/\s*ESCALATE|preview-confirm|_fak_confirm)",
    re.IGNORECASE)
_RESTART_EXHAUSTED_TAIL_BYTES = 16 * 1024
_RESTART_EXHAUSTED_RE = re.compile(
    r"managed-context status (?P<status>restart_exhausted|reset_limit) "
    r"(?:limit=(?P<limit>\d+) )?(?:count=(?P<count>\d+) )?"
    r"(?:reason|dominant_cause)=(?P<cause>[^\s]+)", re.IGNORECASE)

SPAWN_CAUSE_WEEKLY_LIMIT = "weekly_limit"
SPAWN_CAUSE_STALE_CRED = "stale_cred"
SPAWN_CAUSE_CHILD_CRASH = "child_crash"
SPAWN_CAUSE_EXEC_RACE = "exec_race"
SPAWN_CAUSE_UNKNOWN = "unknown"
SPAWN_FAILED_CAUSES = (
    SPAWN_CAUSE_WEEKLY_LIMIT, SPAWN_CAUSE_STALE_CRED, SPAWN_CAUSE_CHILD_CRASH,
    SPAWN_CAUSE_EXEC_RACE, SPAWN_CAUSE_UNKNOWN,
)
_SPAWN_CRASH_RE = re.compile(
    r"Traceback \(most recent call last\)"
    r"|panic:|goroutine \d+ \[running\]"
    r"|Segmentation fault|SIGSEGV|signal:\s*(?:killed|segmentation|abort)"
    r"|core dumped|std::bad_alloc|std::terminate"
    r"|out of memory|OOMKilled|Cannot allocate memory"
    r"|fatal error:"
    r"|exec format error|executable file not found|fork/exec",
    re.IGNORECASE)

_HOLD_NO_COMMIT_REASONS = frozenset({
    NO_COMMIT_SELF_MODIFY,
    NO_COMMIT_POLICY_BLOCK,
    NO_COMMIT_PREVIEW_CONFIRM,
})
_ISSUE_TOKEN_RE = re.compile(r"(?<![\w-])#(\d+)\b")
OPEN_WITNESSED = "OPEN_WITNESSED"
_HEALTH_SKIP_STUB_RATE = 0.5
_GUARD_PROBE_LEDGER = "guard-probe.json"
DEFAULT_GUARD_PROBE_TTL_H = 24
_GUARD_PROBE_TIMEOUT_S = 20.0
_GUARD_PROBE_MIN_FLAGS = 40
_GUARD_PROBE_PEELED = frozenset({"core-lock-all", "all", "help", "h"})
_GUARD_FLAG_DEF_RE = re.compile(r"^  -([A-Za-z][A-Za-z0-9_-]*)", re.M)

BENIGN_ACTIONS = frozenset({
    "spawned", "would_spawn",
    "repair_spawned", "would_repair",
    "repair_in_flight",
    "no_issue", "no_lane", "lane_busy", "lane_leased",
    "same_issue_wip",
    "dirty_path_collision",
    "active_guard_livelock",
    "active_compact_runaway",
    "multi_lane_scope",
    "issue_contract_hold",
    "force_reason_required",
    "refused",
    "weekly_capped", "backend_unhealthy",
    "backend_health_skip",
    "seat_cooled",
})
SPAWN_FAILED_RED_STREAK = 3
SEAT_STREAK_REPROBE_MIN = 30


def is_orientation_doc(path: str) -> bool:
    """True for a repo-root orientation doc every worker prompt names by rote."""
    return normalize_repo_path(path) in SAME_ISSUE_WIP_ORIENTATION_DOCS


def worker_model_effort(backend: str, acct: dict) -> tuple[str | None, str | None]:
    """The (model, reasoning-effort) to pin on one worker's CLI, per backend."""
    if backend == "opencode":
        return acct.get("model"), None
    if backend == "claude":
        if str(acct.get("model") or "") == "local":
            return "local", None
        return CLAUDE_WORKER_MODEL, str(acct.get("model_effort") or CLAUDE_WORKER_EFFORT)
    return None, None


def launch_gate_for_guard(guarded: bool, backend: str) -> dict[str, Any]:
    """Machine-readable launch readiness for an already-priced worker command."""
    if guarded:
        return {"ready": True, "guarded": True, "blockers": []}
    next_action = "make-fak-guard-resolvable"
    if backend != "claude":
        next_action = "make-fak-guard-resolvable-and-set-guard-base-url"
    return {
        "ready": False,
        "guarded": False,
        "blockers": [{
            "code": "UNGUARDED_WORKER",
            "reason": "worker command is spawnable but would run without fak guard",
            "next_action": next_action,
        }],
    }


def refusal_class(verdict: str) -> str:
    """Contention class for a refusal verdict; '' when the verdict is neither."""
    return _REFUSAL_CLASSES.get(str(verdict or "").strip().upper(), "")


def launch_gate_blocked(code: str, reason: str, next_action: str) -> dict[str, Any]:
    """Machine-readable launch hold for a tick that never reached command pricing."""
    blocker: dict[str, Any] = {
        "code": code,
        "reason": reason,
        "next_action": next_action,
    }
    contention = refusal_class(code)
    if contention:
        blocker["refusal_class"] = contention
    return {"ready": False, "blockers": [blocker]}


def win_creationflags(exe: str) -> int:
    """Return the no-console flag for one unattended Windows worker."""
    del exe
    os_mod = ird_attr("os", os)
    if os_mod.name != "nt":
        return 0
    return _DETACHED_PROCESS


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _py() -> str:
    return sys.executable or "python"


def _fak_command_prefix(root: Path) -> list[str]:
    """Argv prefix for invoking ``fak``, freshest candidate first (#5856)."""
    os_mod = ird_attr("os", os)
    dw = ird_attr("dispatch_worker", dispatch_worker)
    explicit = os_mod.environ.get("FAK_BIN")
    if explicit:
        return [explicit]
    candidates: list[str] = []
    onpath = dw._which_on_exact_path("fak", os_mod.environ.get("PATH"))
    if onpath:
        candidates.append(onpath)
    for name in ("fak.exe", "fak"):
        cand = root / name
        if cand.is_file():
            candidates.append(str(cand))
            break
    if not candidates:
        return ["go", "run", "./cmd/fak"]
    return [max(candidates, key=dw.fak_binary_build_time)]


def _fak_dev_command_prefix(root: Path) -> list[str]:
    """Argv prefix for invoking the development-only ``fak-dev`` binary."""
    os_mod = ird_attr("os", os)
    dw = ird_attr("dispatch_worker", dispatch_worker)
    explicit = (os_mod.environ.get("FAK_DEV_BIN") or "").strip()
    compatible = (os_mod.environ.get("FAK_BIN") or "").strip()
    if not explicit and compatible and Path(compatible).stem.lower() == "fak-dev":
        explicit = compatible
    if explicit:
        return [explicit]
    candidates: list[str] = []
    onpath = dw._which_on_exact_path("fak-dev", os_mod.environ.get("PATH"))
    if onpath:
        candidates.append(onpath)
    for name in ("fak-dev.exe", "fak-dev"):
        cand = root / name
        if cand.is_file():
            candidates.append(str(cand))
            break
    if not candidates:
        return ["go", "run", "./cmd/fak-dev"]
    return [max(candidates, key=dw.fak_binary_build_time)]


def split_command_env(value: str) -> list[str]:
    """Split a command-valued env var without eating Windows path separators."""
    return shlex.split(value, posix=(ird_attr("os", os).name != "nt"))


def fak_loop_cmd(root: Path) -> list[str]:
    """Resolve the durable fak binary for optional loop-ledger appends."""
    os_mod = ird_attr("os", os)
    shutil_mod = ird_attr("shutil", shutil)
    configured = os_mod.environ.get("FAK_BIN")
    if configured:
        return split_command_env(configured)
    on_path = shutil_mod.which("fak")
    if on_path:
        return [on_path]
    in_tree = root / ("fak.exe" if os_mod.name == "nt" else "fak")
    return [str(in_tree)] if in_tree.exists() else []


def contract_overlay_path(runs_dir: Path, number: int) -> Path:
    return runs_dir / CONTRACT_OVERLAY_DIRNAME / f"issue-{int(number)}.md"


def read_contract_overlay(runs_dir: Path, number: int | None) -> str:
    """The local contract backfill for one issue, '' when absent/unreadable."""
    if not number:
        return ""
    try:
        return contract_overlay_path(runs_dir, int(number)).read_text(
            encoding="utf-8", errors="replace").strip()
    except (OSError, TypeError, ValueError):
        return ""


def contract_overlay_times(runs_dir: Path) -> dict[int, float]:
    """``{issue: overlay mtime}`` for every local backfill on disk."""
    out: dict[int, float] = {}
    overlay_dir = runs_dir / CONTRACT_OVERLAY_DIRNAME
    if not overlay_dir.is_dir():
        return out
    for f in overlay_dir.glob("issue-*.md"):
        try:
            out[int(f.stem.split("-", 1)[1])] = f.stat().st_mtime
        except (OSError, ValueError, IndexError):
            continue
    return out


def _issue_record_for_contract(issue: dict[str, Any] | None,
                               number: int | None,
                               overlay: str = "") -> dict[str, Any]:
    issue = issue if isinstance(issue, dict) else {}
    labels = []
    for lab in issue.get("labels") or []:
        if isinstance(lab, dict):
            name = str(lab.get("name") or "").strip()
        else:
            name = str(lab or "").strip()
        if name:
            labels.append({"name": name})
    try:
        num = int(issue.get("number") or number or 0)
    except (TypeError, ValueError):
        num = int(number or 0)
    body = str(issue.get("body") or "")
    if overlay:
        runs_name = ird_attr("RUNS_DIRNAME", RUNS_DIRNAME)
        body = (body + "\n\n<!-- local contract overlay "
                f"({runs_name}/{CONTRACT_OVERLAY_DIRNAME}) -->\n" + overlay)
    return {
        "number": num,
        "title": str(issue.get("title") or f"issue #{num}").strip(),
        "body": body,
        "labels": labels,
    }


def unresolved_acceptance_gate(body: Any) -> str:
    """The verbatim acceptance-gate line that marks a contract as not executable yet."""
    text = body if isinstance(body, str) else ""
    section = _ACCEPTANCE_GATE_SECTION.search(text)
    if not section:
        return ""
    line = next((s for s in (ln.strip() for ln in section.group(1).splitlines()) if s), "")
    if not line:
        return ""
    probe = line.lstrip("-*+> \t").strip().strip("`*_").strip().lower()
    if _GATE_UNRESOLVED_TOKEN.match(probe) or _GATE_OPERATOR_INPUT.search(probe):
        return line[:200]
    return ""


def issue_contract_hold_reason(contract: dict[str, Any]) -> str:
    if contract.get("unavailable"):
        return str(contract.get("reason") or "issue contract unavailable")
    gate_hold = str(contract.get("acceptance_gate_hold") or "")
    if gate_hold:
        return f"{ACCEPTANCE_GATE_HOLD_REASON}: {gate_hold}"
    review = contract.get("review") if isinstance(contract.get("review"), dict) else {}
    parts = [str(r) for r in (review.get("reasons") or []) if r]
    parts.extend(f"missing:{m}" for m in (review.get("missing_fields") or []) if m)
    score = int(contract.get("score") or 0)
    min_score = ird_attr("DEFAULT_ISSUE_CONTRACT_MIN_SCORE", DEFAULT_ISSUE_CONTRACT_MIN_SCORE)
    if contract.get("ok") and score >= min_score:
        return "issue contract passed"
    if score < min_score:
        parts.append(f"score:{score}<floor:{min_score}")
    return ", ".join(parts) if parts else "issue contract below spawn floor"


def _contract_missing_fields(contract: dict[str, Any]) -> list[str]:
    review = contract.get("review") if isinstance(contract.get("review"), dict) else {}
    return [str(m) for m in (review.get("missing_fields") or []) if m]


def _git_capture(root: Path, args: list[str], *, timeout: int = 30) -> tuple[int, str]:
    """Run one ``git`` subcommand under ``root``; return ``(rc, stdout)``."""
    os_mod = ird_attr("os", os)
    sub_mod = ird_attr("subprocess", subprocess)
    nowin = ird_attr("no_window_creationflags", no_window_creationflags)
    kwargs: dict[str, Any] = {
        "cwd": str(root), "capture_output": True, "text": True,
        "encoding": "utf-8", "errors": "replace", "timeout": timeout,
    }
    if os_mod.name == "nt":
        kwargs["creationflags"] = nowin()
    try:
        proc = sub_mod.run(["git", *args], **kwargs)
    except (OSError, subprocess.SubprocessError):
        return 127, ""
    return proc.returncode, proc.stdout or ""


def _spawn_header_lane(log: Path) -> str | None:
    """Parse ``lane=<L>`` from a worker log's ``# fak-spawn`` header."""
    try:
        with open(log, "r", encoding="utf-8", errors="replace") as fh:
            head = fh.readline()
    except OSError:
        return None
    if not head.startswith("# fak-spawn"):
        return None
    m = _SPAWN_LANE_RE.search(head)
    return m.group(1) if m else None


def low_yield_soft_excludes(
    root: Path,
    runs_dir: Path,
    *,
    lane_trees: dict[str, list[str]] | None = None,
    now_ts: float | None = None,
    max_excludes: int = LOW_YIELD_MAX_EXCLUDES,
    min_sessions: int = LOW_YIELD_MIN_SESSIONS,
) -> dict[str, Any]:
    """Fold the recent resolve corpus into the set of lanes to SOFT-exclude (#2062)."""
    import time
    ly_mod = ird_attr("lane_yield", lane_yield)
    dp_mod = ird_attr("dispatch_preflight", dispatch_preflight)
    result: dict[str, Any] = {"exclude": set(), "lanes": [], "flagged": []}
    try:
        if not runs_dir.is_dir():
            return result
        now = time.time() if now_ts is None else now_ts
        since_iso = time.strftime(
            "%Y-%m-%dT%H:%M:%S +0000",
            time.gmtime(now - ly_mod._LOW_YIELD_LOOKBACK_MIN * 60))

        def _closes(_lane: str, tree: list[str]) -> int | None:
            return ly_mod.count_lane_ancestry_closes(
                root, tree, since_iso=since_iso)

        def _finished(log: Path) -> bool:
            return not dp_mod.resolve_sidecar_pid_is_live(
                log.with_suffix(".pid"))

        fold = ly_mod.low_yield_lanes(
            runs_dir, closes_counter=_closes, lane_trees=lane_trees or {},
            now_ts=now, include_log=_finished)
        flagged = [
            r for r in fold.get("lanes", [])
            if r.get("verdict") == "LOW_YIELD"
            and int(r.get("sessions", 0)) >= min_sessions
            and r.get("tree_known")
        ]
        flagged.sort(key=lambda r: (-int(r.get("turns", 0)), str(r.get("lane"))))
        chosen = flagged[: max(0, max_excludes)]
        result["exclude"] = {str(r["lane"]) for r in chosen}
        result["lanes"] = [
            {"lane": str(r["lane"]), "turns": int(r.get("turns", 0)),
             "sessions": int(r.get("sessions", 0)), "closes": r.get("closes"),
             "evidence_logs": list(r.get("evidence_logs", []))}
            for r in chosen
        ]
        result["flagged"] = [str(r["lane"]) for r in flagged]
    except Exception:
        return {"exclude": set(), "lanes": [], "flagged": []}
    return result


def lane_issue_numbers(root: Path, explicit_lane: str | None,
                       exclude: set[str] | None = None,
                       guarded: bool | None = None,
                       soft_exclude: set[str] | None = None) -> dict[str, Any]:
    """Pick the lane (busiest, or explicit) and return its OPEN issue numbers, oldest first."""
    id_mod = ird_attr("issue_dispatch", issue_dispatch)
    dw_mod = ird_attr("dispatch_worker", dispatch_worker)
    lc_mod = ird_attr("lane_core", lane_core)
    router = id_mod.run_json(
        [_py(), str(root / "tools" / "issue_lane_router.py"), "--json"],
        root, timeout=130)
    lanes = router.get("lanes") or {}
    guarded = dw_mod.guard_enabled() if guarded is None else guarded
    nums_by_lane: dict[str, list[int]] = {}
    trees: dict[str, Any] = {}
    for ln, info in lanes.items():
        iss = info.get("issues") if isinstance(info, dict) else info
        trees[ln] = info.get("tree") if isinstance(info, dict) else None
        nums: list[int] = []
        for it in (iss or []):
            n = it.get("number") if isinstance(it, dict) else it
            try:
                nums.append(int(n))
            except (TypeError, ValueError):
                continue
        nums_by_lane[ln] = sorted(nums)
    caps_by_issue: dict[int, list[str]] = {}
    for it in (router.get("issues") or []):
        if not isinstance(it, dict):
            continue
        try:
            num = int(it.get("number"))
        except (TypeError, ValueError):
            continue
        caps_by_issue[num] = [str(c).lower() for c in (it.get("required_caps") or []) if c]
    exclude = exclude or set()
    soft_exclude = soft_exclude or set()
    low_yield_excluded: list[str] = []
    low_yield_relief: list[str] = []
    core_lanes_out: list[str] = []
    lane_priority: dict[str, int] = {}
    safe_lanes_busy: list[str] = []
    by_lane_count = {k: len(v) for k, v in nums_by_lane.items()}
    if explicit_lane:
        chosen = explicit_lane
        held: list[str] = []
        eligible_ranked = [[chosen, nums_by_lane.get(chosen, [])]]
    else:
        held_set = ({ln for ln in nums_by_lane
                     if not lc_mod.lane_dispatchable_under_guard(True, trees.get(ln))}
                    if guarded else set())
        held = sorted(ln for ln in nums_by_lane if ln in held_set and nums_by_lane[ln])
        safe_open = {k: v for k, v in nums_by_lane.items() if k not in held_set and v}
        safe_lanes_busy = sorted(k for k in safe_open if k in exclude)
        eligible = {k: v for k, v in nums_by_lane.items()
                    if k not in exclude and k not in held_set}
        soft_hits = sorted(k for k in soft_exclude if k in eligible)
        if soft_hits:
            survivors = {k: v for k, v in eligible.items() if k not in soft_exclude}
            if survivors:
                eligible = survivors
                low_yield_excluded = soft_hits
            else:
                low_yield_relief = soft_hits
        core_lanes = {ln for ln in eligible
                      if lc_mod.is_core_source_lane_tree(trees.get(ln))}
        lane_priority = id_mod.lane_priority_scores(
            root, {ln: list(v) for ln, v in eligible.items()})
        core_lanes_out = sorted(core_lanes)

        def _value_rank(kv: tuple[str, list[int]]) -> tuple[int, int, int, str]:
            ln, v = kv
            return (-(1 if ln in core_lanes else 0),
                    -int(lane_priority.get(ln, 0)),
                    -len(v),
                    ln)

        eligible_ranked = [[k, v] for k, v in
                           sorted(eligible.items(), key=_value_rank)
                           if v]
        chosen = eligible_ranked[0][0] if eligible_ranked else None
    return {"lane": chosen, "numbers": nums_by_lane.get(chosen or "", []),
            "by_lane_count": by_lane_count,
            "eligible_by_lane": eligible_ranked,
            "excluded_lanes": sorted(exclude),
            "safe_lanes_busy": safe_lanes_busy if not explicit_lane else [],
            "self_modify_held": held,
            "core_lanes": core_lanes_out,
            "lane_priority": lane_priority,
            "caps_by_issue": caps_by_issue,
            "low_yield_excluded": low_yield_excluded,
            "low_yield_relief": low_yield_relief,
            "router_error": router.get("_error")}


def pick_target_issue(numbers: list[int], skip: set[int]) -> int | None:
    """The first lane issue not in ``skip`` (live workers union recently-attempted)."""
    for n in numbers:
        if n not in skip:
            return n
    return None


def candidate_priority(labels: Any) -> int:
    """The dispatchorder Candidate.priority integer an issue earns from its labels."""
    best = 0
    for lab in labels or []:
        name = lab.get("name") if isinstance(lab, dict) else lab
        weight = DISPATCH_PRIORITY_WEIGHT.get(str(name or "").strip(), 0)
        if weight > best:
            best = weight
    return best


def candidate_blocked_by(body: Any) -> list[str]:
    """The dispatchorder Candidate.blocked_by list parsed from 'depends-on:/blocked-by: #N'."""
    text = body if isinstance(body, str) else ""
    out: list[str] = []
    seen: set[str] = set()
    for marker in _PREREQ_MARKER.finditer(text):
        for num in _ISSUE_REF.findall(marker.group(1)):
            if num not in seen:
                seen.add(num)
                out.append(num)
    return out


def contract_scan_stream(eligible_by_lane: list[Any] | None,
                         skip: set[int]) -> list[tuple[str, int]]:
    """The cross-lane candidate stream the bounded contract scan walks."""
    queues: list[tuple[str, list[int]]] = []
    for entry in eligible_by_lane or []:
        try:
            lane, numbers = entry[0], entry[1]
        except (TypeError, IndexError, KeyError):
            continue
        nums = [n for n in (numbers or []) if n not in skip]
        if lane and nums:
            queues.append((str(lane), nums))
    stream: list[tuple[str, int]] = []
    depth = 0
    while any(depth < len(nums) for _, nums in queues):
        for lane, nums in queues:
            if depth < len(nums):
                stream.append((lane, nums[depth]))
        depth += 1
    return stream


def contract_held_issues(runs_dir: Path, *, ttl_h: int = DEFAULT_CONTRACT_HOLD_TTL_H,
                         now_ts: float | None = None,
                         updated_ts: dict[int, float] | None = None) -> set[int]:
    """Issue numbers whose contract review HELD within the last ``ttl_h`` hours."""
    return {int(r.get("issue") or 0) for r in contract_held_records(
        runs_dir, ttl_h=ttl_h, now_ts=now_ts, updated_ts=updated_ts)}


def _missing_fields_from_hold_reason(reason: str) -> list[str]:
    out: list[str] = []
    for token in reason.split(","):
        token = token.strip()
        if token.startswith("missing:"):
            field = token.split(":", 1)[1].strip()
            if field:
                out.append(field)
    return out


def contract_held_records(
    runs_dir: Path, *, ttl_h: int = DEFAULT_CONTRACT_HOLD_TTL_H,
    now_ts: float | None = None,
    updated_ts: dict[int, float] | None = None,
    only: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Recent contract-held ledger rows, deduped to the latest verdict per issue."""
    if ttl_h <= 0:
        return []
    ledger = runs_dir / _CONTRACT_HOLD_LEDGER
    if not ledger.is_file():
        return []
    import time
    now = now_ts if now_ts is not None else time.time()
    horizon = now - ttl_h * 3600
    held: dict[int, tuple[float, int, dict[str, Any]]] = {}
    try:
        lines = ledger.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    only_set = {int(n) for n in only} if only is not None else None
    for idx, line in enumerate(lines):
        try:
            row = json.loads(line)
            ts = float(row.get("ts") or 0)
            if ts >= horizon:
                n = int(row.get("issue"))
                if only_set is not None and n not in only_set:
                    continue
                if updated_ts and (updated_ts.get(n) or 0.0) > ts:
                    continue
                reason = str(row.get("reason") or "")
                missing = row.get("missing_fields")
                if not isinstance(missing, list):
                    missing = _missing_fields_from_hold_reason(reason)
                rec = {
                    "issue": n,
                    "number": n,
                    "score": int(row.get("score") or 0),
                    "reason": reason,
                    "title": str(row.get("title") or "")[:200],
                    "missing_fields": [str(m) for m in missing if m],
                }
                prev = held.get(n)
                if prev is None or ts > prev[0] or (ts == prev[0] and idx > prev[1]):
                    held[n] = (ts, idx, rec)
        except (TypeError, ValueError):
            continue
    return [rec for _ts, _idx, rec in sorted(held.values(),
                                            key=lambda item: (item[0], item[1]))]


def record_contract_holds(runs_dir: Path, rows: list[dict[str, Any]], *,
                          live: bool, now_ts: float | None = None) -> None:
    """Append this tick's contract-HELD issues to the skip ledger."""
    if not live or not rows:
        return
    import time
    now = now_ts if now_ts is not None else time.time()
    iso = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ledger = runs_dir / _CONTRACT_HOLD_LEDGER
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
        with ledger.open("a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({
                    "utc": iso, "ts": now, "issue": int(r.get("issue") or 0),
                    "score": int(r.get("score") or 0),
                    "reason": str(r.get("reason") or "")[:200],
                    "title": str(r.get("title") or "")[:200],
                    "missing_fields": [str(m) for m in
                                       (r.get("missing_fields") or []) if m][:24],
                }, ensure_ascii=False) + "\n")
    except OSError:
        pass


def record_contract_forced_bypass(runs_dir: Path, row: dict[str, Any], *,
                                  live: bool, now_ts: float | None = None) -> None:
    """Append one operator-forced contract-gate bypass to the audit ledger (#2637)."""
    if not live or not row:
        return
    import time
    now = now_ts if now_ts is not None else time.time()
    iso = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ledger = runs_dir / _CONTRACT_FORCED_LEDGER
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
        with ledger.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "utc": iso, "ts": now,
                "issue": int(row.get("issue") or 0),
                "lane": str(row.get("lane") or "")[:80],
                "score": int(row.get("score") or 0),
                "reason": str(row.get("reason") or "")[:300],
                "gate_reason": str(row.get("gate_reason") or "")[:300],
                "title": str(row.get("title") or "")[:200],
            }, ensure_ascii=False) + "\n")
    except OSError:
        pass


def contract_forced_bypass_count(runs_dir: Path) -> int:
    """Total operator-forced contract-gate bypasses recorded so far (#2637)."""
    ledger = runs_dir / _CONTRACT_FORCED_LEDGER
    if not ledger.is_file():
        return 0
    try:
        return sum(1 for line in
                   ledger.read_text(encoding="utf-8", errors="replace").splitlines()
                   if line.strip())
    except OSError:
        return 0


def multi_lane_held_records(
    runs_dir: Path,
    *,
    ttl_h: int = DEFAULT_MULTI_LANE_HOLD_TTL_H,
    now_ts: float | None = None,
    updated_ts: dict[int, float] | None = None,
    only: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Recent MULTI_LANE_SCOPE rows, deduped to the latest verdict per issue."""
    if ttl_h <= 0:
        return []
    ledger = runs_dir / _MULTI_LANE_HOLD_LEDGER
    if not ledger.is_file():
        return []
    import time
    now = now_ts if now_ts is not None else time.time()
    horizon = now - ttl_h * 3600
    held: dict[int, tuple[float, int, dict[str, Any]]] = {}
    try:
        lines = ledger.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    only_set = {int(n) for n in only} if only is not None else None
    for idx, line in enumerate(lines):
        try:
            row = json.loads(line)
            ts = float(row.get("ts") or 0)
            if ts < horizon:
                continue
            n = int(row.get("issue"))
            if only_set is not None and n not in only_set:
                continue
            if updated_ts and (updated_ts.get(n) or 0.0) > ts:
                continue
            uncovered_lanes = row.get("uncovered_lanes") or []
            rec = {
                "issue": n,
                "number": n,
                "lane": str(row.get("lane") or ""),
                "title": str(row.get("title") or "")[:200],
                "reason": str(row.get("reason") or "")[:240],
                "uncovered_lanes": [str(ln) for ln in uncovered_lanes if ln][:24],
            }
            prev = held.get(n)
            if prev is None or ts > prev[0] or (ts == prev[0] and idx > prev[1]):
                held[n] = (ts, idx, rec)
        except (TypeError, ValueError):
            continue
    return [rec for _ts, _idx, rec in sorted(held.values(),
                                            key=lambda item: (item[0], item[1]))]


def multi_lane_held_issues(
    runs_dir: Path,
    *,
    ttl_h: int = DEFAULT_MULTI_LANE_HOLD_TTL_H,
    now_ts: float | None = None,
    updated_ts: dict[int, float] | None = None,
) -> set[int]:
    """Issue numbers held for operator split/reroute after MULTI_LANE_SCOPE."""
    return {int(r.get("issue") or 0) for r in multi_lane_held_records(
        runs_dir, ttl_h=ttl_h, now_ts=now_ts, updated_ts=updated_ts)}


def record_multi_lane_holds(runs_dir: Path, rows: list[dict[str, Any]], *,
                            live: bool, now_ts: float | None = None) -> None:
    """Append live MULTI_LANE_SCOPE holds so the next tick advances."""
    if not live or not rows:
        return
    import time
    now = now_ts if now_ts is not None else time.time()
    iso = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ledger = runs_dir / _MULTI_LANE_HOLD_LEDGER
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
        with ledger.open("a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({
                    "utc": iso,
                    "ts": now,
                    "issue": int(r.get("issue") or 0),
                    "lane": str(r.get("lane") or "")[:80],
                    "title": str(r.get("title") or "")[:200],
                    "reason": str(r.get("reason") or "")[:240],
                    "uncovered_lanes": [str(ln) for ln in
                                        (r.get("uncovered_lanes") or []) if ln][:24],
                }, ensure_ascii=False) + "\n")
    except OSError:
        pass


def collision_held_records(
    runs_dir: Path,
    *,
    ttl_h: int = DEFAULT_COLLISION_HOLD_TTL_H,
    now_ts: float | None = None,
    updated_ts: dict[int, float] | None = None,
    only: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Recent DIRTY_PATH_COLLISION / SAME_ISSUE_WIP holds, deduped to latest per issue."""
    if ttl_h <= 0:
        return []
    ledger = runs_dir / _COLLISION_HOLD_LEDGER
    if not ledger.is_file():
        return []
    import time
    now = now_ts if now_ts is not None else time.time()
    horizon = now - ttl_h * 3600
    held: dict[int, tuple[float, int, dict[str, Any]]] = {}
    try:
        lines = ledger.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    only_set = {int(n) for n in only} if only is not None else None
    for idx, line in enumerate(lines):
        try:
            row = json.loads(line)
            ts = float(row.get("ts") or 0)
            if ts < horizon:
                continue
            n = int(row.get("issue"))
            if only_set is not None and n not in only_set:
                continue
            kind = str(row.get("kind") or "")
            _local_collision = kind in ("dirty_path", "same_issue_wip")
            if (updated_ts and not _local_collision
                    and (updated_ts.get(n) or 0.0) > ts):
                continue
            rec = {
                "issue": n,
                "number": n,
                "kind": kind,
                "refusal_class": str(row.get("refusal_class")
                                     or REFUSAL_CLASS_WORKTREE_COTENANCY),
                "lane": str(row.get("lane") or ""),
                "title": str(row.get("title") or "")[:200],
                "reason": str(row.get("reason") or "")[:240],
                "dirty_paths": [str(p) for p in (row.get("dirty_paths") or []) if p][:24],
            }
            prev = held.get(n)
            if prev is None or ts > prev[0] or (ts == prev[0] and idx > prev[1]):
                held[n] = (ts, idx, rec)
        except (TypeError, ValueError):
            continue
    return [rec for _ts, _idx, rec in sorted(held.values(),
                                            key=lambda item: (item[0], item[1]))]


def collision_held_issues(
    runs_dir: Path,
    *,
    ttl_h: int = DEFAULT_COLLISION_HOLD_TTL_H,
    now_ts: float | None = None,
    updated_ts: dict[int, float] | None = None,
) -> set[int]:
    """Issue numbers held after a dirty-path / same-issue-WIP collision this window."""
    return {int(r.get("issue") or 0) for r in collision_held_records(
        runs_dir, ttl_h=ttl_h, now_ts=now_ts, updated_ts=updated_ts)}


def record_collision_holds(runs_dir: Path, rows: list[dict[str, Any]], *,
                           live: bool, now_ts: float | None = None) -> None:
    """Append live dirty-path / same-issue-WIP collision holds."""
    if not live or not rows:
        return
    import time
    now = now_ts if now_ts is not None else time.time()
    iso = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ledger = runs_dir / _COLLISION_HOLD_LEDGER
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
        with ledger.open("a", encoding="utf-8") as f:
            for r in rows:
                issue = int(r.get("issue") or 0)
                if issue <= 0:
                    continue
                f.write(json.dumps({
                    "utc": iso,
                    "ts": now,
                    "issue": issue,
                    "kind": str(r.get("kind") or "")[:40],
                    "refusal_class": REFUSAL_CLASS_WORKTREE_COTENANCY,
                    "lane": str(r.get("lane") or "")[:80],
                    "title": str(r.get("title") or "")[:200],
                    "reason": str(r.get("reason") or "")[:240],
                    "dirty_paths": [str(p) for p in
                                    (r.get("dirty_paths") or []) if p][:24],
                }, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _parse_gh_timestamp(value: Any) -> float | None:
    """Epoch seconds from a gh ISO-8601 timestamp."""
    s = str(value or "").strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return dt.datetime.fromisoformat(s).timestamp()
    except ValueError:
        return None


def open_issue_updated_map(root: Path, *, cap: int = 1000,
                           runner: Any | None = None) -> dict[int, float]:
    """``{number: updatedAt-epoch}`` for open issues -- ONE bulk ``gh issue list`` call."""
    run = runner or ird_attr("subprocess", subprocess).run
    nowin = ird_attr("no_window_creationflags", no_window_creationflags)
    try:
        proc = run(
            ["gh", "issue", "list", "--state", "open", "--limit", str(cap),
             "--json", "number,updatedAt"],
            cwd=str(root), capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=60, creationflags=nowin())
        rows = json.loads(proc.stdout or "[]") if proc.returncode == 0 else []
    except (OSError, subprocess.SubprocessError, ValueError):
        return {}
    out: dict[int, float] = {}
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        try:
            n = int(row.get("number"))
        except (TypeError, ValueError):
            continue
        ts = _parse_gh_timestamp(row.get("updatedAt"))
        if ts is not None:
            out[n] = ts
    return out


def lane_tree(root: Path, lane: str) -> list[str]:
    """The repo-relative file-tree globs the fenced lease should cover for `lane`."""
    cache = ird_attr("_LANE_TREE_CACHE", _LANE_TREE_CACHE)
    key = str(root)
    trees = cache.get(key)
    if trees is None:
        trees = {}
        try:
            ilr = ird_attr("issue_lane_router", issue_lane_router)
            _concurrent, trees, _exclusive = ilr.lane_taxonomy(root)
        except Exception:  # noqa: BLE001
            trees = {}
        cache[key] = trees
    globs = trees.get(lane)
    if globs:
        return list(globs)
    return [f"internal/{lane}/**"]


def multi_lane_scope(text: str, chosen_lane: str, chosen_tree: list[str],
                     trees: dict[str, list[str]], lane_set: set[str]) -> dict[str, Any]:
    """PURE: file families ``text`` names that ``chosen_lane``'s lease can't cover."""
    ilr = ird_attr("issue_lane_router", issue_lane_router)
    lease_view = {"__lease__": list(chosen_tree or [])}
    covered: list[str] = []
    uncovered: list[dict[str, Any]] = []
    for p in ilr.named_repo_paths(text):
        if ilr.path_matches_lane(p, lease_view):
            covered.append(p)
            continue
        other = sorted(ln for ln in ilr.path_matches_lane(p, trees)
                       if ln in lane_set and ln != chosen_lane)
        if other:
            uncovered.append({"path": p, "lanes": other})
    uncovered_lanes = sorted({ln for u in uncovered for ln in u["lanes"]})
    return {"multi_lane": bool(uncovered), "chosen_lane": chosen_lane,
            "chosen_tree": list(chosen_tree or []), "covered_paths": covered,
            "uncovered": uncovered, "uncovered_lanes": uncovered_lanes}


def scan_multi_lane_scope(root: Path, text: str, chosen_lane: str) -> dict[str, Any]:
    """Resolve the taxonomy and run :func:`multi_lane_scope` for a live dispatch."""
    try:
        ilr = ird_attr("issue_lane_router", issue_lane_router)
        concurrent, trees, _exclusive = ilr.lane_taxonomy(root)
    except Exception as exc:  # noqa: BLE001
        return {"multi_lane": False, "unavailable": True, "reason": str(exc)}
    if not trees:
        return {"multi_lane": False, "unavailable": True,
                "reason": "dos doctor returned no lane trees"}
    lt_fn = ird_attr("lane_tree", lane_tree)
    scan = multi_lane_scope(text, chosen_lane, lt_fn(root, chosen_lane),
                            trees, set(concurrent) or set(trees))
    scan["unavailable"] = False
    return scan


def multi_lane_scope_reason(issue: int, scan: dict[str, Any]) -> str:
    """One actionable line naming the uncovered families + the split/override path."""
    fam = ", ".join(f"{u['path']} ({'/'.join(u['lanes'])})"
                    for u in scan.get("uncovered") or [])
    return (f"issue #{issue} names file families outside lane "
            f"'{scan.get('chosen_lane')}' lease tree {scan.get('chosen_tree')}: {fam} "
            f"— the lease is the collision authority, and a narrow "
            f"'{scan.get('chosen_lane')}' lease would NOT protect "
            f"{scan.get('uncovered_lanes')} while the worker edits them. Split into "
            f"per-lane child issues, or re-dispatch under a lane whose lease covers "
            f"every named family (or --force to accept the under-scoped lease)")


def parse_git_status_paths(status: str) -> list[str]:
    """Repo-relative paths from ``git status --porcelain=v1`` output."""
    out: list[str] = []
    for raw in (status or "").splitlines():
        if len(raw) < 4:
            continue
        spec = raw[3:].strip()
        parts = spec.split(" -> ", 1) if " -> " in spec else [spec]
        for p in parts:
            p = normalize_repo_path(p)
            if p and p not in out:
                out.append(p)
    return out


def normalize_repo_path(path: str) -> str:
    p = str(path or "").strip().strip('"').replace("\\", "/")
    if p.startswith("./"):
        p = p[2:]
    return p


def dirty_repo_paths(root: Path) -> dict[str, Any]:
    git_fn = ird_attr("_git_capture", _git_capture)
    rc, out = git_fn(root, [
        "-c", "core.quotepath=false",
        "status", "--porcelain=v1", "--untracked-files=all",
    ], timeout=15)
    if rc != 0:
        return {"paths": [], "unavailable": True, "rc": rc}
    return {"paths": parse_git_status_paths(out), "unavailable": False}


def _path_text_variants(path: str) -> list[str]:
    p = normalize_repo_path(path)
    variants = [p] if p else []
    if p.startswith("fak/"):
        variants.append(p[len("fak/"):])
    elif p.startswith(("internal/", "cmd/", "experiments/")):
        variants.append("fak/" + p)
    return list(dict.fromkeys(variants))


def text_mentions_repo_path(text: str, path: str) -> bool:
    hay = (text or "").replace("\\", "/")
    if not any(ch.isalnum() for ch in normalize_repo_path(path)):
        return False
    for variant in _path_text_variants(path):
        if re.search(r"(?<![\w./-])" + re.escape(variant) + r"(?![\w./-])", hay):
            return True
    return False


def dirty_path_collision(text: str, dirty_paths: list[str]) -> dict[str, Any]:
    matches: list[str] = []
    for p in dirty_paths or []:
        norm = normalize_repo_path(p)
        if norm and text_mentions_repo_path(text, norm) and norm not in matches:
            matches.append(norm)
    return {"collides": bool(matches), "dirty_paths": matches}


def dirty_path_collision_reason(issue: int, scan: dict[str, Any]) -> str:
    paths = ", ".join(str(p) for p in (scan.get("dirty_paths") or [])[:8])
    more = ""
    if len(scan.get("dirty_paths") or []) > 8:
        more = f" (+{len(scan.get('dirty_paths') or []) - 8} more)"
    return (f"issue #{issue} names dirty local path(s) already modified in this "
            f"checkout: {paths}{more} — refusing DIRTY_PATH_COLLISION so a new "
            f"worker cannot overwrite peer WIP; wait for those paths to commit/"
            f"clear, or dispatch a disjoint issue")


def _read_log_tail(path: Path, *, max_bytes: int = 128 * 1024) -> str:
    try:
        with open(path, "rb") as fh:
            try:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                fh.seek(max(0, size - max_bytes), os.SEEK_SET)
            except OSError:
                pass
            return fh.read(max_bytes).decode("utf-8", errors="replace")
    except OSError:
        return ""


def same_issue_wip_collision(
    runs_dir: Path,
    issue: int | None,
    dirty_paths: list[str],
    *,
    lookback_min: int = SAME_ISSUE_WIP_LOOKBACK_MIN,
    now_ts: float | None = None,
) -> dict[str, Any]:
    """Detect same-issue uncommitted WIP left by a previous finished resolver."""
    if issue is None or lookback_min <= 0 or not runs_dir.is_dir() or not dirty_paths:
        return {"collides": False, "dirty_paths": [], "orientation_paths": []}
    import time
    now = now_ts if now_ts is not None else time.time()
    horizon = now - lookback_min * 60
    matches: list[dict[str, Any]] = []
    orientation: list[str] = []
    for log in sorted(runs_dir.glob(f"resolve-{int(issue)}-*.log"),
                      key=lambda p: p.stat().st_mtime if p.exists() else 0,
                      reverse=True):
        try:
            mtime = log.stat().st_mtime
        except OSError:
            continue
        if mtime < horizon:
            continue
        text = _read_log_tail(log)
        witness_claim = ""
        witness_path = log.with_suffix(WITNESS_SIDECAR_SUFFIX)
        if witness_path.exists():
            try:
                doc = json.loads(witness_path.read_text(encoding="utf-8"))
                witness_claim = str(doc.get("claim") or "")
            except (OSError, ValueError, AttributeError):
                witness_claim = ""
        if not (_SAME_ISSUE_WIP_RE.search(text)
                or witness_claim in _SAME_ISSUE_WIP_CLAIMS):
            continue
        dirty = []
        for p in dirty_paths:
            norm = normalize_repo_path(p)
            if not norm or not text_mentions_repo_path(text, norm):
                continue
            if is_orientation_doc(norm):
                if norm not in orientation:
                    orientation.append(norm)
                continue
            if norm not in dirty:
                dirty.append(norm)
        if dirty:
            matches.append({
                "log": log.name,
                "path": str(log),
                "mtime": mtime,
                "witness_claim": witness_claim or None,
                "dirty_paths": dirty,
            })
    if not matches:
        return {"collides": False, "dirty_paths": [],
                "orientation_paths": orientation}
    paths: list[str] = []
    for match in matches:
        for p in match.get("dirty_paths") or []:
            if p not in paths:
                paths.append(p)
    return {
        "collides": True,
        "issue": int(issue),
        "dirty_paths": paths,
        "orientation_paths": orientation,
        "evidence": matches[:3],
    }


def same_issue_wip_reason(issue: int, scan: dict[str, Any]) -> str:
    paths = ", ".join(str(p) for p in (scan.get("dirty_paths") or [])[:8])
    more = ""
    if len(scan.get("dirty_paths") or []) > 8:
        more = f" (+{len(scan.get('dirty_paths') or []) - 8} more)"
    evidence = scan.get("evidence") or []
    first = evidence[0] if evidence and isinstance(evidence[0], dict) else {}
    log = first.get("log") if first else None
    claim = first.get("witness_claim") if first else None
    source = f" in {log}" if log else ""
    if claim:
        source += f" ({claim})"
    return (f"issue #{issue} has recent same-issue uncommitted WIP{source} "
            f"naming dirty local path(s): {paths}{more} — refusing "
            f"SAME_ISSUE_WIP so a second resolver cannot stack onto unfinished "
            f"work; continue/commit those paths first, or dispatch a disjoint issue")


def _candidate_issue_numbers(eligible_lanes: list[Any] | None,
                             fallback: list[int] | None = None) -> set[int]:
    out: set[int] = set()
    for entry in eligible_lanes or []:
        try:
            nums = entry[1] or []
        except (TypeError, IndexError, KeyError):
            continue
        for n in nums:
            try:
                out.add(int(n))
            except (TypeError, ValueError):
                continue
    if not out:
        for n in fallback or []:
            try:
                out.add(int(n))
            except (TypeError, ValueError):
                continue
    return out


def capability_skips(candidates: Any, caps_by_issue: dict[int, list[str]],
                     node_caps: Any) -> tuple[set[int], list[dict[str, Any]]]:
    """Return candidates whose explicit requirements exceed this host's caps."""
    available = {str(cap).lower() for cap in node_caps if cap}
    required = {
        int(number): {str(cap).lower() for cap in caps_by_issue.get(int(number), []) if cap}
        for number in candidates
    }
    skipped = {number for number, caps in required.items() if not caps <= available}
    rows = [
        {"issue": number, "required_caps": sorted(required[number])}
        for number in sorted(skipped)
    ]
    return skipped, rows
