#!/usr/bin/env python3
r"""Launch account-scoped Claude Code agent chat sessions.

This is the manual front door for the fleet account switcher. The existing
resume/watchdog tools already know which ``.claude*`` accounts are real workers
and which are blocked by auth or usage limits; this helper uses the same source
of truth to start a new Claude Code chat under the right ``CLAUDE_CONFIG_DIR``.

GLM / Z.ai parity (``--glm``)
----------------------------
Claude Code speaks the Anthropic Messages API, not GLM's OpenAI-compatible wire.
``--glm`` closes that gap so a GLM/Z.ai (or any OpenAI-compatible) model launches
*through Claude Code* at parity with a native Claude account: this launcher starts
a local ``fak serve`` gateway in front of the chosen GLM backend (the kernel
translates Anthropic ``/v1/messages`` -> OpenAI ``/v1/chat/completions`` and
adjudicates every tool call), then wires Claude Code's ``ANTHROPIC_BASE_URL`` +
model-tier env to that loopback gateway under the selected account's
``CLAUDE_CONFIG_DIR``. The gateway is a managed child: it is health-checked before
the session starts and torn down when it ends. The backend is configurable — the
Z.ai Coding-Plan endpoint is the default; ``--glm-base-url`` / ``--glm-model``
point it at a local or DGX-served GLM (SGLang/vLLM) instead. This mirrors the
proven wiring in ``fak/scripts/dogfood-claude.sh``.

Examples:
  python tools/claude_agent_chat.py list
  python tools/claude_agent_chat.py plan --goal "finish the release notes"
  python tools/claude_agent_chat.py launch --goal "hard tasks default to tier 1"
  python tools/claude_agent_chat.py launch -t1 --goal "hard implementation task"
  python tools/claude_agent_chat.py launch --work-kind gardening --goal "tidy the index"
  python tools/claude_agent_chat.py launch --tier auto --prompt "say pong"
  python tools/claude_agent_chat.py launch --account gem7 --goal "audit docs"
  python tools/claude_agent_chat.py launch --glm --goal "implement the feature"
  python tools/claude_agent_chat.py plan  --glm --json   # inspect the GLM wiring
  python tools/claude_agent_chat.py launch --glm \
      --glm-base-url http://dgx:8000/v1 --glm-model glm-5.2 --goal "run on the DGX"
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fleet_accounts  # noqa: E402
import fleet_version  # noqa: E402


SCHEMA = "claude-agent-chat/1"
DEFAULT_AGENT_NAME = "fleet-agent"
DEFAULT_AGENT_DESCRIPTION = "Agentic Claude Code session for the fleet workspace."
DEFAULT_AGENT_PROMPT = (
    "You are an agentic Claude Code session in the fleet workspace. "
    "Work from current files and command output, preserve unrelated user or peer "
    "changes, keep edits tightly scoped, and verify claims from artifacts before "
    "reporting completion."
)
PERMISSION_MODES = ("acceptEdits", "auto", "bypassPermissions", "default", "dontAsk", "plan")

# GLM / Z.ai default backend (Z.ai Coding-Plan, the route this fleet uses; see
# GLM52-HOSTED-CACHE-COHERENCE-2026-06-19.md). The launcher fronts this with a
# local `fak serve` gateway so Claude Code's Anthropic wire reaches GLM. Override
# any field for a local/DGX GLM (SGLang/vLLM) via the --glm-* flags or env.
GLM_DEFAULT_BASE_URL = "https://api.z.ai/api/coding/paas/v4"
GLM_DEFAULT_MODEL = "zai-coding-plan/glm-5.2"
GLM_DEFAULT_API_KEY_ENV = "ZAI_API_KEY"
GLM_DEFAULT_PROVIDER = "openai"
# Loopback gateway ignores this key (it forwards the UPSTREAM key resolved from
# --api-key-env); Claude Code only requires ANTHROPIC_API_KEY to be non-empty.
GLM_PLACEHOLDER_ANTHROPIC_KEY = "fak-local-glm"
DEFAULT_GATEWAY_PORT = 8080
DEFAULT_GATEWAY_POLICY = "fak/examples/dogfood-claude-policy.json"
# Account tags/products that, when selected without --glm, IMPLY GLM mode — an
# opencode/GLM tier-2 account is not Anthropic-native, so launching it through
# Claude Code only works gateway-fronted.
GLM_ACCOUNT_HINTS = ("glm", "zai")


class SelectionError(RuntimeError):
    """Account selection failed with a user-actionable reason."""


class GatewayError(RuntimeError):
    """The GLM gateway could not be started or did not become healthy."""


def repo_root() -> Path:
    return fleet_version.repo_root(Path(__file__))


def default_claude_exe() -> str:
    return os.environ.get("FLEET_CLAUDE_EXE") or shutil.which("claude") or shutil.which("claude.exe") or "claude"


def default_fak_bin() -> str:
    """Locate the fak binary the GLM gateway runs. Prefer the repo-local build
    (``tools/.bin/fak``, what dogfood-claude.sh uses), then PATH, then a bare
    name as a last resort so --dry-run/plan still render a command string."""
    override = os.environ.get("FLEET_FAK_BIN")
    if override:
        return override
    for name in ("fak", "fak.exe"):
        local = repo_root() / "tools" / ".bin" / name
        if local.exists():
            return str(local)
    return shutil.which("fak") or shutil.which("fak.exe") or "fak"


def resolve_glm_backend(args: argparse.Namespace,
                        env: dict[str, str] | None = None) -> dict[str, str]:
    """Resolve the GLM upstream the gateway fronts.

    Precedence per field: explicit ``--glm-*`` flag > ``FAK_GLM_*`` env >
    built-in Z.ai Coding-Plan default. The env layer lets an operator pin a
    local/DGX GLM once (e.g. ``FAK_GLM_BASE_URL``) without passing flags every
    launch. ``env`` is injected for hermetic tests; defaults to ``os.environ``.
    """
    e = os.environ if env is None else env

    def pick(flag_value: str, env_name: str, default: str) -> str:
        if flag_value:
            return flag_value
        return e.get(env_name) or default

    return {
        "base_url": pick(getattr(args, "glm_base_url", "") or "",
                         "FAK_GLM_BASE_URL", GLM_DEFAULT_BASE_URL),
        "model": pick(getattr(args, "glm_model", "") or "",
                      "FAK_GLM_MODEL", GLM_DEFAULT_MODEL),
        "api_key_env": pick(getattr(args, "glm_api_key_env", "") or "",
                            "FAK_GLM_API_KEY_ENV", GLM_DEFAULT_API_KEY_ENV),
        "provider": pick(getattr(args, "glm_provider", "") or "",
                         "FAK_GLM_PROVIDER", GLM_DEFAULT_PROVIDER),
    }


def glm_backend_is_hosted_default(backend: dict[str, str]) -> bool:
    """True when the backend is the hosted Z.ai default (a missing upstream key
    is then fatal — every turn would 401). A local/DGX base-url may need no key."""
    return backend.get("base_url") == GLM_DEFAULT_BASE_URL


def glm_account_implies_gateway(account: dict[str, Any]) -> bool:
    """A selected account that is not Anthropic-native (opencode/GLM tier-2)
    can only launch through Claude Code gateway-fronted, so it IMPLIES GLM mode."""
    if str(account.get("product") or "").lower() == "opencode":
        return True
    blob = " ".join(str(account.get(k) or "")
                    for k in ("tag", "account", "model")).lower()
    return any(hint in blob for hint in GLM_ACCOUNT_HINTS)


def build_serve_argv(backend: dict[str, str], *, fak_bin: str,
                     port: int = DEFAULT_GATEWAY_PORT,
                     engine: str = "inkernel",
                     policy: str = "",
                     require_key_env: str = "") -> list[str]:
    """Build the ``fak serve`` argv that fronts the GLM backend, mirroring
    dogfood-claude.sh's SERVE_ARGS. Pure / unit-testable. The kernel exposes the
    Anthropic ``/v1/messages`` surface and proxies to ``base_url`` on the OpenAI
    wire, adjudicating every tool call."""
    argv = [
        fak_bin, "serve",
        "--addr", f"127.0.0.1:{port}",
        "--provider", backend["provider"],
        "--engine", engine,
        "--base-url", backend["base_url"],
        "--model", backend["model"],
    ]
    if backend.get("api_key_env"):
        argv.extend(["--api-key-env", backend["api_key_env"]])
    if policy and policy != "none":
        argv.extend(["--policy", policy])
    if require_key_env:
        argv.extend(["--require-key-env", require_key_env])
    return argv


def read_registry(path: str | None) -> dict[str, Any]:
    if not path:
        return fleet_accounts.load_registry()
    return fleet_accounts.load_registry(path)


def load_roster(home: str | None = None, registry_path: str | None = None,
                policy_path: str | None = None) -> list[dict[str, Any]]:
    policy = fleet_accounts.load_policy(policy_path or fleet_accounts.POLICY_PATH)
    rows = fleet_accounts.discover_accounts(home or fleet_accounts.USER, policy)
    return fleet_accounts.annotate_accounts(rows, registry=read_registry(registry_path))


def public_account(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "product",
        "account",
        "tag",
        "dir",
        "kind",
        "available",
        "blocked",
        "block_kind",
        "block_reason",
        "reset",
        "weekly",
        "throttled",
        "active_sessions",
        "live_sessions",
        "auth_blocked_sessions",
        "reason",
        "notes",
        "model_tier",
        "model",
        "small_model",
        "model_effort",
        "agent",
        "profile_source",
    )
    return {k: row.get(k) for k in keys if k in row}


def worker_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("kind") == "worker"]


def blocked_summary(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    out = []
    for r in worker_rows(rows):
        if r.get("available"):
            continue
        out.append({
            "tag": str(r.get("tag") or r.get("account") or ""),
            "account": str(r.get("account") or ""),
            "reason": str(r.get("block_reason") or r.get("reason") or "blocked"),
        })
    return out


def _exact_matches(rows: list[dict[str, Any]], token: str) -> list[dict[str, Any]]:
    needle = token.lower()
    matches = []
    for r in rows:
        values = [
            str(r.get("tag") or "").lower(),
            str(r.get("account") or "").lower(),
            str(r.get("dir") or "").lower(),
        ]
        if needle in values:
            matches.append(r)
    return matches


def _fuzzy_matches(rows: list[dict[str, Any]], token: str) -> list[dict[str, Any]]:
    needle = token.lower()
    return [
        r for r in rows
        if needle in str(r.get("tag") or "").lower()
        or needle in str(r.get("account") or "").lower()
    ]


def normalize_tier(value: str = "auto") -> str:
    v = (value or "auto").lower()
    if v in ("1", "t1", "tier1", "hard"):
        return "t1"
    if v in ("2", "t2", "tier2", "light", "easy"):
        return "t2"
    if v in ("3", "t3", "tier3"):
        return "t3"
    return "auto"


def normalize_work_kind(value: str = "") -> str:
    """Canonicalize a --work-kind token, or '' when none/unknown was given.

    Unlike a tier, a work-kind is passed THROUGH to route_account as its task_class
    (so the legible gardening/engineering class + reason survive into the packet's
    route field) rather than being collapsed to t1/t2 here. We only recognize the
    closed work-kind vocabulary; anything else (incl. a bare tier alias) returns ''
    so the caller falls back to the --tier path."""
    v = (value or "").strip().lower()
    if v in fleet_accounts.GARDENING_WORK_KINDS or v in fleet_accounts.ENGINEERING_WORK_KINDS:
        return v
    return ""


def choose_account(rows: list[dict[str, Any]], requested: str = "auto",
                   allow_blocked: bool = False, *, task_text: str = "",
                   tier: str = "auto", work_kind: str = "",
                   allow_tier_fallback: bool = False,
                   product: str | None = None) -> dict[str, Any]:
    wanted_product = (product or "").lower()
    scoped_rows = [
        r for r in rows
        if not wanted_product or str(r.get("product") or "").lower() == wanted_product
    ]
    workers = worker_rows(scoped_rows)
    if not workers:
        raise SelectionError("no worker accounts match the requested product")

    if requested == "auto":
        # A stated work-kind takes precedence over --tier and is passed through as
        # the task_class so gardening pins tier 2 / engineering pins tier 1 with the
        # legible class+reason. Gardening is NON-strict (up-shifts to tier 1 rather
        # than stalling when no tier-2 account is free); a bare tier is strict.
        kind = normalize_work_kind(work_kind)
        if kind:
            task_class = kind
            strict = False
        else:
            task_class = normalize_tier(tier)
            strict = task_class != "auto"
        route = fleet_accounts.route_account(
            rows,
            task_text=task_text,
            task_class=task_class,
            allow_tier_fallback=allow_tier_fallback,
            strict_tier=strict,
            product=product,
        )
        if route.get("ok") and route.get("account"):
            chosen = dict(route["account"])
            chosen["_route"] = {
                k: v for k, v in route.items()
                if k not in ("account",)
            }
            return chosen
        bits = "; ".join(
            f"{b.get('tag')}={b.get('reason')}"
            for b in route.get("blocked_target_accounts", [])
        )
        raise SelectionError(str(route.get("reason") or "no available worker accounts")
                             + (f": {bits}" if bits else ""))

    matches = _exact_matches(scoped_rows, requested) or _fuzzy_matches(scoped_rows, requested)
    if not matches:
        raise SelectionError(f"account {requested!r} was not found")
    if len(matches) > 1:
        priority = {"worker": 0, "excluded": 1, "non-account": 2}
        best = min(priority.get(str(m.get("kind")), 9) for m in matches)
        narrowed = [m for m in matches if priority.get(str(m.get("kind")), 9) == best]
        if len(narrowed) == 1:
            matches = narrowed
        else:
            names = ", ".join(str(m.get("tag") or m.get("account")) for m in matches)
            raise SelectionError(f"account {requested!r} is ambiguous: {names}")
    row = matches[0]
    if row.get("kind") != "worker":
        reason = row.get("reason") or row.get("block_reason") or row.get("kind")
        raise SelectionError(f"account {requested!r} is not offered: {reason}")
    if not row.get("available") and not allow_blocked:
        reason = row.get("block_reason") or row.get("reason") or "blocked"
        raise SelectionError(f"account {requested!r} is blocked: {reason}")
    return row


def ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def powershell_command(argv: list[str], env: dict[str, str], cwd: str) -> str:
    parts = [f"$env:{k}={ps_quote(v)}" for k, v in sorted(env.items())]
    parts.append(f"Set-Location {ps_quote(cwd)}")
    exe = ps_quote(argv[0])
    args = " ".join(ps_quote(a) for a in argv[1:])
    parts.append(f"& {exe}" + (f" {args}" if args else ""))
    return "; ".join(parts)


def posix_command(argv: list[str], env: dict[str, str], cwd: str) -> str:
    exports = " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(env.items()))
    cmd = " ".join(shlex.quote(a) for a in argv)
    return f"cd {shlex.quote(cwd)} && {exports} {cmd}".strip()


def load_agent_prompt(args: argparse.Namespace) -> str:
    if args.agent_prompt_file:
        return Path(args.agent_prompt_file).read_text(encoding="utf-8")
    return args.agent_prompt or DEFAULT_AGENT_PROMPT


def initial_prompt(args: argparse.Namespace) -> str:
    if args.goal:
        return "/goal " + args.goal.strip()
    if args.prompt:
        return args.prompt.strip()
    return ""


def agent_definition(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.no_agent:
        return None
    return {
        args.agent_name: {
            "description": args.agent_description,
            "prompt": load_agent_prompt(args),
        }
    }


def build_argv(args: argparse.Namespace, account: dict[str, Any],
               glm_mode: bool = False) -> list[str]:
    argv = [args.claude_exe or default_claude_exe()]
    if args.name:
        argv.extend(["--name", args.name])
    else:
        argv.extend(["--name", f"fleet-{account.get('tag') or 'agent'}"])
    # In GLM mode the model is pinned via the ANTHROPIC_*_MODEL env remap (the
    # gateway serves one GLM model for every Claude tier); a --model flag would
    # fight that, so it is omitted unless the operator forced one explicitly.
    if glm_mode:
        if args.model:
            argv.extend(["--model", args.model])
    else:
        model = args.model or str(account.get("model") or "")
        if model and model != "local":
            argv.extend(["--model", model])
    # GLM upstreams ignore Claude's effort knob; only pass effort outside GLM mode
    # (or when the operator set one explicitly).
    effort = args.effort or ("" if glm_mode else str(account.get("model_effort") or ""))
    if effort:
        argv.extend(["--effort", effort])
    if args.settings:
        argv.extend(["--settings", args.settings])
    for path in args.add_dir or []:
        argv.extend(["--add-dir", path])
    for mcp in args.mcp_config or []:
        argv.extend(["--mcp-config", mcp])
    agents = agent_definition(args)
    if agents is not None:
        argv.extend(["--agent", args.agent_name])
        argv.extend(["--agents", json.dumps(agents, separators=(",", ":"))])
    if args.permission_mode:
        argv.extend(["--permission-mode", args.permission_mode])
    if args.dangerously_skip_permissions:
        argv.append("--dangerously-skip-permissions")
    if args.print:
        argv.append("--print")
        if args.output_format:
            argv.extend(["--output-format", args.output_format])
    prompt = initial_prompt(args)
    if prompt:
        argv.append(prompt)
    return argv


def choose_glm_account(rows: list[dict[str, Any]], args: argparse.Namespace,
                       prompt: str) -> dict[str, Any]:
    """Choose the account whose ``CLAUDE_CONFIG_DIR`` owns a GLM session.

    In GLM mode the account supplies *identity* (which ``.claude*`` config dir),
    NOT an Anthropic model — the model comes from the gateway. So selection is
    relaxed vs. the native path: an explicit ``--account`` is honored across any
    product; ``auto`` prefers a GLM/opencode tier-2 account if one is available,
    else falls back to any available worker account for its config dir.
    """
    if args.account != "auto":
        return choose_account(rows, args.account, args.allow_blocked)

    # auto: try the GLM/tier-2 router first (any product), then any worker.
    try:
        return choose_account(rows, "auto", args.allow_blocked, task_text=prompt,
                              tier="t2", allow_tier_fallback=False, product=None)
    except SelectionError:
        pass
    available = [r for r in worker_rows(rows) if r.get("available")]
    if not available:
        raise SelectionError("no available worker account to own the GLM session config dir")
    # Least-busy first, matching the native router's preference.
    available.sort(key=lambda r: (int(r.get("live_sessions") or 0),
                                  int(r.get("active_sessions") or 0),
                                  str(r.get("tag") or "")))
    return available[0]


def glm_env(backend: dict[str, str], account: dict[str, Any], port: int) -> dict[str, str]:
    """The Claude-Code env that routes a session through the GLM gateway, mirroring
    dogfood-claude.sh's remap: point ANTHROPIC_BASE_URL at the loopback gateway and
    map every Claude model tier (and the small/fast background model) onto the one
    GLM model the gateway serves, so the main loop and background calls all hit it."""
    model = backend["model"]
    return {
        "CLAUDE_CONFIG_DIR": str(account.get("dir") or ""),
        # Claude Code appends /v1/messages itself, so NO /v1 here.
        "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}",
        "ANTHROPIC_API_KEY": GLM_PLACEHOLDER_ANTHROPIC_KEY,
        "ANTHROPIC_MODEL": model,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
        "ANTHROPIC_SMALL_FAST_MODEL": model,
    }


def build_packet(args: argparse.Namespace, rows: list[dict[str, Any]],
                 env_source: dict[str, str] | None = None) -> dict[str, Any]:
    prompt = initial_prompt(args)

    # GLM mode is explicit (--glm) or implied by selecting a non-Anthropic-native
    # account (opencode/GLM tier-2). Resolve the account first so the implication
    # can be detected, then decide.
    glm_requested = bool(getattr(args, "glm", False))
    if glm_requested:
        account = choose_glm_account(rows, args, prompt)
    elif args.account != "auto":
        # Explicit account by name: select across ALL products so an opencode/GLM
        # account resolves (its selection then IMPLIES gateway mode below). The
        # native claude-only scoping is for the auto-router default, not a named pick.
        account = choose_account(rows, args.account, args.allow_blocked)
    else:
        account = choose_account(
            rows,
            args.account,
            args.allow_blocked,
            task_text=prompt,
            tier=getattr(args, "tier", "auto"),
            work_kind=getattr(args, "work_kind", ""),
            allow_tier_fallback=getattr(args, "allow_tier_fallback", False),
            product="claude",
        )
    route = account.pop("_route", None)
    # An explicitly-selected GLM/opencode account implies GLM mode even without --glm.
    glm_mode = glm_requested or (
        args.account != "auto" and glm_account_implies_gateway(account))

    cwd = str(Path(args.cwd).resolve())

    glm_backend: dict[str, str] | None = None
    serve_argv: list[str] | None = None
    if glm_mode:
        glm_backend = resolve_glm_backend(args, env_source)
        # Fail before spawning anything if the hosted default has no upstream key.
        if glm_backend_is_hosted_default(glm_backend):
            key_env = glm_backend["api_key_env"]
            src = os.environ if env_source is None else env_source
            if not src.get(key_env):
                raise SelectionError(
                    f"GLM hosted backend ({glm_backend['base_url']}) needs an upstream key: "
                    f"set ${key_env} (or pass --glm-base-url for a local/DGX GLM that needs none)")
        port = int(getattr(args, "gateway_port", DEFAULT_GATEWAY_PORT))
        env = glm_env(glm_backend, account, port)
        serve_argv = build_serve_argv(
            glm_backend,
            fak_bin=getattr(args, "fak_bin", "") or default_fak_bin(),
            port=port,
            policy=getattr(args, "gateway_policy", DEFAULT_GATEWAY_POLICY),
            require_key_env=getattr(args, "require_key_env", ""),
        )
    else:
        env = {"CLAUDE_CONFIG_DIR": str(account.get("dir") or "")}

    argv = build_argv(args, account, glm_mode=glm_mode)
    agents = agent_definition(args)
    packet: dict[str, Any] = {
        "schema": SCHEMA,
        "app_version": fleet_version.app_version(repo_root()),
        "mode": "print" if args.print else "interactive",
        "account": public_account(account),
        "agent": {
            "enabled": agents is not None,
            "name": None if agents is None else args.agent_name,
            "definition": agents,
        },
        "cwd": cwd,
        "prompt": prompt,
        "route": route,
        "argv": argv,
        "env": env,
        "commands": {
            "powershell": powershell_command(argv, env, cwd),
            "posix": posix_command(argv, env, cwd),
        },
        "blocked_accounts": blocked_summary(rows),
    }
    if glm_mode:
        packet["glm"] = {
            "enabled": True,
            "backend": glm_backend,
            "gateway_port": int(getattr(args, "gateway_port", DEFAULT_GATEWAY_PORT)),
            "serve_argv": serve_argv,
            "serve_command": {
                "powershell": powershell_command(serve_argv, {}, cwd),
                "posix": " ".join(shlex.quote(a) for a in serve_argv),
            },
        }
    return packet


def print_roster(rows: list[dict[str, Any]]) -> None:
    available = [r for r in worker_rows(rows) if r.get("available")]
    blocked = [r for r in worker_rows(rows) if not r.get("available")]
    excluded = [r for r in rows if r.get("kind") == "excluded"]
    non_accounts = [r for r in rows if r.get("kind") == "non-account"]
    print("AVAILABLE")
    if available:
        for r in available:
            tier = f"t{r.get('model_tier', '?')}"
            model = r.get("model") or ""
            print(f"  {r['tag']:<18} {r['account']:<30} {tier:<3} {model:<18} active={r.get('active_sessions', 0)} live={r.get('live_sessions', 0)}")
    else:
        print("  (none)")
    if blocked:
        print("\nBLOCKED")
        for r in blocked:
            tier = f"t{r.get('model_tier', '?')}"
            model = r.get("model") or ""
            print(f"  {r['tag']:<18} {r['account']:<30} {tier:<3} {model:<18} {r.get('block_reason') or r.get('reason')}")
    if excluded:
        print("\nEXCLUDED")
        for r in excluded:
            print(f"  {r['tag']:<18} {r['account']:<30} {r.get('reason')}")
    if non_accounts:
        print("\nNON-ACCOUNT")
        for r in non_accounts:
            print(f"  {r['tag']:<18} {r['account']:<30} {r.get('reason')}")


def print_packet(packet: dict[str, Any]) -> None:
    acct = packet["account"]
    print(f"account: {acct.get('tag')} ({acct.get('account')})")
    print(f"tier:    t{acct.get('model_tier', '?')} model={acct.get('model') or '(default)'} effort={acct.get('model_effort') or '(default)'}")
    print(f"cwd:     {packet['cwd']}")
    print(f"mode:    {packet['mode']}")
    if packet.get("route"):
        route = packet["route"]
        fb = " fallback" if route.get("fallback_used") else ""
        print(f"route:   target=t{route.get('target_tier')} selected=t{route.get('selected_tier')}{fb}")
    if packet.get("glm", {}).get("enabled"):
        glm = packet["glm"]
        b = glm["backend"]
        print(f"glm:     model={b['model']} base_url={b['base_url']} "
              f"key_env=${b['api_key_env'] or '(none)'} gateway=:{glm['gateway_port']}")
    if packet["agent"]["enabled"]:
        print(f"agent:   {packet['agent']['name']}")
    if packet.get("prompt"):
        print(f"prompt:  {packet['prompt']}")
    if packet.get("glm", {}).get("enabled"):
        print("\nGLM gateway (started automatically by `launch`; shown for reference):")
        print(packet["glm"]["serve_command"]["posix"])
    print("\nPowerShell:")
    print(packet["commands"]["powershell"])
    print("\nPOSIX:")
    print(packet["commands"]["posix"])
    if packet.get("blocked_accounts"):
        print("\nblocked accounts:")
        for b in packet["blocked_accounts"]:
            print(f"  {b['tag']}: {b['reason']}")


def _healthz(port: int, timeout: float = 1.5) -> dict[str, Any] | None:
    """GET http://127.0.0.1:<port>/healthz, returning the parsed JSON or None."""
    try:
        with urllib.request.urlopen(  # noqa: S310 - fixed loopback URL
                f"http://127.0.0.1:{port}/healthz", timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "replace")
        doc = json.loads(body)
        return doc if isinstance(doc, dict) else {"raw": body}
    except (urllib.error.URLError, OSError, ValueError):
        return None


def start_gateway(packet: dict[str, Any], *, wait_s: float = 30.0) -> subprocess.Popen:
    """Start the GLM ``fak serve`` gateway for a packet and block until it is healthy.

    Mirrors dogfood-claude.sh's start+health+model-guard: refuse to start if the
    port is already serving a FOREIGN kernel (avoid wiring Claude Code to the wrong
    model), launch the child, poll /healthz until ready or the process dies, and
    confirm the healthy kernel reports OUR GLM model. Registers an atexit/signal
    teardown so the gateway never outlives this process. Raises GatewayError on any
    failure (caller surfaces it; nothing half-wired is left running)."""
    glm = packet["glm"]
    port = int(glm["gateway_port"])
    want_model = str(glm["backend"]["model"])
    serve_argv = [str(x) for x in glm["serve_argv"]]
    cwd = packet["cwd"]

    existing = _healthz(port)
    if existing is not None:
        raise GatewayError(
            f"port {port} already serving (model={existing.get('model')!r}) — a stale/foreign "
            f"kernel is up. Stop it or pass --gateway-port <other>.")

    fak_bin = serve_argv[0]
    if not (os.path.sep in fak_bin or shutil.which(fak_bin)):
        raise GatewayError(
            f"fak binary {fak_bin!r} not found — build it (tools/.bin/fak) or pass --fak-bin")

    proc = subprocess.Popen(serve_argv, cwd=cwd)
    _register_gateway_teardown(proc)

    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise GatewayError(
                f"fak serve exited (code {proc.returncode}) before becoming healthy; "
                f"check the backend at {glm['backend']['base_url']}")
        health = _healthz(port)
        if health is not None:
            reported = str(health.get("model") or "")
            # The kernel advertises the model id it serves; confirm it is ours so
            # we never adopt a different process that won the port mid-startup.
            if reported and want_model and reported != want_model:
                _stop_gateway(proc)
                raise GatewayError(
                    f"gateway on :{port} reports model {reported!r}, wanted {want_model!r}")
            return proc
        time.sleep(0.3)

    _stop_gateway(proc)
    raise GatewayError(f"GLM gateway did not become healthy on :{port} within {wait_s:.0f}s")


def _stop_gateway(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    except OSError:
        pass


def _register_gateway_teardown(proc: subprocess.Popen) -> None:
    """Ensure the gateway child is stopped on normal exit and on SIGINT/SIGTERM,
    matching dogfood-claude.sh's ``trap cleanup EXIT INT TERM``."""
    atexit.register(_stop_gateway, proc)

    def _handler(signum, _frame):  # pragma: no cover - signal path
        _stop_gateway(proc)
        # Re-raise default behavior so the process actually exits.
        raise KeyboardInterrupt() if signum == signal.SIGINT else SystemExit(128 + signum)

    for sig in (signal.SIGINT, getattr(signal, "SIGTERM", None)):
        if sig is None:
            continue
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):  # pragma: no cover - non-main-thread/platform
            pass


def launch_packet(packet: dict[str, Any], detached: bool = False) -> int:
    gateway: subprocess.Popen | None = None
    if packet.get("glm", {}).get("enabled"):
        # Start the GLM gateway BEFORE Claude Code so the first turn already routes
        # through the kernel. start_gateway raises on any failure.
        gateway = start_gateway(packet)
        print(f"GLM gateway up on :{packet['glm']['gateway_port']} "
              f"(model={packet['glm']['backend']['model']})")

    env = os.environ.copy()
    env.update(packet["env"])
    argv = [str(x) for x in packet["argv"]]
    cwd = packet["cwd"]
    try:
        if detached:
            if os.name == "nt":
                creationflags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
                proc = subprocess.Popen(argv, cwd=cwd, env=env, creationflags=creationflags)
            else:
                proc = subprocess.Popen(argv, cwd=cwd, env=env, start_new_session=True)
            print(f"launched pid={proc.pid} account={packet['account'].get('tag')}")
            # A detached session outlives this process; a launcher-managed gateway
            # cannot follow it, so refuse the combination rather than orphan-wire it.
            if gateway is not None:
                _stop_gateway(gateway)
                print("note: GLM gateway stopped — --detached cannot keep a launcher-managed "
                      "gateway alive; start `fak serve` separately for a detached GLM session",
                      file=sys.stderr)
            return 0
        return subprocess.call(argv, cwd=cwd, env=env)
    finally:
        if gateway is not None and not detached:
            _stop_gateway(gateway)


def command_support(exe: str) -> dict[str, Any]:
    proc = subprocess.run(
        [exe, "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    help_text = (proc.stdout or "") + (proc.stderr or "")
    flags = ["--agent", "--agents", "--permission-mode", "--dangerously-skip-permissions"]
    return {
        "exe": exe,
        "returncode": proc.returncode,
        "ok": proc.returncode == 0 and all(flag in help_text for flag in flags),
        "flags": {flag: flag in help_text for flag in flags},
    }


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--home", default=None, help="home dir containing .claude* accounts")
    parser.add_argument("--registry", default=None, help="sessions registry path (default: fleet_accounts default)")
    parser.add_argument("--policy", default=None, help="account policy path")


def add_launch_args(parser: argparse.ArgumentParser) -> None:
    add_common_args(parser)
    parser.add_argument("--account", default="auto", help="account tag/basename/path, or auto")
    parser.add_argument("--allow-blocked", action="store_true", help="allow explicitly selected blocked worker accounts")
    parser.add_argument(
        "-t", "--tier",
        choices=("auto", "1", "2", "3", "t1", "t2", "t3", "tier1", "tier2", "tier3", "hard", "light"),
        default="t1",
        help="model tier target: t1 by default; use auto for the trivial-task router; shorthand accepts -t1/-t2",
    )
    parser.add_argument(
        "--work-kind", "--kind", dest="work_kind", default="",
        choices=("", "gardening", "garden", "maintenance", "maint", "cleanup",
                 "chore", "triage", "engineering", "eng", "dev", "feature",
                 "implementation"),
        help="state the KIND of work so the right tier is pinned: gardening/maintenance "
             "-> tier 2 (non-strict, up-shifts if no tier-2 account), engineering -> tier 1. "
             "Takes precedence over --tier; this is how a maintenance loop drops to tier 2 "
             "without its prompt having to read as 'light'.",
    )
    parser.add_argument("--allow-tier-fallback", action="store_true",
                        help="allow hard/tier1 auto selection to fall back to tier2 when no tier1 account is available")
    parser.add_argument("--cwd", default=str(repo_root()), help="working directory for Claude Code")
    parser.add_argument("--claude-exe", default=default_claude_exe(), help="Claude Code executable")
    parser.add_argument("--name", default="", help="Claude Code session display name")
    parser.add_argument("--goal", default="", help="start with a /goal prompt")
    parser.add_argument("--prompt", default="", help="start with a plain prompt")
    parser.add_argument("--print", action="store_true", help="use claude --print for a non-interactive run")
    parser.add_argument("--output-format", choices=("text", "json", "stream-json"), default="")
    parser.add_argument("--permission-mode", choices=PERMISSION_MODES, default="auto")
    parser.add_argument("--dangerously-skip-permissions", action="store_true")
    parser.add_argument("--model", default="", help="Claude model/alias")
    parser.add_argument("--effort", default="", help="Claude effort level")
    parser.add_argument("--settings", default="", help="Claude settings JSON or file")
    parser.add_argument("--mcp-config", action="append", default=[], help="MCP config JSON or file; repeatable")
    parser.add_argument("--add-dir", action="append", default=[], help="additional tool-access directory; repeatable")
    parser.add_argument("--no-agent", action="store_true", help="do not pass Claude Code --agent/--agents")
    parser.add_argument("--agent-name", default=DEFAULT_AGENT_NAME)
    parser.add_argument("--agent-description", default=DEFAULT_AGENT_DESCRIPTION)
    parser.add_argument("--agent-prompt", default="")
    parser.add_argument("--agent-prompt-file", default="")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    # --- GLM / Z.ai parity: front a GLM backend with a fak gateway -------------
    glm = parser.add_argument_group(
        "GLM/Z.ai", "launch a GLM (or any OpenAI-compatible) model through Claude Code, "
        "gateway-fronted so the kernel adjudicates every tool call")
    glm.add_argument("--glm", action="store_true",
                     help="GLM mode: start a fak gateway in front of a GLM backend and wire "
                          "Claude Code to it (default backend: Z.ai Coding-Plan glm-5.2)")
    glm.add_argument("--glm-base-url", default="",
                     help=f"GLM upstream base URL (default: {GLM_DEFAULT_BASE_URL}); "
                          "point at a local/DGX SGLang/vLLM endpoint, e.g. http://dgx:8000/v1")
    glm.add_argument("--glm-model", default="",
                     help=f"GLM model id served upstream (default: {GLM_DEFAULT_MODEL})")
    glm.add_argument("--glm-api-key-env", default="",
                     help=f"env var holding the GLM upstream API key (default: {GLM_DEFAULT_API_KEY_ENV})")
    glm.add_argument("--glm-provider", default="",
                     help=f"upstream provider wire (default: {GLM_DEFAULT_PROVIDER})")
    glm.add_argument("--fak-bin", default="",
                     help="fak binary for the gateway (default: tools/.bin/fak, then PATH)")
    glm.add_argument("--gateway-port", type=int, default=DEFAULT_GATEWAY_PORT,
                     help=f"loopback port for the GLM gateway (default: {DEFAULT_GATEWAY_PORT})")
    glm.add_argument("--gateway-policy", default=DEFAULT_GATEWAY_POLICY,
                     help="capability-floor manifest the gateway enforces (or 'none')")
    glm.add_argument("--require-key-env", default="",
                     help="env var holding a bearer token the gateway requires (default: loopback, no auth)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch account-scoped Claude Code agent chat sessions.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="show account roster and runtime availability")
    add_common_args(p_list)
    p_list.add_argument("--json", action="store_true")

    p_plan = sub.add_parser("plan", help="build the launch packet without starting Claude")
    add_launch_args(p_plan)

    p_launch = sub.add_parser("launch", help="start Claude Code under the selected account")
    add_launch_args(p_launch)
    p_launch.add_argument("--detached", action="store_true", help="launch in a new/background process")
    p_launch.add_argument("--dry-run", action="store_true", help="print the packet instead of launching")

    p_doctor = sub.add_parser("doctor", help="check local Claude Code support for agent chat flags")
    p_doctor.add_argument("--claude-exe", default=default_claude_exe())
    p_doctor.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "doctor":
        doc = command_support(args.claude_exe)
        if args.json:
            print(json.dumps(doc, indent=2))
        else:
            print(f"claude: {doc['exe']}")
            for flag, ok in doc["flags"].items():
                print(f"  {flag:<32} {'ok' if ok else 'missing'}")
        return 0 if doc["ok"] else 1

    rows = load_roster(args.home, args.registry, args.policy)
    if args.cmd == "list":
        if args.json:
            print(json.dumps({"schema": SCHEMA, "accounts": [public_account(r) for r in rows]}, indent=2))
        else:
            print_roster(rows)
        return 0

    if args.goal and args.prompt:
        print("claude_agent_chat: pass only one of --goal or --prompt", file=sys.stderr)
        return 2

    try:
        packet = build_packet(args, rows)
    except SelectionError as exc:
        doc = {
            "schema": SCHEMA,
            "ok": False,
            "reason": str(exc),
            "accounts": [public_account(r) for r in rows],
        }
        if args.json:
            print(json.dumps(doc, indent=2))
        else:
            print(f"claude_agent_chat: {exc}", file=sys.stderr)
            if blocked_summary(rows):
                print("blocked accounts:", file=sys.stderr)
                for b in blocked_summary(rows):
                    print(f"  {b['tag']}: {b['reason']}", file=sys.stderr)
        return 1

    if args.cmd == "plan" or args.dry_run:
        if args.json:
            print(json.dumps({"ok": True, **packet}, indent=2))
        else:
            print_packet(packet)
        return 0

    return launch_packet(packet, detached=args.detached)


if __name__ == "__main__":
    raise SystemExit(main())
