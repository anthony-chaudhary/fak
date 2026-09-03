#!/usr/bin/env python3
"""GLM/opencode guard-lane witness for tools/dispatch_worker.py (issue #730).

The fleet-through-kernel change guards the **claude** lane by default but leaves the
**opencode/GLM** lane unguarded on purpose: opencode fronts a *local* upstream (a GLM
server), so `fak guard --provider openai` with no base URL would MISROUTE it to the
public OpenAI API. `guard_wrap` already has the hook — it wraps a non-claude backend
only when `FLEET_DOGFOOD_GUARD_BASEURL` names that local upstream — but until #730 no
node set it, so the GLM lane dogfooded 0%.

This test pins the contract the per-node config relies on, so it can't silently rot:

  * base URL SET   => `fak guard --provider openai --base-url <glm> -- opencode …`,
                      guarded=True (the kernel fronts the OpenAI wire to the GLM box).
  * base URL UNSET => the worker is launched UNCHANGED, guarded=False (we refuse to
                      misroute a local-upstream worker to api.openai.com).
  * claude lane    => still guarded with NO base-url override (regression guard: the
                      opencode wiring must not change the claude default).

Run: `python tools/dispatch_worker_glm_guard_test.py`  (exit 0 = all pass),
or `python -m pytest tools/dispatch_worker_glm_guard_test.py -q`.
"""
from __future__ import annotations

import ast
import atexit
import itertools
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import account_probe as ap  # noqa: E402
import dispatch_glm_docs as gd  # noqa: E402
import dispatch_worker as dw  # noqa: E402

GLM_BASE = "http://127.0.0.1:8001/v1"
LAB_PROXY_BASE = "http://127.0.0.1:18080/v1"
SEAT_BASE = "http://127.0.0.1:7777/v1"
# Pinned so guard-wrapping never asks the OS for a free port (a bind is host state,
# and the port lands in argv). Any value works; nothing listens on it in a test.
GUARD_ADDR = "127.0.0.1:65432"

# --- the injected opencode seat (#5403) ---------------------------------------
# `opencode_guard_base_url` resolves, in order: (1) the provider-scoped
# FLEET_DOGFOOD_GUARD_BASEURL_<PROVIDER>, (2) the selected provider's opencode account
# config `options.baseURL`, (3) the legacy global FLEET_DOGFOOD_GUARD_BASEURL, and only
# for the default provider. The tests below assert tiers 2 and 3, so tier 2 has to be
# STATED rather than inherited from whichever `~/.config/opencode/opencode.json` the
# running box happens to carry: reading the real seat is what made these red for
# everyone who has one, green for everyone who does not, and therefore meaningless.
#
# `dispatch_worker` resolves the account config entirely from the env mapping it is
# handed -- OPENCODE_CONFIG first, then XDG_CONFIG_HOME/opencode/ -- so naming both
# inside a temp dir injects the seat through a seam the product already has, the same
# way `target_resolver` injects it for the `test_glm_gateway_*` pair below. The
# precedence chain itself is untouched; only the inputs are now ours.
_SEAT_DIR = tempfile.mkdtemp(prefix="fak-opencode-seat-")
atexit.register(shutil.rmtree, _SEAT_DIR, ignore_errors=True)
_SEAT_SEQ = itertools.count()


def _seat_config(seat: str | None) -> Path:
    """A path for the injected opencode account config: written with ``seat`` as the
    default provider's ``options.baseURL``, or left ABSENT for a host with no seat.
    Unique per call so one test's seat can never bleed into the next."""
    path = Path(_SEAT_DIR) / f"opencode-{next(_SEAT_SEQ)}.json"
    if seat is not None:
        path.write_text(json.dumps({"provider": {
            dw.OPENCODE_DEFAULT_PROVIDER_ID: {"options": {"baseURL": seat}}}}),
            encoding="utf-8")
    return path


def _env(base: str | None, *, seat: str | None = None) -> dict[str, str]:
    """A hermetic env: a resolvable FAK_BIN (the Python exe) so guard-wrapping engages
    without a built fak, FLEET_DOGFOOD_GUARD_BASEURL as the tier-3 knob, and the tier-2
    account config pinned to ``seat`` (default: NO seat configured). Every host-read
    `dispatch_worker` performs on this path is named here, so the verdict is the same on
    a box with an opencode seat and on a box without one."""
    cfg = _seat_config(seat)
    e = {
        "FAK_BIN": sys.executable,
        # Tier 2, pinned twice: OPENCODE_CONFIG is consulted first and falls back to
        # the ambient value only when absent, and XDG_CONFIG_HOME governs the
        # `<root>/opencode/opencode.json` candidates that otherwise sit under $HOME.
        "OPENCODE_CONFIG": str(cfg),
        "XDG_CONFIG_HOME": _SEAT_DIR,
        "FLEET_DOGFOOD_GUARD_ADDR": GUARD_ADDR,
    }
    if base is not None:
        e["FLEET_DOGFOOD_GUARD_BASEURL"] = base
    return e


def test_opencode_guarded_when_base_url_set() -> None:
    # No seat configured (injected), so the legacy tier-3 knob is the only base URL
    # in play -- which is the tier this test names.
    raw = dw.build_command("glm", "opencode")
    cmd, guarded = dw.guarded_launch_command(raw, "glm", "opencode", ROOT, env=_env(GLM_BASE))
    assert guarded is True, "opencode should be guarded when the GLM base URL is set"
    joined = " ".join(cmd)
    assert "guard" in cmd, joined
    assert "--provider" in cmd and cmd[cmd.index("--provider") + 1] == "openai", joined
    assert "--base-url" in cmd and cmd[cmd.index("--base-url") + 1] == GLM_BASE, joined
    # The real worker argv survives intact after the `--` separator.
    assert cmd[cmd.index("--") + 1:] == raw, joined


def test_opencode_guarded_with_loopback_lab_proxy_base() -> None:
    raw = dw.build_command("glm", "opencode")
    cmd, guarded = dw.guarded_launch_command(raw, "glm", "opencode", ROOT, env=_env(LAB_PROXY_BASE))
    assert guarded is True, "opencode should be guarded when the lab proxy base is configured"
    joined = " ".join(cmd)
    assert "--base-url" in cmd and cmd[cmd.index("--base-url") + 1] == LAB_PROXY_BASE, joined
    assert "guard" in cmd and cmd[cmd.index("--provider") + 1] == "openai", joined
    assert cmd[cmd.index("--") + 1:] == raw, joined


def test_opencode_guarded_gets_wallclock_max_duration() -> None:
    # The detached docs-lane spawn path (dispatch_glm_docs.py) runs OFF the main
    # dispatcher's reap loop, so the guard must cap the worker's wall clock itself --
    # otherwise a worker retry-looping against a down gateway runs unbounded.
    raw = dw.build_command("glm", "opencode")
    cmd, guarded = dw.guarded_launch_command(raw, "glm", "opencode", ROOT, env=_env(GLM_BASE))
    assert guarded is True, cmd
    assert "--max-duration" in cmd, cmd
    assert cmd[cmd.index("--max-duration") + 1] == dw.OPENCODE_GUARD_MAX_DURATION, cmd
    # It stays BEFORE the `--` separator (a guard flag, not a worker arg).
    assert cmd.index("--max-duration") < cmd.index("--"), cmd
    assert cmd[cmd.index("--") + 1:] == raw, cmd


def test_opencode_unguarded_without_base_url() -> None:
    # No seat configured AND no knob: there is no base URL at ANY tier, which is the
    # only state in which "unguarded" is the right answer. Before #5403 this test read
    # the real seat off disk, so it asserted the opposite of what it claimed on any box
    # that had one.
    raw = dw.build_command("glm", "opencode")
    cmd, guarded = dw.guarded_launch_command(raw, "glm", "opencode", ROOT, env=_env(None))
    assert guarded is False, "opencode must stay UNguarded without a base URL (no misroute)"
    assert cmd == raw, cmd


def test_configured_seat_base_url_outranks_the_legacy_env_knob() -> None:
    # The counterpart of the two above, and the guard against "fixing" them by
    # reordering the tiers: a seat that fronts its own upstream WINS over the legacy
    # global knob (#4771 depends on this). With the seat injected rather than read off
    # the host, both directions are pinned by the same seam.
    raw = dw.build_command("glm", "opencode")
    cmd, guarded = dw.guarded_launch_command(
        raw, "glm", "opencode", ROOT, env=_env(GLM_BASE, seat=SEAT_BASE))
    assert guarded is True, cmd
    assert cmd[cmd.index("--base-url") + 1] == SEAT_BASE, cmd


def test_claude_lane_guarded_without_base_url_override() -> None:
    # Regression guard: the opencode base-url wiring must not leak a --base-url onto
    # the claude lane, which proxies the public Anthropic API with no override.
    raw = dw.build_command("docs", "claude")
    cmd, guarded = dw.guarded_launch_command(raw, "docs", "claude", ROOT, env=_env(GLM_BASE))
    assert guarded is True, cmd
    assert "--base-url" not in cmd, cmd
    assert cmd[cmd.index("--provider") + 1] == "anthropic", cmd


# --- issue #4771: gateway-down vs gateway-ready dispatch behavior --------------
# The glm-docs dispatcher probes the SAME local guard gateway a worker would route
# through BEFORE spawning. These pin both branches so the docs fleet can never again
# silently refuse every launch on an un-probed gateway.


def _glm_target_resolver(base: str):
    """A hermetic ``target_resolver`` for probe_opencode_account: no dispatch import,
    no opencode.json read -- just the glm model and the guard base under test."""
    def _resolve(row, **kw):
        return gd.GLM_MODEL, base
    return _resolve


def _without_gateway_override(fn):
    """Run ``fn`` with FAK_GLM_GUARD_GATEWAY unset so the documented :18080 default
    applies regardless of the ambient environment; restore it afterwards."""
    saved = os.environ.pop("FAK_GLM_GUARD_GATEWAY", None)
    try:
        return fn()
    finally:
        if saved is not None:
            os.environ["FAK_GLM_GUARD_GATEWAY"] = saved


def test_glm_gateway_ready_probe_admits_spawn() -> None:
    # Gateway UP: the local guard gateway answers GET /models with 200 -> OK, which is
    # NOT in the preflight refuse set, so dispatch_glm_docs proceeds past the gateway
    # gate to a real (non-gateway) capacity decision.
    def ok_conn(base_url, *, timeout, **kwargs):
        return {"reachable": True, "status": 200, "generated_text": "pong",
                "body": '{"data":[{"id":"mock","object":"model"}]}', "error": ""}
    pf = ap.probe_opencode_account(
        {"account": "opencode", "tag": "zai-coding-plan"},
        connector=ok_conn, target_resolver=_glm_target_resolver(LAB_PROXY_BASE))
    assert pf["status"] == "OK", pf
    assert pf["status"].upper() not in gd._PREFLIGHT_REFUSE, pf


def test_glm_gateway_down_probe_refuses_spawn() -> None:
    # Gateway DOWN: the local guard gateway refuses the TCP connect -> GATEWAY_DOWN,
    # which IS in the refuse set, so dispatch_glm_docs skips the spawn (never fires a
    # worker at an unreachable gateway) with a truthful gateway-reachability reason.
    def down_conn(base_url, *, timeout, **kwargs):
        return {"reachable": False, "status": None, "body": "",
                "error": "[WinError 10061] connection refused"}
    pf = ap.probe_opencode_account(
        {"account": "opencode", "tag": "zai-coding-plan"},
        connector=down_conn, target_resolver=_glm_target_resolver(LAB_PROXY_BASE))
    assert pf["status"] == "GATEWAY_DOWN", pf
    assert pf["status"].upper() in gd._PREFLIGHT_REFUSE, pf
    assert "18080" in (pf.get("block_reason") or ""), pf


def test_glm_seat_routes_through_local_18080_guard_gateway() -> None:
    # #4771: apply_glm_guard_gateway pins the PROVIDER-SCOPED guard base (highest
    # precedence), so opencode_guard_base_url resolves the local :18080 gateway for the
    # glm command -- over the seat's own public-provider baseURL that had hijacked it.
    def check() -> None:
        env: dict[str, str] = {}
        base = gd.apply_glm_guard_gateway(env)
        assert base == gd.DEFAULT_GLM_GUARD_GATEWAY == LAB_PROXY_BASE, base
        assert env[gd.GLM_GUARD_BASEURL_ENV] == base, env
        cmd = ["opencode", "run", "-m", gd.GLM_MODEL, "prompt"]
        assert dw.opencode_guard_base_url(cmd, env) == base, (
            "the glm worker must front the supervised local :18080 gateway")
    _without_gateway_override(check)


def test_operator_guard_base_is_not_overridden() -> None:
    # An operator who pins the provider-scoped var themselves WINS: the default pin
    # must never stomp an explicit endpoint (e.g. a DGX lab proxy on another port).
    env = {gd.GLM_GUARD_BASEURL_ENV: "http://127.0.0.1:9999/v1"}
    assert gd.apply_glm_guard_gateway(env) == "http://127.0.0.1:9999/v1", env
    assert env[gd.GLM_GUARD_BASEURL_ENV] == "http://127.0.0.1:9999/v1", env


def test_empty_gateway_env_opts_out_of_forced_local_route() -> None:
    # FAK_GLM_GUARD_GATEWAY exported EMPTY is the explicit opt-out: nothing is pinned,
    # so the seat falls back to its own base-url resolution.
    saved = os.environ.get("FAK_GLM_GUARD_GATEWAY")
    try:
        os.environ["FAK_GLM_GUARD_GATEWAY"] = ""
        env: dict[str, str] = {}
        assert gd.apply_glm_guard_gateway(env) == "", env
        assert gd.GLM_GUARD_BASEURL_ENV not in env, env
    finally:
        if saved is None:
            os.environ.pop("FAK_GLM_GUARD_GATEWAY", None)
        else:
            os.environ["FAK_GLM_GUARD_GATEWAY"] = saved


# --- --codex-loop-gate is CODEX-ONLY -----------------------------------------
# 5a2f88f244 (#2400) lost the newline before `if backend == "codex":`, appending it to
# the comment on the line above. The flag's append then ran for every NON-CLAUDE
# backend, so opencode workers were handed a codex-only guard flag. Python accepted it
# silently -- the swallowed `if` left a syntactically valid file -- which is exactly why
# it survived. These pin the flag to the one backend it belongs to.


def _wrap(backend: str, cmd: list[str]) -> list[str]:
    return dw.guard_wrap(cmd, fak_bin="fak", lane="t", backend=backend,
                         workspace=ROOT, env={})


def test_codex_loop_gate_flag_is_codex_only() -> None:
    # Codex KEEPS it (the flag's actual purpose -- this must not regress the other way).
    assert "--codex-loop-gate" in _wrap("codex", ["codex", "exec", "prompt"])
    # Claude never had it.
    assert "--codex-loop-gate" not in _wrap("claude", ["claude", "-p", "x"])


def test_opencode_does_not_get_the_codex_loop_gate_flag() -> None:
    # The regression itself: an opencode worker must not carry a codex-only flag.
    raw = dw.build_command("glm", "opencode")
    cmd, _ = dw.guarded_launch_command(raw, "glm", "opencode", ROOT,
                                       env={"FLEET_DOGFOOD_GUARD": "1"})
    assert "--codex-loop-gate" not in cmd, cmd



def test_glm_docs_spawn_is_guard_wrapped() -> None:
    """The dedicated pool must actually apply the guard config it preflights."""
    tree = ast.parse((ROOT / "tools" / "dispatch_glm_docs.py").read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    wraps = [n for n in calls if isinstance(n.func, ast.Attribute)
             and isinstance(n.func.value, ast.Name) and n.func.value.id == "dw"
             and n.func.attr == "guarded_launch_command"]
    assert len(wraps) == 1, "GLM docs spawn must pass through guarded_launch_command"
    assert ast.unparse(wraps[0].args[1]) == "'docs'"
    assert ast.unparse(wraps[0].args[2]) == "'opencode'"


def test_glm_docs_spawn_passes_prompt_payload() -> None:
    """The dedicated opencode launcher passes the full prompt out-of-band."""
    src = (ROOT / "tools" / "dispatch_glm_docs.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "ird"
        and node.func.attr == "spawn_issue_worker"
    ]
    assert len(calls) == 1, "expected one dedicated GLM spawn seam"
    kws = {kw.arg: kw.value for kw in calls[0].keywords if kw.arg}
    assert "prompt_payload" in kws
    assert isinstance(kws["prompt_payload"], ast.Subscript)
    assert ast.unparse(kws["prompt_payload"]) == "rb['prompt']"

def main() -> int:
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    if failures:
        print(f"\n{failures} test(s) failed")
        return 1
    print("\nall GLM/opencode guard-lane tests passed")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
