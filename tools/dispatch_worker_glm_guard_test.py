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

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import account_probe as ap  # noqa: E402
import dispatch_glm_docs as gd  # noqa: E402
import dispatch_worker as dw  # noqa: E402

GLM_BASE = "http://127.0.0.1:8001/v1"
LAB_PROXY_BASE = "http://127.0.0.1:18080/v1"


def _env(base: str | None) -> dict[str, str]:
    """A hermetic env: a resolvable FAK_BIN (the Python exe) so guard-wrapping
    engages without a built fak, with FLEET_DOGFOOD_GUARD_BASEURL as the knob."""
    e = {"FAK_BIN": sys.executable}
    if base is not None:
        e["FLEET_DOGFOOD_GUARD_BASEURL"] = base
    return e


def test_opencode_guarded_when_base_url_set() -> None:
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
    raw = dw.build_command("glm", "opencode")
    cmd, guarded = dw.guarded_launch_command(raw, "glm", "opencode", ROOT, env=_env(None))
    assert guarded is False, "opencode must stay UNguarded without a base URL (no misroute)"
    assert cmd == raw, cmd


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
    def ok_conn(base_url, *, timeout):
        return {"reachable": True, "status": 200,
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
    def down_conn(base_url, *, timeout):
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
