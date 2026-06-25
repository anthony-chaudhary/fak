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

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import dispatch_worker as dw  # noqa: E402

GLM_BASE = "http://127.0.0.1:8001/v1"


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
