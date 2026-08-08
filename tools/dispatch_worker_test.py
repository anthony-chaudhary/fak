#!/usr/bin/env python3
"""Hermetic tests for tools/dispatch_worker.py."""
from __future__ import annotations

import importlib.util
import inspect
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "dispatch_worker.py"


def load():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("dispatch_worker", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class DispatchWorkerTest(unittest.TestCase):
    def test_resolve_backend_flag_beats_env_beats_default(self) -> None:
        mod = load()
        self.assertEqual(mod.resolve_backend("claude", {"FLEET_WORKER_BACKEND": "opencode"}), "claude")
        self.assertEqual(mod.resolve_backend(None, {"FLEET_WORKER_BACKEND": "opencode"}), "opencode")
        self.assertEqual(mod.resolve_backend(None, {}), "claude")

    def test_resolve_backend_rejects_unknown(self) -> None:
        mod = load()
        with self.assertRaises(ValueError):
            mod.resolve_backend("cursor", None)
        with self.assertRaises(ValueError):
            mod.resolve_backend(None, {"FLEET_WORKER_BACKEND": "nope"})

    def test_claude_command_shape_matches_dos_toml_reference(self) -> None:
        mod = load()
        cmd = mod.build_command("adjudicator", "claude")
        self.assertEqual(cmd[0], "claude")
        self.assertEqual(cmd[1:4], ["-p", "--permission-mode", "bypassPermissions"])
        # BARE project-skill form (git-tracked .claude/skills/dos-dispatch-loop),
        # not the namespaced plugin form -- the plugin cache is per-account and is
        # empty for freshly-enrolled worker dirs, so the namespaced form fails
        # closed ("Unknown command") and the worker exits 0 with zero work done.
        self.assertEqual(cmd[4], "/dos-dispatch-loop --lane adjudicator")

    def test_opencode_command_uses_dispatch_agent_and_skip_permissions(self) -> None:
        mod = load()
        cmd = mod.build_command("agent", "opencode")
        self.assertEqual(cmd[0], "opencode")
        self.assertIn("--dangerously-skip-permissions", cmd)
        # --print-logs surfaces opencode's run-level failures into the worker log
        # instead of a silent banner-only no-op (#1275).
        self.assertIn("--print-logs", cmd)
        self.assertEqual(cmd[cmd.index("--agent") + 1], "dos-dispatch")
        self.assertEqual(cmd[-1], "dispatch lane agent")

    def test_build_command_rejects_empty_lane(self) -> None:
        mod = load()
        with self.assertRaises(ValueError):
            mod.build_command("", "claude")

    def test_child_env_stamps_assignment_and_passes_through(self) -> None:
        mod = load()
        env = mod.child_env("canon", "claude", Path("C:/work/fleet"), base={"PATH": "x", "KEEP_ME": "1"})
        self.assertEqual(env["DISPATCH_LANE"], "canon")
        self.assertEqual(env["DISPATCH_BACKEND"], "claude")
        self.assertEqual(env["DISPATCH_WORKSPACE"], str(Path("C:/work/fleet")))
        self.assertEqual(env["KEEP_ME"], "1")  # base env preserved

    def test_dry_run_does_not_call_runner(self) -> None:
        mod = load()

        def fail_runner(_cmd, _cwd, _env):
            raise AssertionError("dry run must not call runner")

        # main() with --dry-run must not launch.
        rc = mod.main(["--lane", "docs", "--dry-run", "--json"])
        self.assertEqual(rc, 0)

        payload = mod.build_payload(
            lane="docs", backend="claude", workspace=Path("C:/work/fleet"), dry_run=True
        )
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["dry_run"])
        self.assertIsNone(payload["result"])
        self.assertEqual(payload["backend"], "claude")
        fail_runner  # referenced to keep the lint honest about intent

    def test_live_launch_calls_runner_with_resolved_command_and_env(self) -> None:
        mod = load()
        seen: list[tuple] = []

        def runner(cmd, cwd, env):
            seen.append((list(cmd), cwd, env))
            return {"returncode": 0, "stdout": "", "stderr": ""}

        command = mod.build_command("recall", "opencode")
        env = mod.child_env("recall", "opencode", Path("C:/work/fleet"), base={})
        result = mod.launch(command, Path("C:/work/fleet"), env, runner=runner)
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(len(seen), 1)
        ran_cmd, ran_cwd, ran_env = seen[0]
        self.assertEqual(ran_cmd[0], "opencode")
        self.assertEqual(ran_env["DISPATCH_LANE"], "recall")
        self.assertEqual(ran_env["DISPATCH_BACKEND"], "opencode")

    def test_live_nonzero_returncode_propagates_to_payload_ok_false(self) -> None:
        mod = load()

        def runner(_cmd, _cwd, _env):
            return {"returncode": 1, "stdout": "", "stderr": "boom"}

        result = mod.launch(mod.build_command("x", "claude"), Path("."), {}, runner=runner)
        payload = mod.build_payload(
            lane="x", backend="claude", workspace=Path("."), dry_run=False, result=result
        )
        self.assertFalse(payload["ok"])

    def test_launch_real_runs_trivial_command_and_returns_rc(self) -> None:
        mod = load()
        result = mod.launch([sys.executable, "-c", "import sys; sys.exit(0)"],
                            Path("."), dict(os.environ), timeout_s=30)
        self.assertEqual(result["returncode"], 0)

    def test_launch_real_timeout_kills_tree_and_returns_124(self) -> None:
        mod = load()
        # A 5s sleeper under a 1s cap must time out (124) and be reaped by
        # terminate_tree -- not hang. Real-process exercise of the timeout path.
        result = mod.launch([sys.executable, "-c", "import time; time.sleep(5)"],
                            Path("."), dict(os.environ), timeout_s=1)
        self.assertEqual(result["returncode"], 124)
        self.assertTrue(result.get("timeout"))

    def test_resolve_exe_falls_back_to_name_when_not_found(self) -> None:
        mod = load()
        # A name that will not resolve should fall back to the bare name rather
        # than raise (launch then surfaces FileNotFoundError as returncode 127).
        self.assertEqual(mod.resolve_exe("definitely-not-a-real-backend-xyz"), "definitely-not-a-real-backend-xyz")

    def test_normalize_timeout_caps_by_default_and_opts_out_at_zero(self) -> None:
        mod = load()
        # The default cap bounds an unattended worker; 0/negative/None opt out.
        self.assertEqual(mod.normalize_timeout(mod.DEFAULT_TIMEOUT_S), mod.DEFAULT_TIMEOUT_S)
        self.assertEqual(mod.normalize_timeout(60), 60)
        self.assertIsNone(mod.normalize_timeout(0))
        self.assertIsNone(mod.normalize_timeout(-5))
        self.assertIsNone(mod.normalize_timeout(None))
        # The default is a real bound, not the old unbounded None.
        self.assertIsNotNone(mod.DEFAULT_TIMEOUT_S)
        self.assertGreater(mod.DEFAULT_TIMEOUT_S, 0)

    # --- Dogfood: front the worker with the kernel (`fak guard`) ---------------

    def test_guard_enabled_default_on_and_opt_out_values(self) -> None:
        mod = load()
        self.assertTrue(mod.guard_enabled({}))                                 # unset -> ON (dogfood default)
        self.assertTrue(mod.guard_enabled({"FLEET_DOGFOOD_GUARD": "1"}))
        self.assertTrue(mod.guard_enabled({"FLEET_DOGFOOD_GUARD": "on"}))
        for off in ("0", "off", "false", "no", "", "disable", "DISABLED", " Off "):
            self.assertFalse(mod.guard_enabled({"FLEET_DOGFOOD_GUARD": off}), off)

    def test_resolve_fak_bin_prefers_env_then_freshest_else_none(self) -> None:
        mod = load()
        # An explicit FAK_BIN that exists wins (use this very test file as a stand-in).
        existing = str(Path(__file__).resolve())
        self.assertEqual(
            mod.resolve_fak_bin(Path("C:/nope"), {"FAK_BIN": existing}), existing)
        # A non-existent FAK_BIN is ignored; with an empty workspace and a PATH that
        # holds no `fak`, the result is None (fail-open signal).
        got = mod.resolve_fak_bin(
            Path("C:/definitely/not/a/repo/xyz"),
            {"FAK_BIN": "C:/no/such/fak", "PATH": str(Path(__file__).resolve().parent / "_no_fak_here_xyz")})
        self.assertIsNone(got)

    def test_resolve_fak_bin_takes_the_freshest_build_not_the_in_tree_one(self) -> None:
        """#5856: nothing refreshes tools/.bin, but FakSelfUpdate rebuilds the PATH copy
        every 20 min. An unconditional in-tree preference therefore fronted every worker's
        `fak guard` with a build 34 commits behind HEAD and stamped +dirty. Rank by build
        time so the abandoned copy loses -- and so a developer's fresh dogfood build still
        wins, which is the intent the old order was reaching for."""
        mod = load()
        exe = "fak.exe" if os.name == "nt" else "fak"
        with tempfile.TemporaryDirectory() as td:
            ws, pathdir = Path(td) / "ws", Path(td) / "bin"
            intree = ws / "tools" / ".bin" / exe
            intree.parent.mkdir(parents=True)
            pathdir.mkdir(parents=True)
            onpath = pathdir / exe
            for p in (intree, onpath):
                p.write_text("stub", encoding="utf-8")
            env = {"PATH": str(pathdir)}

            # The refreshed PATH copy is newer -> it wins. This is the fleet case, and the
            # exact assertion the old in-tree-first resolver fails.
            os.utime(intree, (1_000_000, 1_000_000))
            os.utime(onpath, (2_000_000, 2_000_000))
            self.assertEqual(mod.resolve_fak_bin(ws, env), str(onpath))

            # A just-rebuilt dogfood binary is newer -> it wins. The dev case still holds.
            os.utime(intree, (3_000_000, 3_000_000))
            self.assertEqual(mod.resolve_fak_bin(ws, env), str(intree))

            # Equal build times -> PATH, the copy something is accountable for refreshing.
            os.utime(intree, (4_000_000, 4_000_000))
            os.utime(onpath, (4_000_000, 4_000_000))
            self.assertEqual(mod.resolve_fak_bin(ws, env), str(onpath))

            # An unreadable candidate sorts last rather than raising (fail-open).
            self.assertEqual(mod.fak_binary_build_time(str(ws / "no" / "such")), -1.0)

    def test_guard_provider_maps_claude_to_anthropic_else_openai(self) -> None:
        mod = load()
        self.assertEqual(mod.guard_provider("claude"), "anthropic")
        self.assertEqual(mod.guard_provider("opencode"), "openai")

    def test_guard_audit_path_is_per_session_under_dispatch_runs(self) -> None:
        mod = load()
        p = mod.guard_audit_path(Path("C:/work/fak"), "gate way/1", "claude")
        self.assertEqual(p.parent.name, "guard-audit")
        self.assertEqual(p.parent.parent.name, ".dispatch-runs")
        self.assertTrue(p.name.endswith(".jsonl"))
        self.assertNotIn("/", p.name)   # lane separators sanitized out of the filename
        self.assertNotIn(" ", p.name)
        self.assertTrue(p.name.startswith("gate_way_1-claude-"))  # lane prefix preserved for globbing

    def test_guard_audit_path_is_unique_per_call(self) -> None:
        mod = load()
        # Two workers on the SAME lane must NOT resolve to the same journal file, or
        # their independent hash chains would braid into one unverifiable file.
        a = mod.guard_audit_path(Path("C:/work/fak"), "gateway", "claude")
        b = mod.guard_audit_path(Path("C:/work/fak"), "gateway", "claude")
        self.assertNotEqual(a, b)

    def test_guard_wrap_claude_fronts_with_fak_guard_anthropic(self) -> None:
        mod = load()
        raw = mod.build_command("gateway", "claude")
        wrapped = mod.guard_wrap(raw, fak_bin="/usr/bin/fak", lane="gateway",
                                 backend="claude", workspace=Path("C:/work/fak"))
        self.assertEqual(wrapped[0], "/usr/bin/fak")
        self.assertEqual(wrapped[1], "guard")
        self.assertEqual(wrapped[wrapped.index("--provider") + 1], "anthropic")
        self.assertEqual(wrapped[wrapped.index("--precompact-hook") + 1], "enforce")
        # ADEQUACY guardrail, not mere presence. This seeds guard's per-session
        # ContextTokensLeft, drawn down by each turn's FULL resident window. It MUST
        # exceed the worker's irreducible ~62K baseline prompt (issue body + orientation
        # + injected fleet memory + ~40K startup.json route blob) or every worker is born
        # over-budget and crashes on turn 1 — the 2026-07-05 (#2972) regression that a
        # `== "48000"` check here protected. Pin a floor so a future baseline-growth
        # commit fails HERE, loudly, instead of silently crash-looping the fleet.
        budget = int(wrapped[wrapped.index("--context-budget-tokens") + 1])
        WORKER_BASELINE_FLOOR_TOKENS = 62400
        self.assertGreaterEqual(
            budget, WORKER_BASELINE_FLOOR_TOKENS,
            f"context budget {budget} < baseline floor {WORKER_BASELINE_FLOOR_TOKENS}: "
            "workers would be born over-budget on turn 1 (see #2972)")
        self.assertIn("--restart-on-budget", wrapped)
        # 16 (not 2): the old finite-but-tiny limit killed healthy workers after ~2
        # compaction epochs (~4-5 min), far short of their 30-min wall-clock backstop.
        self.assertEqual(wrapped[wrapped.index("--restart-limit") + 1], "16")
        # Graceful in-guard wall-clock backstop fronting the raised restart limit:
        # drains at DEFAULT_TIMEOUT_S - 60 before launch()'s hard-kill at 1800s.
        self.assertIn("--max-duration", wrapped)
        self.assertEqual(wrapped[wrapped.index("--max-duration") + 1], "1740s")
        # The COMPACT shed-line (#4253): raised above the ~62K baseline so compaction can
        # actually shed and the ACTIVE_COMPACT_RUNAWAY hold stops arming. MUST be > the
        # baseline floor (a shed-line at/below baseline is the 48K default that wedges the
        # dispatcher) and <= the drain ceiling above it.
        self.assertIn("--compact-history-budget", wrapped)
        shed = int(wrapped[wrapped.index("--compact-history-budget") + 1])
        self.assertGreater(
            shed, WORKER_BASELINE_FLOOR_TOKENS,
            f"compact shed-line {shed} <= baseline floor {WORKER_BASELINE_FLOOR_TOKENS}: "
            "worker can never shed under it and stays permanently past-compact (#4253)")
        self.assertLessEqual(shed, budget, "shed-line must sit at/under the drain ceiling")
        self.assertIn("--audit", wrapped)
        audit = Path(wrapped[wrapped.index("--audit") + 1])
        self.assertEqual(wrapped[wrapped.index("--session-id") + 1], audit.stem)
        # The raw worker argv is preserved verbatim AFTER the `--` separator.
        sep = wrapped.index("--")
        self.assertEqual(wrapped[sep + 1:], raw)

    def test_claude_guard_context_budget_derivation_matches_go(self) -> None:
        # Python half of the hand-maintained Go<->Python parity: mirror of
        # cmd/dispatchworker/guard_test.go:TestClaudeGuardContextBudgetDerivation.
        # Go derives from internal/ctxplan; Python cannot import it, so the four
        # module constants are hand-mirrored and THIS golden is the drift tripwire
        # (the exact-value lock the old `== "48000"` pin provided, restored at the
        # derived value). Update this test and the Go one IN THE SAME COMMIT.
        mod = load()
        derived = mod.claude_guard_context_budget_tokens()
        ceiling = (mod.CLAUDE_GUARD_MODEL_WINDOW_TOKENS
                   - mod.CLAUDE_GUARD_OUTPUT_RESERVE_TOKENS)
        # (a) Birth-safety: strictly above the baseline (a worker is never born
        # over-budget; see #2972).
        self.assertGreater(
            derived, mod.CLAUDE_GUARD_BASELINE_TOKENS,
            f"derived budget {derived} <= baseline: workers born over-budget")
        # (b) TURN funding -- the invariant the old goldens were missing. The seeded
        # budget is a CUMULATIVE allowance debited one FULL resident window per turn,
        # so the worst-case per-turn debit is the window ceiling and budget/per_turn is
        # the number of turns a child is guaranteed. It must be a whole epoch, not 2.
        per_turn = max(ceiling, mod.CLAUDE_GUARD_BASELINE_TOKENS)
        self.assertGreaterEqual(
            derived // per_turn, mod.CLAUDE_GUARD_TURNS_PER_EPOCH,
            f"derived budget {derived} funds only {derived // per_turn} full-window "
            f"turns (per_turn {per_turn}): a worker that cannot run an epoch dies "
            "BUDGET_CONTEXT_EXHAUSTED mid-issue")
        # (c) REGRESSION TRIPWIRE: the window ceiling is a PER-TURN quantity and must
        # never clamp the cumulative total again. Clamping is what pinned every child
        # at ~2 turns (min(62000*k, 168000) = 168000 for all k >= 3, so no headroom
        # factor could ever help) and produced 120/120 CLAIM_NO_COMMIT witnesses.
        self.assertGreater(
            derived, ceiling,
            f"derived budget {derived} <= per-turn window ceiling {ceiling}: the "
            "cumulative allowance has been clamped to a per-turn window again")
        # (d) Golden lock: max(200000-32000, 62000) * 12 = 2016000, and the argv
        # string constant wired into CLAUDE_GUARD_BUDGET_ARGS carries it. If this
        # fails, a mirror constant diverged from Go (or Go moved) -- keep the two
        # goldens identical in the same commit.
        self.assertEqual(
            derived, 2016000,
            "derived budget diverged from Go's TestClaudeGuardContextBudgetDerivation "
            "golden; re-sync the mirrored constants")
        self.assertEqual(mod.CLAUDE_GUARD_CONTEXT_BUDGET_TOKENS, "2016000")
        # (e) Non-decreasing in the baseline, and strictly rising once the baseline
        # outgrows the window ceiling (the flat-constant staleness the derivation
        # kills). load() gives a fresh module per test, so patching globals is
        # hermetic.
        mod.CLAUDE_GUARD_BASELINE_TOKENS += 1000
        self.assertGreaterEqual(mod.claude_guard_context_budget_tokens(), derived)
        mod.CLAUDE_GUARD_BASELINE_TOKENS = ceiling + 1000
        self.assertGreater(mod.claude_guard_context_budget_tokens(), derived)

    def test_claude_guard_compact_history_budget(self) -> None:
        # Python half of the compact shed-line parity (#4253). Mirror of
        # cmd/dispatchworker/guard_test.go:TestClaudeGuardCompactHistoryBudget. The
        # Go golden pins this == gateway.HeadlessCompactHistoryBudget; Python cannot
        # import Go, so keep the integer identical here by hand.
        mod = load()
        shed = mod.CLAUDE_GUARD_COMPACT_HISTORY_BUDGET
        # (a) Golden lock: the headless compact budget value (== Go mirror == gateway).
        self.assertEqual(
            shed, 96000,
            "compact shed-line diverged from Go's TestClaudeGuardCompactHistoryBudget "
            "golden / gateway.HeadlessCompactHistoryBudget; re-sync the mirrors")
        # (b) Strictly above the baseline — a shed-line at/below baseline (the 48K
        # default) can never succeed and pins the worker permanently past compact.
        self.assertGreater(
            shed, mod.CLAUDE_GUARD_BASELINE_TOKENS,
            f"compact shed-line {shed} <= baseline: worker stays past-compact")
        # (c) REACHABILITY, not "below the drain ceiling" -- the two are on different
        # scales (this shed-line is a PER-TURN instantaneous target,
        # --context-budget-tokens is a CUMULATIVE allowance), so comparing them
        # directly is a category error and was the assertion that let the composition
        # ship broken. What must hold is that a worker SITTING AT the shed line still
        # has a whole epoch of turns funded; otherwise the session dies of budget
        # exhaustion long before compaction can fire (witnessed as compact=none /
        # "bailed: under_budget" on every turn while the cumulative budget drained).
        drain = int(mod.claude_guard_context_budget_tokens())
        self.assertGreaterEqual(
            drain // shed, mod.CLAUDE_GUARD_TURNS_PER_EPOCH,
            f"drain budget {drain} funds only {drain // shed} turns at the shed-line "
            f"resident {shed}: compaction is unreachable before "
            "BUDGET_CONTEXT_EXHAUSTED")
        # (d) It is actually WIRED into the claude guard argv (not just a constant).
        args = mod.claude_guard_budget_args()
        self.assertIn("--compact-history-budget", args)
        self.assertEqual(args[args.index("--compact-history-budget") + 1], str(shed))

    def test_claude_guard_compact_solvency_floor(self) -> None:
        # Python half of the context-solvency floor parity. Mirror of
        # cmd/dispatchworker/guard_test.go:TestClaudeGuardSolvencyFloorDerivation --
        # same integers, same argv position. Update both in the same commit.
        mod = load()
        usable = (mod.CLAUDE_GUARD_MODEL_WINDOW_TOKENS
                  - mod.CLAUDE_GUARD_OUTPUT_RESERVE_TOKENS)
        floor = mod.claude_guard_compact_solvency_floor_tokens()
        # (a) Golden lock: 85% of (200000-32000).
        self.assertEqual(
            floor, 142800,
            "solvency floor diverged from Go's "
            "TestClaudeGuardSolvencyFloorDerivation golden; re-sync the mirrors")
        # (b) STRICTLY above the compact shed-line. A floor at or below it would force a
        # fire on essentially every past-budget turn and discard the cache economics
        # wholesale -- the override is a last resort, not a replacement for the gate.
        self.assertGreater(
            floor, mod.CLAUDE_GUARD_COMPACT_HISTORY_BUDGET,
            f"solvency floor {floor} <= compact shed-line "
            f"{mod.CLAUDE_GUARD_COMPACT_HISTORY_BUDGET}: the override would swallow "
            "the burst gate entirely")
        # (c) STRICTLY below the usable window, with real headroom for the forced burst
        # to land and repay. A floor at the ceiling rings the alarm after the wall.
        self.assertLess(floor, usable)
        self.assertGreaterEqual(
            usable - floor, 20000,
            f"solvency floor {floor} leaves only {usable - floor} tokens of usable "
            "window; want >= 20000 for the forced burst to land")
        # (d) Fail-safe: a degenerate envelope DISARMS the override (0) rather than
        # forcing a fire every turn. 0 is the documented "pure economics" value.
        self.assertEqual(mod.derive_claude_guard_solvency_floor(1000, 4000), 0)
        # (e) Model-aware arithmetic: a smaller window derives its own lower floor.
        small = mod.derive_claude_guard_solvency_floor(64000, 32000)
        self.assertTrue(0 < small < floor)
        # (f) It is actually WIRED into the claude guard argv (not just a constant).
        args = mod.claude_guard_budget_args()
        self.assertIn("--compact-solvency-floor", args)
        self.assertEqual(
            args[args.index("--compact-solvency-floor") + 1], str(floor))

    def test_measure_launch_baseline_floors_and_tracks(self) -> None:
        # Python half of the measurement-seam parity (#3522). Mirror of
        # cmd/dispatchworker/guard_test.go:TestMeasureLaunchBaselineFloorsAndTracks —
        # same fixtures, same integers. Update both in the same commit.
        mod = load()
        # (a) approx ruler matches the codebase (bytes+3)//4.
        self.assertEqual(mod.approx_tokens_from_bytes(41657), (41657 + 3) // 4)
        self.assertEqual(mod.approx_tokens_from_bytes(0), 0)
        # (b) A degenerate (empty) measurement floors to the shipped baseline.
        self.assertEqual(
            mod.resolve_claude_guard_baseline(mod.measure_launch_baseline_tokens({})),
            mod.CLAUDE_GUARD_BASELINE_TOKENS)
        # (c) A sub-floor measured prompt still floors — no regression.
        small = {"AGENTS.md": 41657, "llms.txt": 57230}
        self.assertEqual(
            mod.resolve_claude_guard_baseline(mod.measure_launch_baseline_tokens(small)),
            mod.CLAUDE_GUARD_BASELINE_TOKENS)
        # (d) A prompt that OUTGROWS the floor raises the baseline (the trap the frozen
        # constant left open).
        grown = {"AGENTS.md": 41657, "llms.txt": 57230, "startup_bundle": 200000}
        measured = mod.measure_launch_baseline_tokens(grown)
        self.assertGreater(measured, mod.CLAUDE_GUARD_BASELINE_TOKENS)
        self.assertEqual(mod.resolve_claude_guard_baseline(measured), measured)
        # (e) Live gather reads a temp workspace and folds a startup bundle named via env;
        # an empty workspace measures nothing (the hermetic default → the floor).
        with tempfile.TemporaryDirectory() as d:
            ws = Path(d)
            (ws / "AGENTS.md").write_text("a" * 400, encoding="utf-8")
            (ws / "llms.txt").write_text("b" * 800, encoding="utf-8")
            bundle = ws / "run.startup.json"
            bundle.write_text("c" * 1200, encoding="utf-8")
            got = mod.gather_launch_constituent_bytes(
                ws, {mod.LAUNCH_STARTUP_BUNDLE_ENV: str(bundle)})
            self.assertEqual(got.get("AGENTS.md"), 400)
            self.assertEqual(got.get("llms.txt"), 800)
            self.assertEqual(got.get("startup_bundle"), 1200)
            self.assertNotIn("MEMORY.md", got)
        self.assertEqual(mod.gather_launch_constituent_bytes("", None), {})
        base, _ = mod.measured_claude_guard_baseline("", None)
        self.assertEqual(base, mod.CLAUDE_GUARD_BASELINE_TOKENS)

    def test_guard_wrap_codex_disables_nested_loop_gate(self) -> None:
        mod = load()
        raw = ["codex", "exec", "ship issue"]
        env = {"FLEET_DOGFOOD_GUARD_BASEURL": "http://127.0.0.1:18080/v1"}
        wrapped = mod.guard_wrap(raw, fak_bin="/usr/bin/fak", lane="loop",
                                 backend="codex", workspace=ROOT, env=env)
        self.assertEqual(wrapped[wrapped.index("--provider") + 1], "openai-responses")
        self.assertIn("--codex-loop-gate", wrapped)
        gate = wrapped.index("--codex-loop-gate")
        self.assertEqual(wrapped[gate + 1], "off")
        self.assertNotIn("--base-url", wrapped)
        self.assertEqual(
            wrapped[wrapped.index("--") + 1:],
            ["codex", "-c", "model_auto_compact_token_limit=96000",
             "exec", "ship issue"],
        )

    def test_guard_wrap_codex_injects_native_compact_limit(self) -> None:
        mod = load()
        raw = ["codex", "exec", "work"]
        wrapped = mod.guard_wrap(
            raw, fak_bin="/usr/bin/fak", lane="gateway", backend="codex",
            workspace=Path("."), env={})
        self.assertIn("--", wrapped)
        child = wrapped[wrapped.index("--") + 1:]
        self.assertEqual(
            child,
            ["codex", "-c", "model_auto_compact_token_limit=96000",
             "exec", "work"],
        )

    def test_guard_wrap_noop_without_fak_bin(self) -> None:
        mod = load()
        raw = mod.build_command("docs", "claude")
        self.assertEqual(
            mod.guard_wrap(raw, fak_bin=None, lane="docs", backend="claude",
                           workspace=Path(".")), raw)

    def test_guard_wrap_opencode_skips_without_base_url_but_wraps_with_one(self) -> None:
        mod = load()
        raw = mod.build_command("recall", "opencode")
        # No FLEET_DOGFOOD_GUARD_BASEURL -> refuse to misroute a local-upstream worker.
        self.assertEqual(
            mod.guard_wrap(raw, fak_bin="/usr/bin/fak", lane="recall",
                           backend="opencode", workspace=Path("."), env={}), raw)
        # With a base URL the operator names the local upstream and we DO front it.
        wrapped = mod.guard_wrap(raw, fak_bin="/usr/bin/fak", lane="recall",
                                 backend="opencode", workspace=Path("."),
                                 env={"FLEET_DOGFOOD_GUARD_BASEURL": "http://127.0.0.1:8131/v1"})
        self.assertEqual(wrapped[0], "/usr/bin/fak")
        self.assertEqual(wrapped[wrapped.index("--provider") + 1], "openai")
        self.assertEqual(wrapped[wrapped.index("--base-url") + 1], "http://127.0.0.1:8131/v1")

    def test_guard_wrap_opencode_injects_inline_provider_repoint(self) -> None:
        mod = load()
        raw = ["opencode", "run", "-m", "zai-coding-plan/glm-5.2", "dispatch"]
        env = {
            "FLEET_DOGFOOD_GUARD_BASEURL": "https://api.example.test/v1",
            "FLEET_DOGFOOD_GUARD_ADDR": "127.0.0.1:8137",
            mod.OPENCODE_GUARD_UPSTREAM_KEY_ENV: "secret-test-key",
            "OPENCODE_CONFIG_CONTENT": '{"autoupdate":false}',
        }

        wrapped = mod.guard_wrap(raw, fak_bin="/usr/bin/fak", lane="recall",
                                 backend="opencode", workspace=Path("."), env=env)

        self.assertEqual(wrapped[wrapped.index("--addr") + 1], "127.0.0.1:8137")
        self.assertEqual(wrapped[wrapped.index("--api-key-env") + 1],
                         mod.OPENCODE_GUARD_UPSTREAM_KEY_ENV)
        self.assertNotIn("secret-test-key", wrapped)
        cfg = json.loads(env["OPENCODE_CONFIG_CONTENT"])
        self.assertFalse(cfg["autoupdate"])
        self.assertEqual(
            cfg["provider"]["zai-coding-plan"]["options"]["baseURL"],
            "http://127.0.0.1:8137/v1")

    def test_opencode_inline_repoint_defaults_to_glm_provider(self) -> None:
        mod = load()

        cfg = json.loads(mod.opencode_guard_config_content(
            ["opencode", "run", "dispatch"], "http://127.0.0.1:8138/v1"))

        self.assertEqual(
            cfg["provider"]["zai-coding-plan"]["options"]["baseURL"],
            "http://127.0.0.1:8138/v1")

    def test_opencode_compact_shed_line(self) -> None:
        # Python half of the OpenCode-native compact shed-line parity (#4661).
        # Mirror of cmd/dispatchworker TestOpencodeCompactShedLine; keep both in step.
        mod = load()
        shed = mod.OPENCODE_COMPACT_SHED_LINE_TOKENS

        # (a) Golden lock: the same 96K headless target the claude/codex arms use.
        self.assertEqual(
            shed, 96000,
            "opencode shed line diverged from Go's TestOpencodeCompactShedLine "
            "and the 96K headless target (#4661)")
        self.assertEqual(shed, mod.CLAUDE_GUARD_COMPACT_HISTORY_BUDGET)
        self.assertEqual(shed, mod.CODEX_COMPACT_TOKEN_LIMIT)

        overlay = mod.opencode_compaction_overlay(
            ["opencode", "run", "-m", "zai-coding-plan/glm-5.2", "dispatch"])

        # (b) The knob that actually moves OpenCode's trigger. OpenCode compacts at
        # `limit.input - compaction.reserved` when limit.input is declared, so THIS
        # is the invariant that makes the shed line real rather than decorative.
        limit = overlay["provider"]["zai-coding-plan"]["models"]["glm-5.2"]["limit"]
        self.assertEqual(limit["input"] - overlay["compaction"]["reserved"], shed,
                         "derived opencode compaction trigger must land ON the shed line")
        self.assertTrue(overlay["compaction"]["auto"])

        # (c) The real window/output are carried verbatim: opencode's schema requires
        # context+output beside input, and a wrong output would mis-cap generation.
        self.assertEqual(limit["context"], 1000000)
        self.assertEqual(limit["output"], 131072)
        # It must LOWER the trigger, not postpone overflow: the un-overridden trigger
        # for this model is context - output (no native limit.input), i.e. ~869K.
        self.assertLess(shed, limit["context"] - limit["output"])

        # (d) The dispatch argv passes no -m, so the account default must still be
        # covered -- every model of the resolved provider carries the shed line.
        default_argv = mod.build_command("recall", "opencode")
        self.assertNotIn("-m", default_argv)
        models = mod.opencode_compaction_overlay(default_argv)[
            "provider"]["zai-coding-plan"]["models"]
        self.assertIn("glm-5.2", models)
        for name, spec in models.items():
            self.assertEqual(spec["limit"]["input"], shed, f"{name} missed the shed line")

        # (e) Fail OPEN on an unknown provider: a partial/absent limit block makes
        # opencode refuse to start, which would kill the worker at launch.
        self.assertEqual(
            mod.opencode_compaction_overlay(
                ["opencode", "run", "-m", "someone-else/mystery-1", "dispatch"]),
            {})

    def test_guard_wrap_opencode_injects_compact_shed_line(self) -> None:
        # End-to-end through the real launcher seam: the shed line rides the SAME
        # per-child OPENCODE_CONFIG_CONTENT env the base-URL repoint uses (#4661).
        mod = load()
        raw = ["opencode", "run", "-m", "zai-coding-plan/glm-5.2", "dispatch"]
        env = {
            "FLEET_DOGFOOD_GUARD_BASEURL": "https://api.example.test/v1",
            "FLEET_DOGFOOD_GUARD_ADDR": "127.0.0.1:8137",
            mod.OPENCODE_GUARD_UPSTREAM_KEY_ENV: "secret-test-key",
            "OPENCODE_CONFIG_CONTENT": '{"autoupdate":false}',
        }

        mod.guard_wrap(raw, fak_bin="/usr/bin/fak", lane="recall",
                       backend="opencode", workspace=Path("."), env=env)

        cfg = json.loads(env["OPENCODE_CONFIG_CONTENT"])
        # Pre-existing operator config and the base-URL repoint both survive the merge.
        self.assertFalse(cfg["autoupdate"])
        self.assertEqual(
            cfg["provider"]["zai-coding-plan"]["options"]["baseURL"],
            "http://127.0.0.1:8137/v1")
        # ...and the shed line is present in the same document.
        self.assertEqual(
            cfg["provider"]["zai-coding-plan"]["models"]["glm-5.2"]["limit"]["input"],
            mod.OPENCODE_COMPACT_SHED_LINE_TOKENS)
        self.assertEqual(cfg["compaction"]["reserved"], 0)

    def test_opencode_upstream_key_reads_account_config(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / "opencode" / "opencode.json"
            cfg.parent.mkdir(parents=True)
            cfg.write_text(json.dumps({
                "provider": {
                    "zai-coding-plan": {
                        "options": {"apiKey": "secret-from-config"}
                    }
                }
            }), encoding="utf-8")

            got = mod.opencode_upstream_api_key(
                ["opencode", "run", "-m", "zai-coding-plan/glm-5.2"],
                {"XDG_CONFIG_HOME": d})

        self.assertEqual(got, "secret-from-config")

    def test_opencode_upstream_base_url_reads_account_config(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / "opencode" / "opencode.json"
            cfg.parent.mkdir(parents=True)
            cfg.write_text(json.dumps({
                "provider": {
                    "deepseek-ai": {
                        "options": {
                            "baseURL": "https://integrate.api.nvidia.com/v1",
                            "apiKey": "{env:NVIDIA_TEST_KEY}",
                        }
                    }
                }
            }), encoding="utf-8")

            got = mod.opencode_upstream_base_url(
                ["opencode", "run", "-m", "deepseek-ai/deepseek-v4-pro"],
                {"XDG_CONFIG_HOME": d})

        self.assertEqual(got, "https://integrate.api.nvidia.com/v1")

    def test_opencode_upstream_key_reads_inline_config_content(self) -> None:
        mod = load()

        got = mod.opencode_upstream_api_key(
            ["opencode", "run", "-m", "zai-coding-plan/glm-5.2"],
            {"OPENCODE_CONFIG_CONTENT": json.dumps({
                "provider": {
                    "zai-coding-plan": {
                        "options": {"apiKey": "{env:ZAI_TEST_KEY}"}
                    }
                }
            }), "ZAI_TEST_KEY": "secret-from-inline-env"})

        self.assertEqual(got, "secret-from-inline-env")

    def test_guard_wrap_opencode_non_default_provider_config_beats_global_glm_base(self) -> None:
        mod = load()
        raw = ["opencode", "run", "-m", "deepseek-ai/deepseek-v4-pro", "dispatch"]
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / "opencode" / "opencode.json"
            cfg.parent.mkdir(parents=True)
            cfg.write_text(json.dumps({
                "provider": {
                    "deepseek-ai": {
                        "options": {
                            "baseURL": "https://integrate.api.nvidia.com/v1",
                            "apiKey": "{env:NVIDIA_TEST_KEY}",
                        }
                    }
                }
            }), encoding="utf-8")
            env = {
                "XDG_CONFIG_HOME": d,
                "NVIDIA_TEST_KEY": "secret-nim-key",
                "FLEET_DOGFOOD_GUARD_BASEURL": "http://127.0.0.1:18080/v1",
                "FLEET_DOGFOOD_GUARD_ADDR": "127.0.0.1:8139",
            }

            wrapped = mod.guard_wrap(raw, fak_bin="/usr/bin/fak", lane="docs",
                                     backend="opencode", workspace=Path("."), env=env)

        self.assertEqual(wrapped[wrapped.index("--base-url") + 1],
                         "https://integrate.api.nvidia.com/v1")
        self.assertEqual(wrapped[wrapped.index("--api-key-env") + 1],
                         mod.OPENCODE_GUARD_UPSTREAM_KEY_ENV)
        self.assertNotIn("secret-nim-key", wrapped)
        cfg_overlay = json.loads(env["OPENCODE_CONFIG_CONTENT"])
        self.assertEqual(
            cfg_overlay["provider"]["deepseek-ai"]["options"]["baseURL"],
            "http://127.0.0.1:8139/v1")

    def test_guard_wrap_opencode_non_default_provider_ignores_global_glm_base_without_config(self) -> None:
        mod = load()
        raw = ["opencode", "run", "-m", "deepseek-ai/deepseek-v4-pro", "dispatch"]
        env = {
            mod.OPENCODE_GUARD_BASE_URL_ENV: "http://127.0.0.1:18080/v1",
            "FLEET_DOGFOOD_GUARD_ADDR": "127.0.0.1:8140",
        }

        wrapped = mod.guard_wrap(raw, fak_bin="/usr/bin/fak", lane="docs",
                                 backend="opencode", workspace=Path("."), env=env)

        self.assertEqual(wrapped, raw)
        self.assertNotIn("OPENCODE_CONFIG_CONTENT", env)

    def test_guard_wrap_opencode_provider_specific_base_url_beats_account_and_global(self) -> None:
        mod = load()
        raw = ["opencode", "run", "-m", "deepseek-ai/deepseek-v4-pro", "dispatch"]
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / "opencode" / "opencode.json"
            cfg.parent.mkdir(parents=True)
            cfg.write_text(json.dumps({
                "provider": {
                    "deepseek-ai": {
                        "options": {
                            "baseURL": "https://integrate.api.nvidia.com/v1",
                            "apiKey": "{env:NVIDIA_TEST_KEY}",
                        }
                    }
                }
            }), encoding="utf-8")
            env = {
                "XDG_CONFIG_HOME": d,
                "NVIDIA_TEST_KEY": "secret-nim-key",
                mod.OPENCODE_GUARD_BASE_URL_ENV: "http://127.0.0.1:18080/v1",
                f"{mod.OPENCODE_GUARD_BASE_URL_ENV}_DEEPSEEK_AI": "http://dgx.local:8000/v1",
                "FLEET_DOGFOOD_GUARD_ADDR": "127.0.0.1:8141",
            }

            wrapped = mod.guard_wrap(raw, fak_bin="/usr/bin/fak", lane="docs",
                                     backend="opencode", workspace=Path("."), env=env)

        self.assertEqual(wrapped[wrapped.index("--base-url") + 1],
                         "http://dgx.local:8000/v1")
        cfg_overlay = json.loads(env["OPENCODE_CONFIG_CONTENT"])
        self.assertEqual(
            cfg_overlay["provider"]["deepseek-ai"]["options"]["baseURL"],
            "http://127.0.0.1:8141/v1")

    def test_guard_wrap_opencode_default_provider_config_beats_legacy_global_glm_override(self) -> None:
        mod = load()
        raw = ["opencode", "run", "-m", "zai-coding-plan/glm-5.2", "dispatch"]
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / "opencode" / "opencode.json"
            cfg.parent.mkdir(parents=True)
            cfg.write_text(json.dumps({
                "provider": {
                    "zai-coding-plan": {
                        "options": {
                            "baseURL": "https://api.z.ai/api/coding/paas/v4",
                            "apiKey": "{env:ZAI_TEST_KEY}",
                        }
                    }
                }
            }), encoding="utf-8")
            env = {
                "XDG_CONFIG_HOME": d,
                "ZAI_TEST_KEY": "secret-zai-key",
                mod.OPENCODE_GUARD_BASE_URL_ENV: "http://127.0.0.1:18080/v1",
                "FLEET_DOGFOOD_GUARD_ADDR": "127.0.0.1:8142",
            }

            wrapped = mod.guard_wrap(raw, fak_bin="/usr/bin/fak", lane="glm",
                                     backend="opencode", workspace=Path("."), env=env)

        self.assertEqual(wrapped[wrapped.index("--base-url") + 1],
                         "https://api.z.ai/api/coding/paas/v4")

    def test_guard_wrap_opencode_default_provider_specific_base_url_beats_account_config(self) -> None:
        mod = load()
        raw = ["opencode", "run", "-m", "zai-coding-plan/glm-5.2", "dispatch"]
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / "opencode" / "opencode.json"
            cfg.parent.mkdir(parents=True)
            cfg.write_text(json.dumps({
                "provider": {
                    "zai-coding-plan": {
                        "options": {
                            "baseURL": "https://api.z.ai/api/coding/paas/v4",
                            "apiKey": "{env:ZAI_TEST_KEY}",
                        }
                    }
                }
            }), encoding="utf-8")
            env = {
                "XDG_CONFIG_HOME": d,
                "ZAI_TEST_KEY": "secret-zai-key",
                f"{mod.OPENCODE_GUARD_BASE_URL_ENV}_ZAI_CODING_PLAN": "http://127.0.0.1:18080/v1",
                "FLEET_DOGFOOD_GUARD_ADDR": "127.0.0.1:8143",
            }

            wrapped = mod.guard_wrap(raw, fak_bin="/usr/bin/fak", lane="glm",
                                     backend="opencode", workspace=Path("."), env=env)

        self.assertEqual(wrapped[wrapped.index("--base-url") + 1],
                         "http://127.0.0.1:18080/v1")

    def test_opencode_upstream_key_reads_explicit_config_path(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / "custom-opencode.json"
            cfg.write_text(json.dumps({
                "provider": {
                    "zai-coding-plan": {
                        "options": {"apiKey": "secret-from-explicit-config"}
                    }
                }
            }), encoding="utf-8")

            got = mod.opencode_upstream_api_key(
                ["opencode", "run", "-m", "zai-coding-plan/glm-5.2"],
                {"OPENCODE_CONFIG": str(cfg)})

        self.assertEqual(got, "secret-from-explicit-config")

    def test_guarded_launch_command_opts_out_when_disabled(self) -> None:
        mod = load()
        raw = mod.build_command("gateway", "claude")
        cmd, guarded = mod.guarded_launch_command(
            raw, "gateway", "claude", Path("C:/work/fak"),
            env={"FLEET_DOGFOOD_GUARD": "0", "FAK_BIN": str(Path(__file__).resolve())})
        self.assertFalse(guarded)
        self.assertEqual(cmd, raw)

    def test_guarded_launch_command_wraps_when_enabled_and_bin_present(self) -> None:
        mod = load()
        raw = mod.build_command("gateway", "claude")
        fak = str(Path(__file__).resolve())  # any existing file stands in for the bin
        cmd, guarded = mod.guarded_launch_command(
            raw, "gateway", "claude", Path("C:/work/fak"), env={"FAK_BIN": fak})
        self.assertTrue(guarded)
        self.assertEqual(cmd[0], fak)
        self.assertEqual(cmd[1], "guard")

    def test_guard_env_augment_sets_timeout_floors_without_clobbering(self) -> None:
        mod = load()
        env = {"FAK_PLANNER_TIMEOUT_S": "1200"}
        mod.guard_env_augment(env)
        self.assertEqual(env["FAK_PLANNER_TIMEOUT_S"], "1200")   # explicit value kept
        self.assertEqual(env["FAK_HTTP_WRITE_TIMEOUT_S"], str(mod.GUARD_TIMEOUT_FLOOR_S))

    def test_build_payload_carries_guarded_and_explicit_command(self) -> None:
        mod = load()
        payload = mod.build_payload(
            lane="gateway", backend="claude", workspace=Path("C:/work/fak"),
            dry_run=True, command=["fak", "guard", "--", "claude"], guarded=True)
        self.assertTrue(payload["guarded"])
        self.assertEqual(payload["command"][0], "fak")

    # --- Shared-path leasing: one gateway front door for a wave (#1501) -----------
    def test_shared_gateway_url_reads_and_trims(self) -> None:
        mod = load()
        self.assertEqual(mod.shared_gateway_url({}), "")
        self.assertEqual(
            mod.shared_gateway_url({mod.SHARED_GATEWAY_URL_ENV: " http://127.0.0.1:8080/ "}),
            "http://127.0.0.1:8080")

    def test_shared_fak_mcp_config_is_guard_http_shape_at_mcp(self) -> None:
        mod = load()
        cfg = mod.shared_fak_mcp_config("http://10.0.0.5:8080")
        # Mirrors guardMCPClientConfig: one server "fak", remote http, /mcp endpoint.
        self.assertEqual(cfg, {"mcpServers": {"fak": {
            "type": "http", "url": "http://10.0.0.5:8080/mcp"}}})
        # Idempotent on a trailing slash.
        self.assertEqual(mod.shared_fak_mcp_url("http://10.0.0.5:8080/"),
                         "http://10.0.0.5:8080/mcp")

    def test_child_env_stamps_shared_gateway_when_set(self) -> None:
        mod = load()
        base = {"PATH": "x", mod.SHARED_GATEWAY_URL_ENV: "http://127.0.0.1:8080/"}
        env = mod.child_env("tools", "claude", Path("C:/work/fak"), base=base)
        self.assertEqual(env["DISPATCH_SHARED_GATEWAY"], "http://127.0.0.1:8080")
        # Absent when the wave names no shared front door -> worker keeps its own.
        env2 = mod.child_env("tools", "claude", Path("C:/work/fak"), base={"PATH": "x"})
        self.assertNotIn("DISPATCH_SHARED_GATEWAY", env2)

    def test_guard_wrap_claude_no_shared_gateway_keeps_private_and_raw(self) -> None:
        mod = load()
        raw = mod.build_command("tools", "claude")
        wrapped = mod.guard_wrap(raw, fak_bin="/usr/bin/fak", lane="tools",
                                 backend="claude", workspace=Path("C:/work/fak"), env={})
        # No shared front door -> guard's own per-session MCP registration is left ON
        # (no --mcp-register=false) and the raw worker argv is untouched after `--`.
        self.assertNotIn("--mcp-register=false", wrapped)
        self.assertNotIn("--mcp-config", wrapped)
        sep = wrapped.index("--")
        self.assertEqual(wrapped[sep + 1:], raw)

    def test_guard_wrap_claude_shared_gateway_repoints_fak_mcp(self) -> None:
        mod = load()
        raw = mod.build_command("tools", "claude")
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            wrapped = mod.guard_wrap(
                raw, fak_bin="/usr/bin/fak", lane="tools", backend="claude",
                workspace=ws, env={mod.SHARED_GATEWAY_URL_ENV: "http://127.0.0.1:8080"})
            # Guard's private per-session MCP registration is disabled so it cannot
            # override the shared front door with this worker's own gateway.
            self.assertIn("--mcp-register=false", wrapped)
            # "fak" is registered at the SHARED serve's /mcp via Claude Code's
            # --mcp-config, inserted into the claude argv after the `--` separator.
            self.assertIn("--mcp-config", wrapped)
            cfg_path = Path(wrapped[wrapped.index("--mcp-config") + 1])
            self.assertTrue(cfg_path.exists())
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            self.assertEqual(cfg["mcpServers"]["fak"]["url"], "http://127.0.0.1:8080/mcp")
            self.assertEqual(cfg["mcpServers"]["fak"]["type"], "http")
            # The --mcp-config lands in the wrapped worker argv (after `--`), not the
            # guard args (before `--`).
            sep = wrapped.index("--")
            self.assertIn("--mcp-config", wrapped[sep + 1:])
            self.assertEqual(wrapped[sep + 1], "claude")


class NoWindowSubprocessDefaultsTest(unittest.TestCase):
    """`install_no_window_subprocess_defaults` patches a module's subprocess helpers to
    default `creationflags`. The KIND of each patched attribute has to survive that."""

    class _FakePopen:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    def setUp(self) -> None:
        self.m = load()

    def _fake_subprocess(self):
        mod = types.SimpleNamespace()
        for name in ("run", "call", "check_call", "check_output"):
            mod.__dict__[name] = (lambda n: (lambda *a, **k: (n, a, k)))(name)
        mod.Popen = self._FakePopen
        return mod

    def _install(self, mod) -> None:
        # The installer is a no-op off Windows, so force the branch: the invariant below
        # is pinned on every host rather than only on the one where it can regress.
        orig = self.m.os.name
        self.m.os.name = "nt"
        try:
            self.m.install_no_window_subprocess_defaults(mod)
        finally:
            self.m.os.name = orig

    def test_popen_stays_a_subclassable_class(self) -> None:
        mod = self._fake_subprocess()
        self._install(mod)
        self.assertTrue(inspect.isclass(mod.Popen), "Popen must stay a class, not a function")
        # The exact shape the stdlib uses: asyncio.windows_utils does
        # `class Popen(subprocess.Popen)` at import time, and unittest.mock imports
        # asyncio -- so a function here costs the whole process `from unittest import
        # mock`, with a traceback pointing at asyncio rather than at this module.
        class Derived(mod.Popen):
            pass

        self.assertTrue(issubclass(Derived, self._FakePopen))

    def test_popen_still_defaults_creationflags(self) -> None:
        mod = self._fake_subprocess()
        self._install(mod)
        p = mod.Popen(["git", "status"])
        self.assertEqual(p.kwargs.get("creationflags"), self.m.no_window_creationflags())
        self.assertEqual(p.args, (["git", "status"],))

    def test_explicit_creationflags_still_win(self) -> None:
        mod = self._fake_subprocess()
        self._install(mod)
        # CREATE_NEW_PROCESS_GROUP: a value no default would produce on either host, so
        # this stays a real assertion on POSIX where the default flag is 0.
        p = mod.Popen(["git", "status"], creationflags=0x200)
        self.assertEqual(p.kwargs.get("creationflags"), 0x200)

    def test_asyncio_still_imports_after_install(self) -> None:
        # The end-to-end witness for the regression: install the defaults on the REAL
        # subprocess module inside a child process, then import the thing that subclasses
        # Popen. A child keeps the (idempotent, process-wide) patch out of this runner.
        prog = (
            "import importlib.util,subprocess,sys\n"
            f"sys.path.insert(0, {str(SCRIPT.parent)!r})\n"
            f"spec = importlib.util.spec_from_file_location('dw', {str(SCRIPT)!r})\n"
            "mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)\n"
            "mod.install_no_window_subprocess_defaults(subprocess)\n"
            "from unittest import mock\n"
            "print('ok')\n"
        )
        out = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("ok", out.stdout)


if __name__ == "__main__":
    unittest.main()
