// guard.go — front each dispatch worker with the kernel (`fak guard`), a Go port
// of the dogfood-guard family in tools/dispatch_worker.py.
//
// A dispatch worker IS the highest-volume dev work on a fleet node, and the LIVE
// production path is this compiled binary (dos.toml `worker_launch_template =
// 'tools/.bin/dispatchworker --lane {lane}'`, preferred over the Python sibling).
// Before this file the Go path talked STRAIGHT to the provider API — the kernel
// adjudicated NONE of the concurrent dispatch fleet, even though the Python sibling
// had guarded-by-default since #... . That made "the dispatch fleet dogfoods fak
// guard" true only on a path nothing runs in production.
//
// Fronting the worker with `fak guard` puts the SAME kernel `fak serve` runs in
// front of every tool call the worker proposes (deny by structure, repair malformed
// args, quarantine poisoned results) and records every verdict in a durable,
// hash-chained DECISION JOURNAL under the gitignored .dispatch-runs/guard-audit/ —
// so the fleet eats the product on the real workflow, WITH a witness. Default ON;
// opt out per node with FLEET_DOGFOOD_GUARD=0. resolveFakBin fails OPEN to an
// unwrapped worker on a host that has not built `fak`, so the default never breaks
// dispatch.
//
// The pure functions mirror dispatch_worker.py 1:1 so the ported guard_test table is
// a parity witness; only the OS touches (process env, crypto/rand token) differ.
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"unicode"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
	"github.com/anthony-chaudhary/fak/internal/randhex"
)

// guardOffValues: a FLEET_DOGFOOD_GUARD whose normalized value is one of these turns
// dogfood guard OFF (mirrors dispatch_worker.GUARD_OFF_VALUES). Empty string is an
// explicit off so `FLEET_DOGFOOD_GUARD=` reads as a disable, not an unset.
var guardOffValues = map[string]struct{}{
	"0": {}, "off": {}, "false": {}, "no": {}, "": {}, "disable": {}, "disabled": {},
}

// guardTimeoutFloorS raises a guarded worker's gateway planner/write timeouts so a
// long frontier turn (extended thinking) is not TRUNCATED at fak serve's default
// 60s/90s floors. Mirrors dispatch_worker.GUARD_TIMEOUT_FLOOR_S.
const guardTimeoutFloorS = 600

// Claude guard args mirror tools/dispatch_worker.py. Headless workers need the
// PreCompact actuator in enforce mode, plus guard's hard budget restart path so
// a compact-runaway session is relaunched instead of only nudged in stderr.
//
// claudeGuardContextBudgetTokens seeds the guard's per-session ContextTokensLeft.
// SEMANTICS (internal/session/usage.go:99 DebitUsage): this is a CUMULATIVE
// allowance, NOT a per-turn ceiling. Every turn subtracts that turn's ENTIRE
// resident context window (internal/agent/chat.go ContextWindowTokens = prompt +
// cache_read + cache_creation — the whole window, not the newly-added delta), so a
// cached prefix is re-charged in full on every turn; when the running total reaches
// <=0 the session drains with BUDGET_CONTEXT_EXHAUSTED. Two consequences that the
// arithmetic below MUST respect:
//
//	turns funded per child ≈ budget / (mean resident tokens per turn)
//
// and the resident is never below the launch baseline, so a budget of baseline × k
// funds AT MOST k turns — under a cumulative meter the "headroom factor" IS the turn
// count, and it stays k no matter how large the baseline grows. A budget is birth-safe
// (> baseline) and still useless if it only funds two turns.
//
// The old 48000 was BELOW the workers' ~62K irreducible baseline prompt (issue body
// + AGENTS/llms orientation + injected fleet memory + the ~40K startup.json 'route'
// blob), so every dispatch claude worker exhausted on turn 1 → 2 restarts burned →
// raw 409 → child exit 1 → CHILD_CRASH (fleet ship rate collapsed to 2-9%).
//
// DONE(dynamic-budget): the budget is no longer a flat constant (any flat value
// would silently fall below the baseline the next time the baseline grew). It is
// DERIVED from the model window's effective ceiling sourced from internal/ctxplan
// (HardContextCap − OutputReserve; doctrine: docs/long-context-defaults.md — the
// advertised window is a hard CAP, never a raw target), floored by the measured
// baseline so a huge prompt still gets its own turn, times the turn count:
//
//	budget = max(hardCap − outputReserve, baseline) × claudeGuardTurnsPerEpoch
//
// FIXED(turn-starvation): the derivation this replaced was `baseline × 2` CLAMPED to
// `hardCap − outputReserve`. Both halves applied a PER-TURN window quantity to a
// CUMULATIVE allowance — a dimensional error, and the clamp was the hard wall:
// min(62000×k, 168000) = 168000 for every k ≥ 3, so NO headroom factor could ever buy
// a third turn. Live witness (.dispatch-runs/resolve-5103-20260726-022520.log): six
// turns at ctx 68.4k→70.2k→73.2k→73.8k→81.6k→83.2k, `context_tokens=124000` exhausted
// every second turn, `restart_exhausted count=3 dominant_cause=BUDGET_CONTEXT_EXHAUSTED`
// (cmd/fak/guard_child.go guardEquivalentRestartLimit) at 5m42s of a 29m runway → 409 →
// child exit 1 → CLAIM_NO_COMMIT on 120/120 worker witnesses. The window ceiling now
// bounds the PER-TURN resident, which is where it is dimensionally correct, and the
// turn count scales the cumulative total. Keep the arithmetic in sync with
// tools/dispatch_worker.py:claude_guard_context_budget_tokens — the Python sibling
// cannot import ctxplan, so it mirrors these constants by hand.
const (
	// claudeGuardBaselineTokens is the FLOOR under the measured launch-prompt
	// baseline — the workers' ~62K irreducible prompt (issue body + AGENTS/llms
	// orientation + injected fleet memory + the ~40K startup.json 'route' blob —
	// the breakdown above), hand-measured once. It is no longer the SOLE baseline:
	// measuredClaudeGuardBaseline now SIZES the constituents readable at launch and
	// floors the result here. The measurement feeds ONLY the baseline, and
	// max(measured, floor) ≥ floor, so no measurement (degenerate, partial, empty
	// inputs, unreadable files) can re-open the #2972 born-over-budget trap by
	// construction — while a launch prompt whose orientation/memory constituents have
	// organically grown PAST this floor raises the baseline (and the budget)
	// automatically, instead of silently outgrowing a frozen constant the way the old
	// flat baseline could. On a model whose window ceiling is SMALLER than this floor
	// (ctxplan's *fable* row: 64000−32000) the floor becomes the per-turn term, so a
	// worker is never born over-budget on a small-window model either.
	claudeGuardBaselineTokens = 62000
	// claudeGuardTurnsPerEpoch is how many FULL-WINDOW turns one child's cumulative
	// context budget funds before --restart-on-budget hands back and reseeds. Under the
	// cumulative meter documented above this is the only honest unit for this knob: the
	// budget buys turns, not window.
	//
	// Sizing: the guard reaps a worker after guardEquivalentRestartLimit = 3 identical
	// BUDGET_CONTEXT_EXHAUSTED restart cycles (cmd/fak/guard_child.go), so a child must
	// outlive a third of the wall clock or the reaper — not the work — ends the run. The
	// wall clock is claudeGuardMaxDuration() = 1740s and the witnessed dispatch turn rate
	// is ~57s/turn (6 turns over the 5m42s sampled in resolve-5103), i.e. ~30 turns of
	// runway. 3 × 12 = 36 > 30, so a healthy worker hits --max-duration (a graceful drain
	// with a final summary) instead of the equivalent-restart reaper. Witnessed residents
	// plateau near 45% of the window ceiling, so in practice this funds ~26 real turns per
	// child and the restart path returns to being a backstop rather than the lifecycle.
	// It is NOT the runaway backstop: that is --max-duration, the dispatcher reap, and
	// claudeGuardRestartLimit — plus the provider's own hard window, which physically
	// bounds the per-turn resident this multiplies.
	claudeGuardTurnsPerEpoch = 12
	// claudeGuardCompactHistoryBudget is the COMPACT shed-line passed to guard as
	// --compact-history-budget — DISTINCT from the drain ceiling above. It is the
	// resident-token target compaction squeezes OLD turns toward; guard's interactive
	// default (gateway.DefaultCompactHistoryBudget) is 48000. That 48K default is BELOW
	// the workers' ~62K irreducible baseline, so a dispatch worker can never shed under
	// it: compact=fired every turn but never succeeds, the resident stays permanently
	// "past compact", and the dispatch tick's ACTIVE_COMPACT_RUNAWAY hold (resident ≥20K
	// past compact over ≥3 turns) arms on every worker and WEDGES the dispatcher (idle
	// seats vs backlog). Neither launcher passed this flag, so the 96000 headless value
	// that exists precisely for this (gateway.HeadlessCompactHistoryBudget, reachable
	// only via --expose-profile headless) never applied. Pass it EXPLICITLY (explicit
	// wins in cmd/fak/guard.go:resolveGuardCompactBudget) so the shed-line sits above the
	// ~62K baseline: compaction can actually succeed and the false runaway hold stops
	// arming — without the tool-surface prune the headless profile also carries.
	//
	// NOT the drain ceiling, and NOT on the same scale as it. This is a PER-TURN
	// instantaneous target; --context-budget-tokens is a CUMULATIVE allowance. They only
	// look comparable because both render under the word "budget": the per-turn stderr
	// nudge prints `ctx:<resident>/<this>` (internal/gateway/debug_stats.go
	// formatCompactionBudgetNudge — the denominator is THIS constant, never the session
	// budget), so a worker can read `ctx:83.2k/96.0k dist:12.8k-to-compact` on the turn
	// its cumulative session budget dies. Requiring shed-line ≤ drain ceiling is therefore
	// a category error and is no longer asserted; what must hold is that the drain ceiling
	// funds enough TURNS at this resident to reach the shed line at all (the old 124000
	// funded two turns against a shed line the resident climbs ~1.8k/turn toward, so
	// compaction was structurally unreachable and every turn logged compact=none /
	// `bailed: under_budget`). Mirrors gateway.HeadlessCompactHistoryBudget (drift pinned
	// by TestClaudeGuardCompactHistoryBudget); Python mirrors it by hand. #4253.
	claudeGuardCompactHistoryBudget = 96000
	// claudeGuardCompactSolvencyPercent is the fraction (in PERCENT) of the model's
	// USABLE per-turn window — hardCap − outputReserve, the same ceiling the drain budget
	// is derived from — at or above which CONTEXT SOLVENCY overrides the head-anchored
	// burst gate's cache economics (`--compact-solvency-floor`, gateway
	// Config.CompactSolvencyFloorTokens).
	//
	// Why the override needs an operator-supplied number at all: the gateway prices a
	// compaction burst in cache dollars (CacheBurstPaysBack, #1408) and has NO term for
	// running out of window, because it never sees a window SIZE — it only sees the bytes
	// of one request. So it refuses hardest exactly where refusing is most expensive.
	// Measured over 3191 served turns in .dispatch-runs, the fire rate INVERTED against
	// occupancy (33.4% at 96–110k, 33.9% at 110–125k, 24.7% at 125–140k, 14.3% at
	// 140–155k, 3.4% at 155–170k, 0.0% above 170k), 100% of the traces that ever fired
	// never fired again, and their un-compacted tails carried resident a median +33.8k
	// further in. 1622 turns ran PAST the 96000 shed-line above without firing (median
	// 23.4k over, max 97.6k over) and 5% of traces peaked past the usable window. This
	// file is the one place in the launch path that knows the window, so it derives the
	// floor and hands it over.
	//
	// Why 85%: the floor must sit well ABOVE the shed-line (96000 ≈ 57% of 168000) or it
	// would fire on every ordinary turn and throw away the cache economics wholesale; and
	// well BELOW the ceiling or the forced fire arrives after the wall. 85% of 168000 =
	// 142800 leaves ~25k of usable window — more than a full witnessed turn's ~1.8k/turn
	// climb plus the largest single tool result — for the forced burst to land and repay.
	// It is a LAST-RESORT line, not a target: between the shed-line and this floor the
	// pure economics still decide every fire.
	claudeGuardCompactSolvencyPercent = 85
	// claudeGuardRestartLimit caps --restart-on-budget relaunches. The budget above
	// funds claudeGuardTurnsPerEpoch full-window turns per epoch (each turn debits its
	// FULL resident window), so a relaunch happens every ~12+ turns rather than the ~2
	// the pre-fix derivation allowed. The old "2" killed a healthy worker after
	// ~4-5 min / ~2 epochs (reset_limit limit=2 → 409 → CHILD_CRASH → CLAIM_NO_COMMIT)
	// at ~15% of its runway — the dominant fleet crash. This is NOT the runaway
	// backstop (that is the wall-clock defaultTimeoutS hard-kill + dispatcher reap, and
	// claudeGuardMaxDuration below); it only trips a degenerate sub-2-min reset storm.
	// 16 × ~2 min > the 30-min wall-clock, so wall-clock is the real bound for a
	// healthy-but-slow worker while a storm still trips here.
	claudeGuardRestartLimit = "16"
	// claudeGuardMaxDurationMarginS backs the in-guard wall-clock budget off the
	// worker's own defaultTimeoutS hard-kill (worker.go) so the guard drains GRACEFULLY
	// (TIME_BUDGET_EXHAUSTED, final summary + audit flush) a minute before launch()'s
	// context cancel would SIGKILL the tree at exit 124. --max-duration is CUMULATIVE
	// across --restart-on-budget relaunches (see `fak guard` help), so it bounds TOTAL
	// worker lifespan regardless of how many restarts the raised limit above permits.
	claudeGuardMaxDurationMarginS = 60
)

// claudeGuardMaxDuration is the guard's in-process wall-clock budget as the
// `--max-duration` argv string: defaultTimeoutS minus a drain margin, so a stuck
// worker stops itself gracefully just before the worker's hard-kill. Mirrors
// tools/dispatch_worker.py:CLAUDE_GUARD_MAX_DURATION.
func claudeGuardMaxDuration() string {
	return strconv.Itoa(defaultTimeoutS-claudeGuardMaxDurationMarginS) + "s"
}

// deriveClaudeGuardContextBudget is the pure arithmetic: the PER-TURN resident term
// (the effective window ceiling hardCap − outputReserve, floored by the baseline so a
// small-window model still funds its own launch prompt) times claudeGuardTurnsPerEpoch,
// because the seeded budget is a cumulative allowance debited one full resident window
// per turn (see the SEMANTICS block above). Model-aware through the ceiling, birth-safe
// by construction (the per-turn term is ≥ baseline and the turn count is ≥ 2, so the
// result always exceeds the baseline), and non-decreasing in the baseline.
//
// The window ceiling deliberately does NOT clamp the returned total: clamping a
// cumulative allowance to a per-turn window is the dimensional error that starved every
// dispatch worker at ~2 turns regardless of any factor.
func deriveClaudeGuardContextBudget(baselineTokens, hardContextCap, outputReserve int) int {
	perTurn := hardContextCap - outputReserve
	if perTurn < baselineTokens {
		perTurn = baselineTokens
	}
	return perTurn * claudeGuardTurnsPerEpoch
}

// deriveClaudeGuardSolvencyFloor is the pure arithmetic behind --compact-solvency-floor:
// claudeGuardCompactSolvencyPercent of the USABLE per-turn window (hardCap −
// outputReserve). Model-aware through the same ceiling the drain budget uses, so a
// small-window model gets a proportionally lower floor rather than an unreachable one.
//
// Deliberately NOT floored by the launch baseline the way the drain budget is: the floor
// is an occupancy ALARM, and a model whose usable window is barely wider than the launch
// prompt is one where the alarm should ring early and often. Returns 0 for a degenerate
// (non-positive) envelope, which DISARMS the override and leaves the gate on pure
// economics — the fail-safe direction.
func deriveClaudeGuardSolvencyFloor(hardContextCap, outputReserve int) int {
	usable := hardContextCap - outputReserve
	if usable <= 0 {
		return 0
	}
	return usable * claudeGuardCompactSolvencyPercent / 100
}

// claudeGuardCompactSolvencyFloorTokens returns the derived solvency floor as the
// `--compact-solvency-floor` argv string for this worker's model. Mirrors
// dispatch_worker.claude_guard_compact_solvency_floor_tokens on the generic envelope.
func claudeGuardCompactSolvencyFloorTokens(workerModel string) string {
	envelope := ctxplan.EnvelopeForModel(workerModel)
	return strconv.Itoa(deriveClaudeGuardSolvencyFloor(envelope.HardContextCap, envelope.OutputReserve))
}

// launchConstituentFiles are the workspace-ROOT launch-prompt constituents a
// self-claiming lane worker can actually SIZE at launch: the orientation files every
// worker loads (AGENTS.md, llms.txt, CLAUDE.md), plus a workspace-root MEMORY.md when a
// repo keeps one. NOTE the real injected fleet memory does NOT live at the workspace
// root — it is in the per-project claude memory dir (…/projects/<ws>/memory/MEMORY.md),
// off the root and not portably derivable here, so in the common fleet layout MEMORY.md
// is absent and the FLOOR covers it; it is listed so a repo that DOES keep a root
// MEMORY.md gets it measured. Likewise the per-issue body and the ~40K startup.json
// 'route' blob are NOT visible here (the worker claims its issue after launch, and the
// startup sidecar is keyed to a log path the worker never learns) — the
// claudeGuardBaselineTokens FLOOR covers that remainder. The startup blob is measured
// too when the launcher names it via DISPATCH_STARTUP_BUNDLE (launchStartupBundleEnv).
// Mirrors dispatch_worker.LAUNCH_CONSTITUENT_FILES.
var launchConstituentFiles = []string{"AGENTS.md", "llms.txt", "CLAUDE.md", "MEMORY.md"}

// launchStartupBundleEnv, when set to a readable path, folds that file's byte size
// (the startup.json 'route' blob) into the measured baseline — the one launch
// constituent that lives off the workspace root. Absent/unreadable ⇒ the floor covers
// it. Mirrors dispatch_worker.LAUNCH_STARTUP_BUNDLE_ENV.
const launchStartupBundleEnv = "DISPATCH_STARTUP_BUNDLE"

// approxTokensFromBytes estimates tokens from a byte count with the codebase's
// standard (bytes+3)/4 heuristic (internal/ctxplan/plan.go, materialize.go) — the
// SAME ruler the ctxplan planner sizes context with, so a measured launch prompt is
// weighed against the window on one scale. Pure; no tokenizer dependency. Mirrors
// dispatch_worker.approx_tokens_from_bytes.
func approxTokensFromBytes(bytes int) int {
	if bytes <= 0 {
		return 0
	}
	return (bytes + 3) / 4
}

// measureLaunchBaselineTokens sums the approximate token footprint of a worker's
// launch-prompt constituents from their real byte sizes — the measurement seam that
// replaces the frozen 62000 guess with what the loop actually carries. constituentBytes
// maps a constituent name -> its byte size (the names are for the observable audit
// only). An empty/nil map measures 0 (degenerate); the caller floors it, so a missing
// measurement can never lower the budget. Mirrors dispatch_worker.measure_launch_baseline_tokens.
func measureLaunchBaselineTokens(constituentBytes map[string]int) int {
	total := 0
	for _, b := range constituentBytes {
		total += approxTokensFromBytes(b)
	}
	return total
}

// resolveClaudeGuardBaseline folds a measured launch-prompt footprint against the
// hand-measured floor: the baseline that drives the budget is max(measured, floor).
// The floor guarantees #2972's invariant (a degenerate/partial measurement can NEVER
// emit a budget below the shipped 62000-derived value, so the born-over-budget trap
// cannot return); a prompt grown past the floor raises the baseline automatically,
// closing the "nothing measures the real prompt" gap. Mirrors
// dispatch_worker.resolve_claude_guard_baseline.
func resolveClaudeGuardBaseline(measuredTokens int) int {
	if measuredTokens > claudeGuardBaselineTokens {
		return measuredTokens
	}
	return claudeGuardBaselineTokens
}

// gatherLaunchConstituentBytes reads the byte sizes of the launch constituents
// readable at launch from workspace (I/O; an unreadable/absent file contributes
// nothing — the degenerate guard). Returns a name->bytes map for measurement AND for
// the observable. An empty workspace measures nothing (hermetic default → the floor).
// Mirrors dispatch_worker.gather_launch_constituent_bytes.
func gatherLaunchConstituentBytes(workspace string, env map[string]string) map[string]int {
	out := map[string]int{}
	if workspace == "" {
		return out
	}
	for _, name := range launchConstituentFiles {
		if info, err := os.Stat(filepath.Join(workspace, name)); err == nil && !info.IsDir() {
			out[name] = int(info.Size())
		}
	}
	if p := strings.TrimSpace(env[launchStartupBundleEnv]); p != "" {
		if info, err := os.Stat(p); err == nil && !info.IsDir() {
			out["startup_bundle"] = int(info.Size())
		}
	}
	return out
}

// measuredClaudeGuardBaseline sizes the readable launch constituents, floors the
// measurement, and returns both the floored baseline and the raw constituent sizes
// (for the observable). Mirrors dispatch_worker.measured_claude_guard_baseline.
func measuredClaudeGuardBaseline(workspace string, env map[string]string) (int, map[string]int) {
	constituents := gatherLaunchConstituentBytes(workspace, env)
	return resolveClaudeGuardBaseline(measureLaunchBaselineTokens(constituents)), constituents
}

// claudeGuardContextBudgetTokens returns the derived per-session context budget as
// the `--context-budget-tokens` argv string, seeded by the MEASURED launch-prompt
// baseline (floored) rather than a frozen constant. The window and output reserve
// come from the ctxplan envelope table (the routine-agent-turn row), NOT a fresh
// local literal — so a model-window change lands here without touching this file.
// An empty workspace measures nothing and falls to the floor (the hermetic default is
// then (200000−32000) × 12 = 2016000, i.e. twelve full-window turns).
// Mirrors dispatch_worker.claude_guard_context_budget_tokens.
func claudeGuardContextBudgetTokens(workspace, workerModel string, env map[string]string) string {
	envelope := ctxplan.EnvelopeForModel(workerModel)
	baseline, _ := measuredClaudeGuardBaseline(workspace, env)
	return strconv.Itoa(deriveClaudeGuardContextBudget(
		baseline, envelope.HardContextCap, envelope.OutputReserve))
}

// claudeGuardBudgetObservable returns the measured launch-prompt baseline and the
// derived context budget for the claude backend, so a launch record exposes what the
// guard was actually seeded with — drift is a visible number in the payload, not an
// argv int nobody reads. env defaults to the process environment when nil.
func claudeGuardBudgetObservable(workspace, workerModel string, env map[string]string) (baseline, budget int) {
	if env == nil {
		env = processEnvMap()
	}
	envelope := ctxplan.EnvelopeForModel(workerModel)
	baseline, _ = measuredClaudeGuardBaseline(workspace, env)
	budget = deriveClaudeGuardContextBudget(baseline, envelope.HardContextCap, envelope.OutputReserve)
	return baseline, budget
}

// claudeGuardArgs builds the claude guard argv at call time so the budget arg
// carries the MEASURED value for this workspace. This Go helper bundles the precompact
// hook AND the budget block; on the Python side those live in two pieces
// (CLAUDE_GUARD_PRECOMPACT_ARGS + claude_guard_budget_args), so only the budget portion
// here mirrors dispatch_worker.claude_guard_budget_args 1:1 (the integers are identical;
// --session-id placement between the two blocks is a guard-parsed, order-independent
// flag and is not pinned across languages).
func claudeGuardArgs(workspace, workerModel string, env map[string]string) []string {
	return []string{
		"--precompact-hook", "enforce",
		"--context-budget-tokens", claudeGuardContextBudgetTokens(workspace, workerModel, env),
		"--compact-history-budget", strconv.Itoa(claudeGuardCompactHistoryBudget),
		"--compact-solvency-floor", claudeGuardCompactSolvencyFloorTokens(workerModel),
		"--restart-on-budget",
		"--restart-limit", claudeGuardRestartLimit,
		"--max-duration", claudeGuardMaxDuration(),
	}
}

// guardEnabled reports whether to front a worker with `fak guard`. Dogfood-by-default
// (ON); a node opts out with FLEET_DOGFOOD_GUARD in {0,off,false,no,"",disable,disabled}.
// Mirrors dispatch_worker.guard_enabled: an ABSENT key is ON, a present-but-off value
// is OFF.
func guardEnabled(env map[string]string) bool {
	raw, ok := env["FLEET_DOGFOOD_GUARD"]
	if !ok {
		return true
	}
	_, off := guardOffValues[strings.ToLower(strings.TrimSpace(raw))]
	return !off
}

// resolveFakBin locates a `fak` binary to front the worker with, or "" (fail OPEN).
// Precedence: $FAK_BIN (if it exists) -> the in-tree tools/.bin/fak[.exe] the dogfood
// launcher builds -> `fak` on the supplied env's PATH. "" means the caller launches
// the worker UNWRAPPED rather than breaking dispatch on a host that has not built fak.
// Mirrors dispatch_worker.resolve_fak_bin.
func resolveFakBin(workspace string, env map[string]string) string {
	if explicit := strings.TrimSpace(env["FAK_BIN"]); explicit != "" && fileExists(explicit) {
		return explicit
	}
	exe := "fak"
	if runtime.GOOS == "windows" {
		exe = "fak.exe"
	}
	intree := filepath.Join(workspace, "tools", ".bin", exe)
	if fileExists(intree) {
		return intree
	}
	// Honor the supplied env's PATH (so the env param fully governs resolution); an
	// ABSENT PATH key falls back to the process PATH via exec.LookPath.
	pathVal, hasPath := env["PATH"]
	return whichOnExactPath("fak", pathVal, hasPath)
}

// whichOnExactPath resolves name using exactly pathVal (when hasPath), honoring
// PATHEXT for command shims on Windows. When the PATH key is absent (hasPath=false)
// it falls back to exec.LookPath over the process PATH. Mirrors
// dispatch_worker._which_on_exact_path; returns "" for "not found" (Python None).
func whichOnExactPath(name, pathVal string, hasPath bool) string {
	if !hasPath {
		if p, err := exec.LookPath(name); err == nil {
			return p
		}
		return ""
	}
	suffixes := []string{""}
	if runtime.GOOS == "windows" && filepath.Ext(name) == "" {
		pathext := os.Getenv("PATHEXT")
		if pathext == "" {
			pathext = ".COM;.EXE;.BAT;.CMD"
		}
		for _, ext := range strings.Split(pathext, string(os.PathListSeparator)) {
			if ext == "" {
				continue
			}
			suffixes = append(suffixes, strings.ToLower(ext), strings.ToUpper(ext))
		}
	}
	for _, dir := range strings.Split(pathVal, string(os.PathListSeparator)) {
		if dir == "" {
			continue
		}
		for _, suf := range suffixes {
			cand := filepath.Join(dir, name+suf)
			if fileExists(cand) {
				return cand
			}
		}
	}
	return ""
}

func fileExists(p string) bool {
	info, err := os.Stat(p)
	return err == nil && !info.IsDir()
}

// guardProvider is the upstream wire `fak guard` proxies for a worker backend.
// claude -> the Anthropic API (passthrough/subscription); every other backend is
// OpenAI-wire. Mirrors dispatch_worker.guard_provider.
func guardProvider(backend string) string {
	if backend == "claude" {
		return "anthropic"
	}
	return "openai"
}

// guardAuditPath is a PER-SESSION durable decision journal under the gitignored
// .dispatch-runs/guard-audit/. The filename is keyed on the lane+backend (for
// globbing) PLUS a per-process discriminator (pid + a random token), because
// `fak guard`'s hash-chained journal has NO inter-process lock: two concurrent
// workers sharing ONE file would braid two independent sha256 chains into a forked,
// unverifiable journal. A unique-per-session file lets each `fak guard` own its own
// valid chain; the rollup globs the lane prefix to aggregate them. Mirrors
// dispatch_worker.guard_audit_path.
func guardAuditPath(workspace, lane, backend string) string {
	safe := sanitizeAuditName(lane + "-" + backend)
	token := fmt.Sprintf("%d-%s", os.Getpid(), randToken())
	return filepath.Join(workspace, ".dispatch-runs", "guard-audit", safe+"-"+token+".jsonl")
}

// sanitizeAuditName keeps the lane/backend prefix globbable: alnum and -_. survive,
// everything else (path separators, spaces) becomes _. Mirrors the Python
// comprehension in guard_audit_path.
func sanitizeAuditName(s string) string {
	var b strings.Builder
	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsDigit(r) || r == '-' || r == '_' || r == '.' {
			b.WriteRune(r)
		} else {
			b.WriteRune('_')
		}
	}
	return b.String()
}

// randToken returns 8 hex chars from crypto/rand — the per-session discriminator that
// keeps two workers on the SAME lane from colliding on one journal file. Date.now /
// rand are fine here (off the resume-cacheable workflow path); a rand failure falls
// back to a pid-derived token (still unique enough paired with the pid prefix).
func randToken() string {
	if token, ok := randhex.String(4); ok {
		return token
	}
	return fmt.Sprintf("%08x", os.Getpid())
}

// guardWrap fronts a raw worker argv with `fak guard -- <worker>` so the kernel
// adjudicates every tool call. Pure given fakBin. Returns the command UNCHANGED when:
//
//   - fakBin is "" (no binary resolved -> fail open), or
//   - the backend fronts a LOCAL upstream we have not been told the base URL of.
//     claude proxies the public Anthropic API (passthrough/subscription) with no
//     base-URL override; opencode (and friends) front a local server (e.g. a GLM
//     endpoint), so guard would MISROUTE them to the provider's public API unless
//     FLEET_DOGFOOD_GUARD_BASEURL names that upstream. We refuse to misroute.
//
// Mirrors dispatch_worker.guard_wrap.
const codexCompactTokenLimit = 96000

func withCodexCompactLimit(command []string) []string {
	if len(command) == 0 {
		return command
	}
	out := make([]string, 0, len(command)+2)
	out = append(out, command[0], "-c", fmt.Sprintf("model_auto_compact_token_limit=%d", codexCompactTokenLimit))
	return append(out, command[1:]...)
}

const (
	opencodeCompactShedLineTokens = 96000
	opencodeDefaultProviderID     = "zai-coding-plan"
)

type opencodeWindowLimit struct {
	Context int
	Output  int
}

var opencodeModelLimits = map[string]map[string]opencodeWindowLimit{
	"zai-coding-plan": {
		"glm-5.2":      {Context: 1000000, Output: 131072},
		"glm-5.1":      {Context: 200000, Output: 131072},
		"glm-5-turbo":  {Context: 200000, Output: 131072},
		"glm-5v-turbo": {Context: 200000, Output: 131072},
		"glm-4.7":      {Context: 204800, Output: 131072},
		"glm-4.5-air":  {Context: 131072, Output: 98304},
	},
}

func opencodeModelProvider(command []string) string {
	for i, token := range command {
		if (token == "-m" || token == "--model") && i+1 < len(command) {
			model := strings.TrimSpace(command[i+1])
			if idx := strings.Index(model, "/"); idx != -1 {
				provider := strings.TrimSpace(model[:idx])
				if provider != "" {
					return provider
				}
			}
		}
	}
	return opencodeDefaultProviderID
}

// opencodeCompactionOverlay builds the OpenCode-native 96K shed-line config overlay,
// or nil when the provider's real limits are unknown (fail OPEN). Mirrors
// tools/dispatch_worker.py:opencode_compaction_overlay.
func opencodeCompactionOverlay(command []string) map[string]any {
	provider := opencodeModelProvider(command)
	catalog, ok := opencodeModelLimits[provider]
	if !ok || len(catalog) == 0 {
		return nil
	}
	models := map[string]any{}
	for model, lim := range catalog {
		if lim.Context <= opencodeCompactShedLineTokens {
			continue
		}
		models[model] = map[string]any{
			"limit": map[string]any{
				"context": lim.Context,
				"input":   opencodeCompactShedLineTokens,
				"output":  lim.Output,
			},
		}
	}
	if len(models) == 0 {
		return nil
	}
	return map[string]any{
		"compaction": map[string]any{
			"auto":     true,
			"reserved": 0,
		},
		"provider": map[string]any{
			provider: map[string]any{
				"models": models,
			},
		},
	}
}

func opencodeGuardConfigContent(command []string, gatewayBaseURL, existing string) string {
	provider := opencodeModelProvider(command)
	overlay := map[string]any{
		"provider": map[string]any{
			provider: map[string]any{
				"options": map[string]any{
					"baseURL": gatewayBaseURL,
				},
			},
		},
	}
	base := map[string]any{}
	if trimmed := strings.TrimSpace(existing); trimmed != "" {
		_ = json.Unmarshal([]byte(trimmed), &base)
	}
	merged := deepMergeConfig(base, overlay)
	if compOverlay := opencodeCompactionOverlay(command); compOverlay != nil {
		merged = deepMergeConfig(merged, compOverlay)
	}
	data, _ := json.Marshal(merged)
	return string(data)
}

// deepMergeConfig overlays overlay onto base, recursing into nested objects. Mirrors
// dispatch_worker._deep_merge_config.
func deepMergeConfig(base, overlay map[string]any) map[string]any {
	if base == nil {
		base = map[string]any{}
	}
	for k, v := range overlay {
		if sub, ok := v.(map[string]any); ok {
			if cur, ok := base[k].(map[string]any); ok {
				base[k] = deepMergeConfig(cur, sub)
				continue
			}
		}
		base[k] = v
	}
	return base
}

func guardWrap(command []string, fakBin, lane, backend, workspace, workerModel string, env map[string]string) []string {
	if len(command) == 0 || fakBin == "" {
		return command
	}
	provider := guardProvider(backend)
	if backend == "codex" {
		// --compact-history-budget is Anthropic-only. On the Responses wire,
		// configure Codex's native compactor before its exec/TUI subcommand.
		command = withCodexCompactLimit(command)
	}
	var extra []string
	audit := guardAuditPath(workspace, lane, backend)
	if backend == "claude" {
		extra = append(extra, claudeGuardArgs(workspace, workerModel, env)...)
		extra = append(extra, "--session-id", strings.TrimSuffix(filepath.Base(audit), filepath.Ext(audit)))
	}
	if backend != "claude" {
		base := strings.TrimSpace(env["FLEET_DOGFOOD_GUARD_BASEURL"])
		if base == "" {
			return command // don't misroute a local-upstream worker
		}
		extra = []string{"--base-url", base}
		if backend == "opencode" {
			addr := strings.TrimSpace(env["FLEET_DOGFOOD_GUARD_ADDR"])
			if addr != "" {
				extra = append([]string{"--addr", addr}, extra...)
			}
			if env != nil {
				targetAddr := addr
				if targetAddr == "" {
					targetAddr = "127.0.0.1:8137"
				}
				env["OPENCODE_CONFIG_CONTENT"] = opencodeGuardConfigContent(command, "http://"+targetAddr+"/v1", env["OPENCODE_CONFIG_CONTENT"])
			}
		}
	}
	out := []string{fakBin, "guard", "--provider", provider}
	out = append(out, extra...)
	out = append(out, "--audit", audit, "--")
	out = append(out, command...)
	return out
}

// guardEnvAugment ensures a guarded worker's gateway won't truncate a long frontier
// turn: it sets FAK_PLANNER_TIMEOUT_S / FAK_HTTP_WRITE_TIMEOUT_S to a generous floor
// when unset (an explicit operator value is left as-is). Mutates and returns env.
// Mirrors dispatch_worker.guard_env_augment.
func guardEnvAugment(env map[string]string) map[string]string {
	for _, key := range []string{"FAK_PLANNER_TIMEOUT_S", "FAK_HTTP_WRITE_TIMEOUT_S"} {
		if env[key] == "" {
			env[key] = strconv.Itoa(guardTimeoutFloorS)
		}
	}
	return env
}

// guardedLaunchCommand resolves the argv to actually launch: command fronted by
// `fak guard` when dogfood mode is on and a fak binary resolves, else command
// unchanged. Returns (launchCommand, guarded) so callers can both run it and report
// what ran. env defaults to the process environment when nil. Mirrors
// dispatch_worker.guarded_launch_command.
func guardedLaunchCommand(command []string, lane, backend, workspace, workerModel string, env map[string]string) ([]string, bool) {
	if env == nil {
		env = processEnvMap()
	}
	fakBin := ""
	if guardEnabled(env) {
		fakBin = resolveFakBin(workspace, env)
	}
	if fakBin == "" {
		return command, false
	}
	wrapped := guardWrap(command, fakBin, lane, backend, workspace, workerModel, env)
	return wrapped, !slices.Equal(wrapped, command)
}

// processEnvMap snapshots the process environment as a map (the default env for the
// guard helpers, mirroring Python's os.environ default).
func processEnvMap() map[string]string {
	m := map[string]string{}
	for _, kv := range os.Environ() {
		if i := strings.IndexByte(kv, '='); i >= 0 {
			m[kv[:i]] = kv[i+1:]
		}
	}
	return m
}

func sliceEqual(a, b []string) bool {
	return slices.Equal(a, b)
}
