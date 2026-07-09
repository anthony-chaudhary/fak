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
// SEMANTICS (internal/session/usage.go DebitUsage): every turn debits that turn's
// ENTIRE resident context window (prompt + cache_read + cache_creation — the whole
// window, not the newly-added delta) from this budget; when it reaches <=0 the
// session drains with BUDGET_CONTEXT_EXHAUSTED. So for a single-issue `-p` worker
// this behaves as a per-turn resident-window ceiling: if turn 1's window exceeds
// the budget, the worker is born over-budget and never runs.
//
// The old 48000 was BELOW the workers' ~62K irreducible baseline prompt (issue body
// + AGENTS/llms orientation + injected fleet memory + the ~40K startup.json 'route'
// blob), so every dispatch claude worker exhausted on turn 1 → 2 restarts burned →
// raw 409 → child exit 1 → CHILD_CRASH (fleet ship rate collapsed to 2-9%).
//
// DONE(dynamic-budget): the budget is no longer a flat constant (any flat value
// would silently fall below the baseline the next time the baseline grew). It is
// DERIVED: baseline × birth-headroom, clamped to the model window's effective
// ceiling sourced from internal/ctxplan (HardContextCap − OutputReserve; doctrine:
// docs/long-context-defaults.md — the advertised window is a hard CAP, never a raw
// target). Grows with the baseline (never a birth wall again), shrinks with the
// window (always a runaway backstop). Keep the arithmetic in sync with
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
	// flat baseline could. (The 124000 shipped default additionally assumes the
	// ctxplan window ceiling stays ≥ baseline×claudeGuardBirthHeadroomFactor; that
	// ceiling side is pinned by the derivation goldens, not by this floor.)
	claudeGuardBaselineTokens = 62000
	// claudeGuardBirthHeadroomFactor: the derived budget is baseline × this factor,
	// so a worker is born with the whole baseline again in headroom before the
	// runaway backstop fires. (History: the committed value this derivation replaced
	// was the crash-looping flat "48000"; a flat 131072 was only ever an uncommitted
	// working-tree interim, never shipped.) For the
	// 62000 baseline / 200000-32000 window this yields 124000 — comfortably above
	// baseline, comfortably under the 168000 effective ceiling.
	claudeGuardBirthHeadroomFactor = 2
	// claudeGuardRestartLimit caps --restart-on-budget relaunches. The budget above
	// drains in ~2 turns per epoch (each turn debits its FULL ~62K+ resident window),
	// so a relaunch happens every ~2 min. The old "2" killed a healthy worker after
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

// deriveClaudeGuardContextBudget is the pure arithmetic: baseline × headroom,
// clamped to the effective window ceiling (hardCap − outputReserve). Monotone in
// the baseline until the ceiling clamps; the ceiling keeps it under the real model
// window even if the baseline balloons.
func deriveClaudeGuardContextBudget(baselineTokens, hardContextCap, outputReserve int) int {
	ceiling := hardContextCap - outputReserve
	budget := baselineTokens * claudeGuardBirthHeadroomFactor
	if budget > ceiling {
		budget = ceiling
	}
	return budget
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
// An empty workspace measures nothing and falls to the floor (the hermetic default
// preserves the shipped 124000). Mirrors dispatch_worker.claude_guard_context_budget_tokens.
func claudeGuardContextBudgetTokens(workspace string, env map[string]string) string {
	envelope := ctxplan.GenericTurnEnvelope()
	baseline, _ := measuredClaudeGuardBaseline(workspace, env)
	return strconv.Itoa(deriveClaudeGuardContextBudget(
		baseline, envelope.HardContextCap, envelope.OutputReserve))
}

// claudeGuardBudgetObservable returns the measured launch-prompt baseline and the
// derived context budget for the claude backend, so a launch record exposes what the
// guard was actually seeded with — drift is a visible number in the payload, not an
// argv int nobody reads. env defaults to the process environment when nil.
func claudeGuardBudgetObservable(workspace string, env map[string]string) (baseline, budget int) {
	if env == nil {
		env = processEnvMap()
	}
	envelope := ctxplan.GenericTurnEnvelope()
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
func claudeGuardArgs(workspace string, env map[string]string) []string {
	return []string{
		"--precompact-hook", "enforce",
		"--context-budget-tokens", claudeGuardContextBudgetTokens(workspace, env),
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
func guardWrap(command []string, fakBin, lane, backend, workspace string, env map[string]string) []string {
	if len(command) == 0 || fakBin == "" {
		return command
	}
	provider := guardProvider(backend)
	var extra []string
	audit := guardAuditPath(workspace, lane, backend)
	if backend == "claude" {
		extra = append(extra, claudeGuardArgs(workspace, env)...)
		extra = append(extra, "--session-id", strings.TrimSuffix(filepath.Base(audit), filepath.Ext(audit)))
	}
	if backend != "claude" {
		base := strings.TrimSpace(env["FLEET_DOGFOOD_GUARD_BASEURL"])
		if base == "" {
			return command // don't misroute a local-upstream worker
		}
		extra = []string{"--base-url", base}
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
func guardedLaunchCommand(command []string, lane, backend, workspace string, env map[string]string) ([]string, bool) {
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
	wrapped := guardWrap(command, fakBin, lane, backend, workspace, env)
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
