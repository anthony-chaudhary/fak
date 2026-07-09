// guard_test.go — parity witness for the Go dogfood-guard port, mirroring the
// guard test table in tools/dispatch_worker_test.py. Hermetic: no process is
// spawned; a real temp file stands in for the resolved `fak` binary.
package main

import (
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

func TestGuardEnabledDefaultOnAndOptOut(t *testing.T) {
	if !guardEnabled(map[string]string{}) { // unset -> ON (dogfood default)
		t.Error("absent FLEET_DOGFOOD_GUARD must be ON")
	}
	for _, on := range []string{"1", "on"} {
		if !guardEnabled(map[string]string{"FLEET_DOGFOOD_GUARD": on}) {
			t.Errorf("%q must be ON", on)
		}
	}
	for _, off := range []string{"0", "off", "false", "no", "", "disable", "DISABLED", " Off "} {
		if guardEnabled(map[string]string{"FLEET_DOGFOOD_GUARD": off}) {
			t.Errorf("%q must be OFF", off)
		}
	}
}

func TestResolveFakBinPrefersEnvThenIntreeThenPathElseEmpty(t *testing.T) {
	// An explicit FAK_BIN that exists wins.
	existing := filepath.Join(t.TempDir(), "fak-stand-in")
	writeFile(t, existing, "x")
	if got := resolveFakBin("C:/nope", map[string]string{"FAK_BIN": existing}); got != existing {
		t.Errorf("explicit FAK_BIN must win: got %q", got)
	}
	// A non-existent FAK_BIN is ignored; with a bogus workspace and a PATH holding no
	// `fak`, the result is "" (the fail-open signal).
	emptyDir := t.TempDir()
	got := resolveFakBin("C:/definitely/not/a/repo/xyz", map[string]string{
		"FAK_BIN": "C:/no/such/fak", "PATH": emptyDir})
	if got != "" {
		t.Errorf("unresolvable fak must be \"\": got %q", got)
	}
}

func TestGuardProviderMapsClaudeToAnthropicElseOpenai(t *testing.T) {
	if guardProvider("claude") != "anthropic" {
		t.Error("claude -> anthropic")
	}
	if guardProvider("opencode") != "openai" {
		t.Error("opencode -> openai")
	}
}

func TestGuardAuditPathIsPerSessionUnderDispatchRuns(t *testing.T) {
	p := guardAuditPath("C:/work/fak", "gate way/1", "claude")
	if filepath.Base(filepath.Dir(p)) != "guard-audit" {
		t.Errorf("parent dir = %q, want guard-audit", filepath.Base(filepath.Dir(p)))
	}
	if filepath.Base(filepath.Dir(filepath.Dir(p))) != ".dispatch-runs" {
		t.Errorf("grandparent = %q, want .dispatch-runs", filepath.Base(filepath.Dir(filepath.Dir(p))))
	}
	name := filepath.Base(p)
	if !strings.HasSuffix(name, ".jsonl") {
		t.Errorf("name %q must end .jsonl", name)
	}
	if strings.ContainsAny(name, "/ ") {
		t.Errorf("name %q must have lane separators/spaces sanitized out", name)
	}
	if !strings.HasPrefix(name, "gate_way_1-claude-") {
		t.Errorf("name %q must keep the sanitized lane-backend prefix for globbing", name)
	}
}

func TestGuardAuditPathUniquePerCall(t *testing.T) {
	// Two workers on the SAME lane must NOT resolve to the same journal file, or their
	// independent hash chains would braid into one unverifiable file.
	a := guardAuditPath("C:/work/fak", "gateway", "claude")
	b := guardAuditPath("C:/work/fak", "gateway", "claude")
	if a == b {
		t.Errorf("per-session journal paths must differ: %q == %q", a, b)
	}
}

// TestClaudeGuardContextBudgetDerivation locks the DERIVED budget (the
// TODO(dynamic-budget) resolution): baseline × birth-headroom, clamped to the
// ctxplan effective window ceiling. Mirror: tools/dispatch_worker.py
// claude_guard_context_budget_tokens must yield the SAME integer; its parity
// partner test_claude_guard_context_budget_derivation_matches_go (in
// tools/dispatch_worker_test.py) pins the same 124000 golden — update both
// in the same commit.
func TestClaudeGuardContextBudgetDerivation(t *testing.T) {
	env := ctxplan.GenericTurnEnvelope()
	ceiling := env.HardContextCap - env.OutputReserve
	got, err := strconv.Atoi(claudeGuardContextBudgetTokens())
	if err != nil {
		t.Fatalf("derived budget must be a wired integer: %v", err)
	}
	// (a) Birth-safety: strictly above the baseline (a worker is never born
	// over-budget), with real headroom, not merely +1.
	if got <= claudeGuardBaselineTokens {
		t.Errorf("derived budget %d <= baseline %d: workers would be born over-budget (see #2972)", got, claudeGuardBaselineTokens)
	}
	// (b) Runaway backstop: at/under the effective window ceiling (HardContextCap −
	// OutputReserve), so the cap always bites below the real model window.
	if got > ceiling {
		t.Errorf("derived budget %d > effective window ceiling %d (HardContextCap %d − OutputReserve %d)", got, ceiling, env.HardContextCap, env.OutputReserve)
	}
	// (c) Golden lock for the shipped constants: min(62000×2, 200000−32000) = 124000.
	// If this fails, someone changed the baseline, the headroom factor, or the
	// ctxplan envelope — update tools/dispatch_worker.py IN THE SAME COMMIT.
	if want := 124000; got != want {
		t.Errorf("derived budget = %d, want %d; keep tools/dispatch_worker.py claude_guard_context_budget_tokens in sync", got, want)
	}
	// (d) Monotone in the baseline below the ceiling: a baseline bump RAISES the
	// budget (the flat-constant staleness this derivation kills)...
	if bumped := deriveClaudeGuardContextBudget(claudeGuardBaselineTokens+1000, env.HardContextCap, env.OutputReserve); bumped <= got {
		t.Errorf("baseline bump must raise the derived budget: %d (baseline+1000) <= %d", bumped, got)
	}
	// ...while a runaway baseline is still clamped to the window ceiling.
	if clamped := deriveClaudeGuardContextBudget(ceiling, env.HardContextCap, env.OutputReserve); clamped != ceiling {
		t.Errorf("over-window baseline must clamp to ceiling %d, got %d", ceiling, clamped)
	}
	// The CLI surface: exact flags, exact order, derived value wired in.
	want := []string{
		"--precompact-hook", "enforce",
		"--context-budget-tokens", strconv.Itoa(got),
		"--restart-on-budget",
		"--restart-limit", "16",
		"--max-duration", "1740s",
	}
	if !slices.Equal(claudeGuardArgs(), want) {
		t.Errorf("claudeGuardArgs() = %v, want %v", claudeGuardArgs(), want)
	}
}

func TestGuardWrapClaudeFrontsWithFakGuardAnthropic(t *testing.T) {
	raw, _ := buildCommand("gateway", "claude")
	wrapped := guardWrap(raw, "/usr/bin/fak", "gateway", "claude", "C:/work/fak", map[string]string{})
	if wrapped[0] != "/usr/bin/fak" || wrapped[1] != "guard" {
		t.Errorf("must front with `fak guard`: %v", wrapped[:2])
	}
	if wrapped[indexOf(wrapped, "--provider")+1] != "anthropic" {
		t.Error("claude provider must be anthropic")
	}
	if wrapped[indexOf(wrapped, "--precompact-hook")+1] != "enforce" {
		t.Error("claude precompact hook must be enforced")
	}
	// ADEQUACY guardrail, not mere presence. This value seeds guard's per-session
	// ContextTokensLeft, which DebitUsage (internal/session/usage.go) draws down by
	// each turn's FULL resident window. It MUST exceed the worker's irreducible ~62K
	// baseline prompt (issue body + AGENTS/llms orientation + injected fleet memory +
	// the ~40K startup.json route blob) or every claude worker is born over-budget and
	// crashes on turn 1 — the 2026-07-05 (#2972) regression, which the previous
	// `== "48000"` assertion here actively PROTECTED. Pin a floor so a future
	// baseline-growth commit fails HERE, loudly, instead of silently crash-looping the fleet.
	budgetStr := wrapped[indexOf(wrapped, "--context-budget-tokens")+1]
	budget, err := strconv.Atoi(budgetStr)
	if err != nil {
		t.Fatalf("claude guard context budget must be a wired integer, got %q", budgetStr)
	}
	const workerBaselineFloorTokens = 62400
	if budget < workerBaselineFloorTokens {
		t.Errorf("claude guard context budget %d < worker baseline floor %d: workers would be born over-budget on turn 1 (see #2972)", budget, workerBaselineFloorTokens)
	}
	if !contains(wrapped, "--restart-on-budget") {
		t.Error("claude guard must restart on budget exhaustion")
	}
	if wrapped[indexOf(wrapped, "--restart-limit")+1] != "16" {
		t.Error("claude guard restart limit must let a healthy worker reach its wall-clock backstop (16), not die at 2 epochs")
	}
	// A graceful in-guard wall-clock backstop must front the raised restart limit so a
	// stuck worker drains cleanly (TIME_BUDGET_EXHAUSTED) before the 1800s hard-kill —
	// the belt to the raised-restart-limit suspenders (see claudeGuardMaxDuration).
	if !contains(wrapped, "--max-duration") {
		t.Error("claude guard must bound total wall-clock with --max-duration")
	}
	if md := wrapped[indexOf(wrapped, "--max-duration")+1]; md != "1740s" {
		t.Errorf("claude guard --max-duration = %q, want 1740s (defaultTimeoutS %d - %ds drain margin)", md, defaultTimeoutS, claudeGuardMaxDurationMarginS)
	}
	if !contains(wrapped, "--audit") {
		t.Error("must pass --audit")
	}
	audit := wrapped[indexOf(wrapped, "--audit")+1]
	wantSessionID := strings.TrimSuffix(filepath.Base(audit), filepath.Ext(audit))
	if wrapped[indexOf(wrapped, "--session-id")+1] != wantSessionID {
		t.Errorf("session id must derive from unique audit path: got %q want %q",
			wrapped[indexOf(wrapped, "--session-id")+1], wantSessionID)
	}
	// The raw worker argv is preserved verbatim AFTER the `--` separator.
	sep := indexOf(wrapped, "--")
	if sep < 0 || !sliceEqual(wrapped[sep+1:], raw) {
		t.Errorf("raw argv must follow `--` verbatim: sep=%d wrapped=%v", sep, wrapped)
	}
}

func TestGuardWrapNoopWithoutFakBin(t *testing.T) {
	raw, _ := buildCommand("docs", "claude")
	if got := guardWrap(raw, "", "docs", "claude", ".", map[string]string{}); !sliceEqual(got, raw) {
		t.Errorf("no fak bin -> command unchanged: %v", got)
	}
}

func TestGuardWrapOpencodeSkipsWithoutBaseURLButWrapsWithOne(t *testing.T) {
	raw, _ := buildCommand("recall", "opencode")
	// No FLEET_DOGFOOD_GUARD_BASEURL -> refuse to misroute a local-upstream worker.
	if got := guardWrap(raw, "/usr/bin/fak", "recall", "opencode", ".", map[string]string{}); !sliceEqual(got, raw) {
		t.Errorf("opencode without base url must stay unwrapped: %v", got)
	}
	// With a base URL the operator names the local upstream and we DO front it.
	wrapped := guardWrap(raw, "/usr/bin/fak", "recall", "opencode", ".",
		map[string]string{"FLEET_DOGFOOD_GUARD_BASEURL": "http://127.0.0.1:8131/v1"})
	if wrapped[0] != "/usr/bin/fak" {
		t.Errorf("opencode with base url must front with fak: %v", wrapped)
	}
	if wrapped[indexOf(wrapped, "--provider")+1] != "openai" {
		t.Error("opencode provider must be openai")
	}
	if wrapped[indexOf(wrapped, "--base-url")+1] != "http://127.0.0.1:8131/v1" {
		t.Error("base url must be forwarded")
	}
}

func TestGuardedLaunchCommandOptsOutWhenDisabled(t *testing.T) {
	raw, _ := buildCommand("gateway", "claude")
	fak := filepath.Join(t.TempDir(), "fak")
	writeFile(t, fak, "x")
	cmd, guarded := guardedLaunchCommand(raw, "gateway", "claude", "C:/work/fak",
		map[string]string{"FLEET_DOGFOOD_GUARD": "0", "FAK_BIN": fak})
	if guarded || !sliceEqual(cmd, raw) {
		t.Errorf("disabled -> unguarded raw command: guarded=%v cmd=%v", guarded, cmd)
	}
}

func TestGuardedLaunchCommandWrapsWhenEnabledAndBinPresent(t *testing.T) {
	raw, _ := buildCommand("gateway", "claude")
	fak := filepath.Join(t.TempDir(), "fak")
	writeFile(t, fak, "x")
	cmd, guarded := guardedLaunchCommand(raw, "gateway", "claude", "C:/work/fak",
		map[string]string{"FAK_BIN": fak})
	if !guarded || cmd[0] != fak || cmd[1] != "guard" {
		t.Errorf("enabled + bin -> guarded fak-fronted command: guarded=%v cmd=%v", guarded, cmd)
	}
}

func TestGuardEnvAugmentSetsTimeoutFloorsWithoutClobbering(t *testing.T) {
	env := map[string]string{"FAK_PLANNER_TIMEOUT_S": "1200"}
	guardEnvAugment(env)
	if env["FAK_PLANNER_TIMEOUT_S"] != "1200" { // explicit value kept
		t.Error("explicit planner timeout must be kept")
	}
	if env["FAK_HTTP_WRITE_TIMEOUT_S"] != "600" {
		t.Errorf("write timeout floor must be set: %q", env["FAK_HTTP_WRITE_TIMEOUT_S"])
	}
}

func TestBuildPayloadCarriesGuardedAndExplicitCommand(t *testing.T) {
	p := buildPayload("gateway", "claude", "C:/work/fak", true, nil, "",
		[]string{"fak", "guard", "--", "claude"}, true)
	if !p.Guarded {
		t.Error("payload must carry guarded=true")
	}
	if p.Command[0] != "fak" {
		t.Errorf("payload must carry the explicit fronted command: %v", p.Command)
	}
}

func writeFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}
