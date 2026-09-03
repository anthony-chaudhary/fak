// guard_test.go — parity witness for the Go dogfood-guard port, mirroring the
// guard test table in tools/dispatch_worker_test.py. Hermetic: no process is
// spawned; a real temp file stands in for the resolved `fak` binary.
package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
	"github.com/anthony-chaudhary/fak/internal/gateway"
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

// TestClaudeGuardContextBudgetDerivation locks the DERIVED budget: the per-TURN
// resident term (the ctxplan effective window ceiling, floored by the measured
// baseline) × claudeGuardTurnsPerEpoch. The seeded budget is a CUMULATIVE allowance
// (internal/session/usage.go:99 debits each turn's whole resident window), so every
// assertion here is in TURN units — the missing unit is what let the pre-fix
// `min(baseline×2, ceiling)` pass its goldens while funding two turns per child.
// Mirror: tools/dispatch_worker.py claude_guard_context_budget_tokens must yield the
// SAME integer; its parity partner test_claude_guard_context_budget_derivation_matches_go
// (in tools/dispatch_worker_test.py) pins the same 2016000 golden — update both in the
// same commit.
func TestClaudeGuardContextBudgetDerivation(t *testing.T) {
	env := ctxplan.GenericTurnEnvelope()
	ceiling := env.HardContextCap - env.OutputReserve
	// Empty workspace measures nothing → the baseline floors to claudeGuardBaselineTokens,
	// so this golden pins the hermetic shipped default independent of any on-disk
	// orientation size.
	got, err := strconv.Atoi(claudeGuardContextBudgetTokens("", "", nil))
	if err != nil {
		t.Fatalf("derived budget must be a wired integer: %v", err)
	}
	// (a) Birth-safety: strictly above the baseline (a worker is never born
	// over-budget), with real headroom, not merely +1.
	if got <= claudeGuardBaselineTokens {
		t.Errorf("derived budget %d <= baseline %d: workers would be born over-budget (see #2972)", got, claudeGuardBaselineTokens)
	}
	// (b) TURN funding — the invariant the old goldens were missing. The worst-case
	// per-turn debit is a full resident window (the ceiling, or the baseline when the
	// baseline is larger than a small model's window), so budget/perTurn is the number
	// of turns a child is guaranteed. It must be a whole epoch, not 2.
	perTurn := max(ceiling, claudeGuardBaselineTokens)
	if turns := got / perTurn; turns < claudeGuardTurnsPerEpoch {
		t.Errorf("derived budget %d funds only %d full-window turns (perTurn %d), want >= %d: a worker that cannot run an epoch dies BUDGET_CONTEXT_EXHAUSTED mid-issue",
			got, turns, perTurn, claudeGuardTurnsPerEpoch)
	}
	// (c) The reaper must not outrun the work: cmd/fak/guard_child.go reaps after
	// guardEquivalentRestartLimit = 3 identical BUDGET_CONTEXT_EXHAUSTED cycles, and
	// claudeGuardMaxDuration() bounds the run at 1740s. At the witnessed ~57s/turn a
	// worker has ~30 turns of wall clock, so 3 epochs must cover more than that or the
	// restart reaper — not --max-duration — ends every run (the 120/120 CLAIM_NO_COMMIT
	// failure mode).
	const (
		equivalentRestartLimit  = 3  // cmd/fak/guard_child.go guardEquivalentRestartLimit
		witnessedSecondsPerTurn = 57 // resolve-5103: 6 turns over 5m42s
	)
	wallClockTurns := (defaultTimeoutS - claudeGuardMaxDurationMarginS) / witnessedSecondsPerTurn
	if fundedTurns := (got / perTurn) * equivalentRestartLimit; fundedTurns <= wallClockTurns {
		t.Errorf("budget funds %d turns across %d restart epochs but the wall clock allows ~%d: the equivalent-restart reaper fires before --max-duration",
			fundedTurns, equivalentRestartLimit, wallClockTurns)
	}
	// (d) Golden lock for the shipped constants: max(200000−32000, 62000) × 12 = 2016000.
	// If this fails, someone changed the baseline, the turn count, or the ctxplan
	// envelope — update tools/dispatch_worker.py IN THE SAME COMMIT.
	if want := 2016000; got != want {
		t.Errorf("derived budget = %d, want %d; keep tools/dispatch_worker.py claude_guard_context_budget_tokens in sync", got, want)
	}
	// (e) REGRESSION TRIPWIRE: the window ceiling is a per-TURN quantity and must never
	// clamp the cumulative total again. Clamping is what pinned every child at ~2 turns
	// (min(62000×k, 168000) = 168000 for all k >= 3, so no factor could ever help).
	if got <= ceiling {
		t.Errorf("derived budget %d <= per-turn window ceiling %d: the cumulative allowance has been clamped to a per-turn window again", got, ceiling)
	}
	// (f) Non-decreasing in the baseline, and strictly rising once the baseline outgrows
	// the window ceiling (the flat-constant staleness this derivation kills).
	if bumped := deriveClaudeGuardContextBudget(claudeGuardBaselineTokens+1000, env.HardContextCap, env.OutputReserve); bumped < got {
		t.Errorf("baseline bump must not lower the derived budget: %d (baseline+1000) < %d", bumped, got)
	}
	if grown := deriveClaudeGuardContextBudget(ceiling+1000, env.HardContextCap, env.OutputReserve); grown <= got {
		t.Errorf("a baseline past the window ceiling must raise the budget: %d <= %d", grown, got)
	}
	// The CLI surface: exact flags, exact order, derived value wired in.
	want := []string{
		"--precompact-hook", "enforce",
		"--context-budget-tokens", strconv.Itoa(got),
		"--compact-history-budget", strconv.Itoa(claudeGuardCompactHistoryBudget),
		"--compact-solvency-floor", "142800",
		"--restart-on-budget",
		"--restart-limit", "16",
		"--max-duration", "1740s",
	}
	if !slices.Equal(claudeGuardArgs("", "", nil), want) {
		t.Errorf("claudeGuardArgs() = %v, want %v", claudeGuardArgs("", "", nil), want)
	}
}

// TestClaudeGuardCompactHistoryBudget pins the COMPACT shed-line fix (#4253): the
// dispatch worker must launch with a --compact-history-budget ABOVE its ~62K resident
// baseline and BELOW the drain ceiling, so compaction can actually shed and the
// ACTIVE_COMPACT_RUNAWAY hold stops arming on every worker. Guards against a silent
// regression back to the 48K interactive default that BELOW the baseline crash-loops
// the dispatcher. Parity partner: tools/dispatch_worker_test.py
// test_claude_guard_compact_history_budget (same integers).
func TestClaudeGuardCompactHistoryBudget(t *testing.T) {
	// (a) Single source of truth: the local mirror MUST equal the gateway constant that
	// exists for exactly this (reachable otherwise only via --expose-profile headless).
	if claudeGuardCompactHistoryBudget != gateway.HeadlessCompactHistoryBudget {
		t.Errorf("compact shed-line %d != gateway.HeadlessCompactHistoryBudget %d: mirror drifted",
			claudeGuardCompactHistoryBudget, gateway.HeadlessCompactHistoryBudget)
	}
	// (b) Strictly ABOVE the baseline — a shed-line at/below baseline (the 48K default)
	// can never succeed and pins the worker permanently past compact (#4253).
	if claudeGuardCompactHistoryBudget <= claudeGuardBaselineTokens {
		t.Errorf("compact shed-line %d <= baseline %d: worker can never shed under it, stays past-compact",
			claudeGuardCompactHistoryBudget, claudeGuardBaselineTokens)
	}
	// (c) Strictly raised above the broken interactive default it replaces.
	if claudeGuardCompactHistoryBudget <= gateway.DefaultCompactHistoryBudget {
		t.Errorf("compact shed-line %d not above the 48K interactive default %d",
			claudeGuardCompactHistoryBudget, gateway.DefaultCompactHistoryBudget)
	}
	// (d) REACHABILITY, not "below the drain ceiling" — the two are on different scales
	// (this shed-line is a PER-TURN instantaneous target; --context-budget-tokens is a
	// CUMULATIVE allowance), so comparing them directly is a category error and was the
	// assertion that let the composition ship broken. What must hold is that a worker
	// SITTING AT the shed line still has a whole epoch of turns funded: otherwise the
	// session dies of budget exhaustion long before compaction can fire, which is exactly
	// what resolve-5103 witnessed (compact=none / `bailed: under_budget x6` on every turn
	// while the cumulative budget drained in two).
	drain, _ := strconv.Atoi(claudeGuardContextBudgetTokens("", "", nil))
	if turns := drain / claudeGuardCompactHistoryBudget; turns < claudeGuardTurnsPerEpoch {
		t.Errorf("drain budget %d funds only %d turns at the shed-line resident %d, want >= %d: compaction is unreachable before BUDGET_CONTEXT_EXHAUSTED",
			drain, turns, claudeGuardCompactHistoryBudget, claudeGuardTurnsPerEpoch)
	}
}

// TestLaunchGoalDetachedGuardBudgetsMirrorDispatchWorker pins the THIRD launch path to the
// two guard budgets this file derives. Go (here) and Python (tools/dispatch_worker.py) were
// already pinned to each other; tools/launch_goal_detached.ps1 spawns the same kind of
// headless guarded worker through the same `fak guard` flags and was pinned to NEITHER, so
// it silently kept the pre-#2972 shape after both siblings were fixed.
//
// What that cost, and why the assertions below are the two that matter: on wave fw08081943
// (2026-08-08) all 9 detached workers died `409 ... BUDGET_CONTEXT_EXHAUSTED` 6-8 minutes
// into a 45-minute runway — ~15% of it, the exact signature both siblings' comments record
// for the starved-budget regime. The compactor was NOT at fault and the logs prove it: 37
// fires, 1.21M tokens shed, 0 anchor_starved, 0 solvency_forced, and all 65 bails were
// correct-by-design reasons (under_budget 54, too_few_msgs 10, burst_unprofitable 1). The
// launcher was simply handing guard a flat 200000 CUMULATIVE allowance against measured
// residents of 73k-294k per turn — ~2 funded turns — then reaping the worker after 3
// restart hops instead of 16.
//
// Both numbers are asserted against the DERIVATION rather than a literal so a future
// envelope change moves all three launch paths together instead of re-opening this gap.

func TestLaunchGoalDetachedSupportsExplicitProducts(t *testing.T) {
	b, err := os.ReadFile(filepath.Join("..", "..", "tools", "launch_goal_detached.ps1"))
	if err != nil {
		t.Fatal(err)
	}
	s := string(b)
	for _, want := range []string{
		"[ValidateSet('claude','codex','opencode')]",
		"[string]$Product     = 'claude'",
		"'--product', $Product",
		"$pfArgs += @('--product', $Product)",
		"if ($Product -ne 'claude')",
		"fleet-accounts exec --product '$quotedProduct'",
		"-WindowStyle Hidden",
		"LAUNCH_WITNESS pid={0}",
	} {
		if !strings.Contains(s, want) {
			t.Fatalf("launch_goal_detached.ps1 missing provider-neutral contract %q", want)
		}
	}
	if strings.Contains(s, "--product', 'claude'") {
		t.Fatal("account resolution remains hard-coded to Claude")
	}
}

func TestLaunchGoalDetachedGuardBudgetsMirrorDispatchWorker(t *testing.T) {
	path := filepath.Join("..", "..", "tools", "launch_goal_detached.ps1")
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	ps1 := string(b)

	// (a) The cumulative drain allowance. The launcher declares it as a param default.
	m := regexp.MustCompile(`\[int\]\$ContextBudgetTokens\s*=\s*(\d+)`).FindStringSubmatch(ps1)
	if m == nil {
		t.Fatalf("%s: no [int]$ContextBudgetTokens default found — did the param rename?", path)
	}
	gotBudget, err := strconv.Atoi(m[1])
	if err != nil {
		t.Fatalf("parse ContextBudgetTokens %q: %v", m[1], err)
	}
	wantBudget, err := strconv.Atoi(claudeGuardContextBudgetTokens("", "", nil))
	if err != nil {
		t.Fatalf("parse derived budget: %v", err)
	}
	if gotBudget != wantBudget {
		t.Errorf("%s -ContextBudgetTokens = %d, want %d (the derived per-epoch allowance):\n"+
			"  the flag is a CUMULATIVE allowance, so it funds budget/resident turns — %d at the\n"+
			"  %d shed-line resident funds %d turn(s), and a worker that cannot run an epoch dies\n"+
			"  BUDGET_CONTEXT_EXHAUSTED mid-issue. Keep it equal to claudeGuardContextBudgetTokens.",
			path, gotBudget, wantBudget, gotBudget, claudeGuardCompactHistoryBudget,
			gotBudget/claudeGuardCompactHistoryBudget)
	}

	// (b) The restart limit. A low cap is not a safety margin — it is a second, tighter
	// deadline that reaps a HEALTHY worker long before its wall clock.
	m = regexp.MustCompile(`"--restart-limit",\s*"(\d+)"`).FindStringSubmatch(ps1)
	if m == nil {
		t.Fatalf("%s: no --restart-limit argv pair found", path)
	}
	if m[1] != claudeGuardRestartLimit {
		t.Errorf("%s --restart-limit = %s, want %s (claudeGuardRestartLimit): a worker relaunches\n"+
			"  every ~%d turns under the derived budget, so a low cap ends the run before the\n"+
			"  --max-duration wall clock does.", path, m[1], claudeGuardRestartLimit, claudeGuardTurnsPerEpoch)
	}

	// (c) The property both numbers exist to protect, asserted end-to-end: a worker sitting
	// AT the shed line must still have a whole epoch funded. This is what actually failed in
	// fw08081943, and it fails for any future pair of values that individually look fine.
	if turns := gotBudget / claudeGuardCompactHistoryBudget; turns < claudeGuardTurnsPerEpoch {
		t.Errorf("%s funds only %d turn(s) at the %d shed-line resident, want >= %d: compaction is\n"+
			"  unreachable before the cumulative budget drains, which is the whole-wave death mode.",
			path, turns, claudeGuardCompactHistoryBudget, claudeGuardTurnsPerEpoch)
	}
}

// TestDetachedLauncherDefaultsSpawnFromABareInvocation pins the three param defaults that
// decide whether `launch_wave_detached.ps1 -Count 30 -Launch` spawns a wave or zero workers
// (#5895). A cron, a CI step and a refill loop all issue the bare form, so a default that
// only works when overridden is a default that does not work.
//
// The failure was silent in the worst way: each spawn threw `pointer file not found` and the
// launcher reported a wave, so the run read as launched-and-unproductive rather than never
// started. Nothing in the tracked tree asserted these strings — the same coverage gap that
// let the guard budget drift in TestLaunchGoalDetachedGuardBudgetsMirrorDispatchWorker.
func TestWaveLauncherRefillsAndReconcilesProcessCensus(t *testing.T) {
	path := filepath.Join("..", "..", "tools", "launch_wave_detached.ps1")
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	body := string(b)
	for _, want := range []string{
		"[int]$RefillCadenceSeconds = 60",
		"[int]$RefillForMinutes = 240",
		"while ($remaining -gt 0 -and [datetime]::UtcNow -lt $deadline)",
		"[datetime]$RefillDeadlineUtc = [datetime]::MinValue",
		"'-RefillDeadlineUtc', $RefillDeadlineUtc.ToString('o')",
		"$refillEligible = $Launch",
		"WAVE WAIT         initial allocation empty",
		"Start-Sleep -Seconds $RefillCadenceSeconds",
		"'-NoRefill', '-Launch'",
		"WAVE CENSUS",
		"os_worker_procs=",
		"seat_free=",
		"process_consistency=",
	} {
		if !strings.Contains(body, want) {
			t.Errorf("wave launcher missing refill/census contract %q", want)
		}
	}
}

func TestWaveLauncherRefillsAfterCapacityOpensWithoutOverlaunch(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("launch_wave_detached.ps1 is a Windows launcher")
	}
	powershell, err := exec.LookPath("powershell.exe")
	if err != nil {
		t.Skip("powershell.exe unavailable")
	}

	repoRoot, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	tmp := t.TempDir()
	tools := filepath.Join(tmp, "tools")
	if err := os.MkdirAll(tools, 0o755); err != nil {
		t.Fatal(err)
	}
	copyFile := func(dst, src string) {
		t.Helper()
		body, readErr := os.ReadFile(src)
		if readErr != nil {
			t.Fatal(readErr)
		}
		if writeErr := os.WriteFile(dst, body, 0o644); writeErr != nil {
			t.Fatal(writeErr)
		}
	}
	copyFile(filepath.Join(tools, "launch_wave_detached.ps1"), filepath.Join(repoRoot, "tools", "launch_wave_detached.ps1"))
	if err := os.MkdirAll(filepath.Join(tmp, "docs"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(tmp, "docs", "pointer.md"), []byte("goal\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	write := func(name, body string) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(tools, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("launch_goal_detached.ps1", `param(
  [string]$PointerFile, [string]$Workspace, [string]$LogDir, [string]$Account,
  [string]$WorkKind, [string]$FakExe, [int]$PreflightMaxWorkers,
  [switch]$AllowTierFallback, [switch]$SkipPreflight, [switch]$ExtendStanding,
  [string]$StandingTaskName
)
Add-Content -Path (Join-Path $PSScriptRoot 'dispatches.txt') -Value $Account
`)
	write("fake-fak.ps1", `param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Rest)
if ($Rest -contains '-h') { Write-Output '  --count int'; exit 0 }
$state = Join-Path $PSScriptRoot 'allocation-count.txt'
$n = if (Test-Path $state) { [int](Get-Content $state) } else { 0 }
$n++; Set-Content $state $n
$countAt = [Array]::IndexOf($Rest, '--count'); $asked = [int]$Rest[$countAt + 1]
$grant = if ($n -eq 1) { 0 } elseif ($n -eq 2) { [Math]::Min(2, $asked) } else { $asked }
if ($grant -eq 0) {
  Write-Output ('{"ok":false,"requested":' + $asked + ',"granted":0,"shortfall":' + $asked + ',"reason":"NO_FREE_SEATS","lanes":[]}')
  exit 0
}
$lanes = @()
1..$grant | ForEach-Object { $lanes += @{tag="acct-$n-$_"; pool="pool-$n-$_"; tier=1; session_cap=1} }
@{ok=$true; requested=$asked; granted=$grant; shortfall=($asked-$grant); target_tier=1; distinct_pools=$grant; lanes=$lanes} |
  ConvertTo-Json -Depth 5 -Compress
`)
	write("python.cmd", `@echo off
set args=%*
echo %args% | findstr /c:"dispatch_preflight.py" >nul
if not errorlevel 1 (
  echo {"verdict":"SPAWN_OK","reason":"capacity available","cap":10,"live":0,"headroom":10,"os_worker_procs":0,"seat":{"total":10,"free":10,"leased":0,"unattributed_live":0}}
  exit /b 0
)
echo {"schema":"launch-admission/1","verdict":"ADMIT","reason":"CLEAR","failopen":false}
exit /b 0
`)

	launcher := filepath.Join(tools, "launch_wave_detached.ps1")
	fakeFak := filepath.Join(tools, "fake-fak.ps1")
	cmd := exec.Command(powershell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", launcher,
		"-Count", "5", "-PointerFile", "docs/pointer.md", "-Workspace", tmp, "-LogDir", filepath.Join(tmp, "logs"),
		"-WorkKind", "engineering", "-Product", "claude", "-FakExe", fakeFak,
		"-RefillCadenceSeconds", "1", "-RefillForMinutes", "1", "-Launch")
	cmd.Env = append(os.Environ(), "PATH="+tools+string(os.PathListSeparator)+os.Getenv("PATH"), "FAK_LAUNCH_SPAWN_PACING_MS=1")
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("launcher failed: %v\n%s", err, output)
	}
	body := string(output)
	if !strings.Contains(body, "WAVE WAIT         initial allocation empty") ||
		!strings.Contains(body, "WAVE REFILL DONE  launched requested total=5") {
		t.Fatalf("launcher did not wait then complete its refill:\n%s", body)
	}
	dispatches, err := os.ReadFile(filepath.Join(tools, "dispatches.txt"))
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Fields(string(dispatches))
	if len(lines) != 5 {
		t.Fatalf("dispatched %d workers, want exactly requested total 5 (no under/overlaunch): %q\n%s", len(lines), lines, body)
	}
	calls, err := os.ReadFile(filepath.Join(tools, "allocation-count.txt"))
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.TrimSpace(string(calls)); got != "3" {
		t.Fatalf("allocation passes=%s, want 3 (empty, partial 2, remaining 3)", got)
	}
}

func TestDetachedLauncherDefaultsSpawnFromABareInvocation(t *testing.T) {
	repoRoot := filepath.Join("..", "..")
	for _, name := range []string{"launch_goal_detached.ps1", "launch_wave_detached.ps1"} {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(repoRoot, "tools", name)
			b, err := os.ReadFile(path)
			if err != nil {
				t.Fatalf("read %s: %v", path, err)
			}
			ps1 := string(b)

			// (a) The default pointer must name a file that is actually in the tree. The
			// previous default died to a rename, not a typo, so pin existence rather than
			// any particular name.
			m := regexp.MustCompile(`\[string\]\$PointerFile\s*=\s*"([^"]+)"`).FindStringSubmatch(ps1)
			if m == nil {
				t.Fatalf("%s: no [string]$PointerFile default found — did the param rename?", name)
			}
			pointer := filepath.Join(repoRoot, filepath.FromSlash(m[1]))
			body, err := os.ReadFile(pointer)
			if err != nil {
				t.Fatalf("%s: default -PointerFile %q is not in the tree (%v): every spawn throws\n"+
					"  `pointer file not found` and the wave is zero workers.", name, m[1], err)
			}

			// (b) The rendered /goal condition must clear the launcher's own 4000 cap. Count
			// RUNES, not bytes: PowerShell compares $cond.Length, which is chars, and this
			// pointer carries multibyte glyphs — byte length overstates it by ~28.
			if cond := len([]rune("/goal " + string(body))); cond > 4000 {
				t.Errorf("%s: default -PointerFile %q renders a %d-char /goal condition (>4000 cap):\n"+
					"  the launcher throws per spawn, so the wave is zero workers. Shrink the pointer.",
					name, m[1], cond)
			}

			// (c) No default may hardcode an absolute path. The old -Workspace named a sibling
			// clone whose missing tools/proc_resource_guard.py fail-safed the preflight to
			// REFUSE_INSPECT (cap=4, granted=0), and the old -LogDir pinned that same sibling's
			// .goal-runs — which is how a liveness probe reads ANOTHER wave's artifacts and a
			// recycled id lets a predecessor vouch for a corpse. Both must derive.
			absDefault := regexp.MustCompile(`\[string\]\$(Workspace|LogDir)\s*=\s*"([a-zA-Z]:[\\/]|\\\\)[^"]*"`)
			for _, bad := range absDefault.FindAllStringSubmatch(ps1, -1) {
				t.Errorf("%s: -%s defaults to the absolute path %q; derive it from $PSScriptRoot\n"+
					"  (tools/ -> repo root) or from $Workspace so a bare invocation targets THIS checkout.",
					name, bad[1], strings.TrimSpace(bad[0]))
			}
		})
	}
}

// TestClaudeGuardSolvencyFloorDerivation pins the CONTEXT-SOLVENCY floor the launch path
// hands the gateway as --compact-solvency-floor. The gateway cannot derive this itself —
// it prices a compaction burst in cache dollars and never sees a window SIZE — so the
// launcher, which does know the envelope, must supply it or the override stays inert and
// the measured fire-rate inversion (33% at 96-125k occupancy down to 0% above 170k, with
// 100% of firing traces never firing again) simply continues.
//
// Parity partner: tools/dispatch_worker_test.py
// test_claude_guard_compact_solvency_floor (same integers, same argv position).
func TestClaudeGuardSolvencyFloorDerivation(t *testing.T) {
	env := ctxplan.EnvelopeForModel("")
	usable := env.HardContextCap - env.OutputReserve
	got := deriveClaudeGuardSolvencyFloor(env.HardContextCap, env.OutputReserve)

	// (a) Golden lock: 85% of (200000 − 32000). Keep identical to the Python mirror.
	if want := 142800; got != want {
		t.Errorf("derived solvency floor = %d, want %d; keep tools/dispatch_worker.py in sync", got, want)
	}
	// (b) STRICTLY ABOVE the compact shed-line. A floor at or below it would force a fire
	// on essentially every past-budget turn and discard the cache economics wholesale —
	// the override is a last resort, not a replacement for the gate.
	if got <= claudeGuardCompactHistoryBudget {
		t.Errorf("solvency floor %d <= compact shed-line %d: the override would swallow the burst gate entirely",
			got, claudeGuardCompactHistoryBudget)
	}
	// (c) STRICTLY BELOW the usable window, with real headroom left for the forced burst
	// to land and repay. A floor at the ceiling rings the alarm after the wall.
	if got >= usable {
		t.Errorf("solvency floor %d >= usable window %d: the forced fire arrives too late to help", got, usable)
	}
	if headroom := usable - got; headroom < 20000 {
		t.Errorf("solvency floor %d leaves only %d tokens of usable window; want >= 20000 for the forced burst to land",
			got, headroom)
	}
	// (d) Fail-safe: a degenerate envelope DISARMS the override (0) rather than forcing a
	// fire on every turn. Zero is the documented "pure economics, byte-for-byte" value.
	if disarmed := deriveClaudeGuardSolvencyFloor(1000, 4000); disarmed != 0 {
		t.Errorf("a degenerate envelope must disarm the override, got %d", disarmed)
	}
	// (e) Model-aware: a smaller window yields a proportionally smaller floor, never the
	// generic constant. (ctxplan's fable row is the small-window case.)
	small := deriveClaudeGuardSolvencyFloor(64000, 32000)
	if small <= 0 || small >= got {
		t.Errorf("a small-window model must derive its own lower floor, got %d (generic %d)", small, got)
	}
	// (f) Actually WIRED into the argv — a derived constant nobody passes is inert.
	args := claudeGuardArgs("", "", nil)
	i := slices.Index(args, "--compact-solvency-floor")
	if i < 0 || i+1 >= len(args) {
		t.Fatalf("claudeGuardArgs() does not carry --compact-solvency-floor: %v", args)
	}
	if args[i+1] != strconv.Itoa(got) {
		t.Errorf("--compact-solvency-floor argv = %q, want %q", args[i+1], strconv.Itoa(got))
	}
}

// TestMeasureLaunchBaselineFloorsAndTracks pins the measurement seam (#3522): the
// baseline is measured from the real launch constituents but can never fall below the
// hand-measured floor, and it RISES when the measured prompt outgrows the floor —
// closing the "nothing measures the real prompt" gap without ever reintroducing the
// born-over-budget trap. Parity partner: tools/dispatch_worker_test.py
// test_measure_launch_baseline_floors_and_tracks (same fixtures, same integers).
func TestMeasureLaunchBaselineFloorsAndTracks(t *testing.T) {
	// (a) approx ruler matches the codebase (bytes+3)/4.
	if got := approxTokensFromBytes(41657); got != (41657+3)/4 {
		t.Errorf("approxTokensFromBytes(41657) = %d, want %d", got, (41657+3)/4)
	}
	if got := approxTokensFromBytes(0); got != 0 {
		t.Errorf("approxTokensFromBytes(0) = %d, want 0", got)
	}
	// (b) A degenerate (empty) measurement floors to the shipped baseline — never below.
	if got := resolveClaudeGuardBaseline(measureLaunchBaselineTokens(nil)); got != claudeGuardBaselineTokens {
		t.Errorf("empty measurement must floor to %d, got %d", claudeGuardBaselineTokens, got)
	}
	// (c) A small measured prompt (below the floor) still floors — no regression.
	small := map[string]int{"AGENTS.md": 41657, "llms.txt": 57230} // ~24722 tokens < floor
	if got := resolveClaudeGuardBaseline(measureLaunchBaselineTokens(small)); got != claudeGuardBaselineTokens {
		t.Errorf("sub-floor measurement (%d tokens) must floor to %d, got %d",
			measureLaunchBaselineTokens(small), claudeGuardBaselineTokens, got)
	}
	// (d) A measured prompt that OUTGROWS the floor raises the baseline (the trap the
	// frozen constant left open): a startup blob big enough to cross 62000 tokens.
	grown := map[string]int{"AGENTS.md": 41657, "llms.txt": 57230, "startup_bundle": 200000}
	measured := measureLaunchBaselineTokens(grown)
	if measured <= claudeGuardBaselineTokens {
		t.Fatalf("fixture must exceed the floor: measured %d <= %d", measured, claudeGuardBaselineTokens)
	}
	if got := resolveClaudeGuardBaseline(measured); got != measured {
		t.Errorf("supra-floor measurement must pass through, got %d want %d", got, measured)
	}
	// (e) The derived budget stays birth-safe on the measured path AND still funds a
	// whole epoch of turns at that measured resident — the turn-unit check, not the old
	// window clamp (a cumulative allowance is never bounded by a per-turn window).
	env := ctxplan.GenericTurnEnvelope()
	ceiling := env.HardContextCap - env.OutputReserve
	budget := deriveClaudeGuardContextBudget(resolveClaudeGuardBaseline(measured), env.HardContextCap, env.OutputReserve)
	if budget <= measured {
		t.Errorf("measured budget %d must exceed measured baseline %d", budget, measured)
	}
	if turns := budget / max(measured, ceiling); turns < claudeGuardTurnsPerEpoch {
		t.Errorf("measured budget %d funds only %d turns at resident %d, want >= %d", budget, turns, max(measured, ceiling), claudeGuardTurnsPerEpoch)
	}
}

// TestGatherLaunchConstituentBytesReadsWorkspaceAndFloors witnesses the live gather:
// it sizes real files under a temp workspace and folds a startup bundle named via env,
// and an empty workspace measures nothing (the hermetic default → floor).
func TestGatherLaunchConstituentBytesReadsWorkspaceAndFloors(t *testing.T) {
	ws := t.TempDir()
	writeFile(t, filepath.Join(ws, "AGENTS.md"), strings.Repeat("a", 400))
	writeFile(t, filepath.Join(ws, "llms.txt"), strings.Repeat("b", 800))
	bundle := filepath.Join(ws, "run.startup.json")
	writeFile(t, bundle, strings.Repeat("c", 1200))

	got := gatherLaunchConstituentBytes(ws, map[string]string{launchStartupBundleEnv: bundle})
	if got["AGENTS.md"] != 400 || got["llms.txt"] != 800 {
		t.Errorf("orientation sizes wrong: %v", got)
	}
	if got["startup_bundle"] != 1200 {
		t.Errorf("startup bundle size = %d, want 1200", got["startup_bundle"])
	}
	if _, ok := got["MEMORY.md"]; ok {
		t.Errorf("absent MEMORY.md must not appear: %v", got)
	}
	// Empty workspace is the hermetic default: nothing measured, budget floors.
	if len(gatherLaunchConstituentBytes("", nil)) != 0 {
		t.Error("empty workspace must gather nothing")
	}
	base, _ := measuredClaudeGuardBaseline("", nil)
	if base != claudeGuardBaselineTokens {
		t.Errorf("empty-workspace baseline = %d, want floor %d", base, claudeGuardBaselineTokens)
	}
}

func TestGuardWrapClaudeFrontsWithFakGuardAnthropic(t *testing.T) {
	raw, _ := buildCommand("gateway", "claude")
	wrapped := guardWrap(raw, "/usr/bin/fak", "gateway", "claude", "C:/work/fak", "", map[string]string{})
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

func TestGuardWrapCodexInjectsNativeCompactLimit(t *testing.T) {
	raw := []string{"codex", "exec", "work"}
	wrapped := guardWrap(raw, "/usr/bin/fak", "gateway", "codex", ".", "", map[string]string{
		"FLEET_DOGFOOD_GUARD_BASEURL": "http://127.0.0.1:8131/v1",
	})
	joined := strings.Join(wrapped, " ")
	want := "-- codex -c model_auto_compact_token_limit=96000 exec work"
	if !strings.Contains(joined, want) {
		t.Fatalf("guardWrap(codex) = %q, want substring %q", joined, want)
	}
}

func TestGuardWrapNoopWithoutFakBin(t *testing.T) {
	raw, _ := buildCommand("docs", "claude")
	if got := guardWrap(raw, "", "docs", "claude", ".", "", map[string]string{}); !sliceEqual(got, raw) {
		t.Errorf("no fak bin -> command unchanged: %v", got)
	}
}

func TestGuardWrapOpencodeSkipsWithoutBaseURLButWrapsWithOne(t *testing.T) {
	raw, _ := buildCommand("recall", "opencode")
	// No FLEET_DOGFOOD_GUARD_BASEURL -> refuse to misroute a local-upstream worker.
	if got := guardWrap(raw, "/usr/bin/fak", "recall", "opencode", ".", "", map[string]string{}); !sliceEqual(got, raw) {
		t.Errorf("opencode without base url must stay unwrapped: %v", got)
	}
	// With a base URL the operator names the local upstream and we DO front it.
	wrapped := guardWrap(raw, "/usr/bin/fak", "recall", "opencode", ".", "", map[string]string{"FLEET_DOGFOOD_GUARD_BASEURL": "http://127.0.0.1:8131/v1"})
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

func TestOpencodeCompactShedLine(t *testing.T) {
	// Go half of the OpenCode-native compact shed-line parity (#4661).
	// Mirror of tools/dispatch_worker_test.py test_opencode_compact_shed_line.
	const shed = opencodeCompactShedLineTokens
	if shed != 96000 {
		t.Fatalf("opencode shed line diverged from 96000: %d", shed)
	}
	if shed != claudeGuardCompactHistoryBudget {
		t.Fatalf("opencode shed line %d != claude budget %d", shed, claudeGuardCompactHistoryBudget)
	}
	if shed != codexCompactTokenLimit {
		t.Fatalf("opencode shed line %d != codex limit %d", shed, codexCompactTokenLimit)
	}

	overlay := opencodeCompactionOverlay([]string{"opencode", "run", "-m", "zai-coding-plan/glm-5.2", "dispatch"})
	if overlay == nil {
		t.Fatal("expected compaction overlay for glm-5.2")
	}

	providerMap, _ := overlay["provider"].(map[string]any)
	zaiMap, _ := providerMap["zai-coding-plan"].(map[string]any)
	modelsMap, _ := zaiMap["models"].(map[string]any)
	glm52Map, _ := modelsMap["glm-5.2"].(map[string]any)
	limitMap, _ := glm52Map["limit"].(map[string]any)

	compactionMap, _ := overlay["compaction"].(map[string]any)
	auto, _ := compactionMap["auto"].(bool)
	reserved, _ := compactionMap["reserved"].(int)

	input, _ := limitMap["input"].(int)
	context, _ := limitMap["context"].(int)
	output, _ := limitMap["output"].(int)

	if !auto {
		t.Errorf("compaction.auto must be true")
	}
	if input-reserved != shed {
		t.Errorf("derived opencode compaction trigger input (%d) - reserved (%d) != shed (%d)", input, reserved, shed)
	}
	if context != 1000000 {
		t.Errorf("expected context 1000000, got %d", context)
	}
	if output != 131072 {
		t.Errorf("expected output 131072, got %d", output)
	}
	if shed >= context-output {
		t.Errorf("shed line %d must be less than context - output (%d)", shed, context-output)
	}

	defaultArgv, _ := buildCommand("recall", "opencode")
	defaultOverlay := opencodeCompactionOverlay(defaultArgv)
	if defaultOverlay == nil {
		t.Fatal("expected default overlay for recall opencode")
	}
	dProviderMap, _ := defaultOverlay["provider"].(map[string]any)
	dZaiMap, _ := dProviderMap["zai-coding-plan"].(map[string]any)
	dModelsMap, _ := dZaiMap["models"].(map[string]any)
	if _, ok := dModelsMap["glm-5.2"]; !ok {
		t.Error("expected glm-5.2 in default models")
	}
	for name, mAny := range dModelsMap {
		mMap, _ := mAny.(map[string]any)
		lMap, _ := mMap["limit"].(map[string]any)
		if lInput, _ := lMap["input"].(int); lInput != shed {
			t.Errorf("%s missed shed line: input=%d", name, lInput)
		}
	}
}

func TestGuardedLaunchCommandOptsOutWhenDisabled(t *testing.T) {
	raw, _ := buildCommand("gateway", "claude")
	fak := filepath.Join(t.TempDir(), "fak")
	writeFile(t, fak, "x")
	cmd, guarded := guardedLaunchCommand(raw, "gateway", "claude", "C:/work/fak", "",
		map[string]string{"FLEET_DOGFOOD_GUARD": "0", "FAK_BIN": fak})
	if guarded || !sliceEqual(cmd, raw) {
		t.Errorf("disabled -> unguarded raw command: guarded=%v cmd=%v", guarded, cmd)
	}
}

func TestGuardedLaunchCommandWrapsWhenEnabledAndBinPresent(t *testing.T) {
	raw, _ := buildCommand("gateway", "claude")
	fak := filepath.Join(t.TempDir(), "fak")
	writeFile(t, fak, "x")
	cmd, guarded := guardedLaunchCommand(raw, "gateway", "claude", "C:/work/fak", "",
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

func modelBudgetForBaseline(model string, baseline int) (int, int) {
	e := ctxplan.EnvelopeForModel(model)
	return baseline, deriveClaudeGuardContextBudget(baseline, e.HardContextCap, e.OutputReserve)
}

func TestClaudeGuardBudgetWorkerModelAware(t *testing.T) {
	models := []string{"fable", "claude-3-5-haiku", "gpt-5.3-codex", "claude-opus-4-8", "unknown-model"}
	got := map[string]int{}
	for _, model := range models {
		_, got[model] = modelBudgetForBaseline(model, 62000)
	}
	if got["fable"] == got["claude-3-5-haiku"] || got["claude-3-5-haiku"] == got["gpt-5.3-codex"] {
		t.Fatalf("model budgets are not distinct: %v", got)
	}
	generic := ctxplan.GenericTurnEnvelope()
	wantOpus := deriveClaudeGuardContextBudget(62000, generic.HardContextCap, generic.OutputReserve)
	if got["claude-opus-4-8"] != wantOpus {
		t.Fatalf("opus budget=%d, want historical generic %d", got["claude-opus-4-8"], wantOpus)
	}
	if got["unknown-model"] != wantOpus {
		t.Fatalf("unknown budget=%d, want generic fallback %d", got["unknown-model"], wantOpus)
	}
}
