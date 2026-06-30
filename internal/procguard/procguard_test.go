package procguard

import (
	"encoding/json"
	"testing"
)

func ip(n int) *int       { return &n }
func fp(f float64) *float64 { return &f }

func TestClassifyResourceLevel(t *testing.T) {
	procs := []Proc{
		{PID: 10, Name: "llama-cli", Threads: ip(129427)},
		{PID: 11, Name: "calm", Threads: ip(50)},
		{PID: 12, Name: "leaky", Handles: ip(90000)},
		{PID: 13, Name: "fat", WSMB: ip(40000)},
	}
	th := Thresholds{MaxThreads: 2000, MaxHandles: 50000, MaxWSMB: 20000}
	got := Classify(procs, th, nil, nil)
	if len(got) != 3 {
		t.Fatalf("want 3 flagged, got %d: %+v", len(got), got)
	}
	// Loudest (highest thread count) first.
	if got[0].PID != 10 {
		t.Fatalf("want pid 10 first (highest threads), got %d", got[0].PID)
	}
	for _, r := range got {
		if r.PID == 11 {
			t.Fatalf("calm process should not be flagged")
		}
	}
}

func TestClassifyMissingDimensionIsSkipped(t *testing.T) {
	// A nil dimension (collector could not read it) must never be a breach.
	procs := []Proc{{PID: 1, Name: "unknown"}}
	got := Classify(procs, DefaultThresholds(), nil, nil)
	if len(got) != 0 {
		t.Fatalf("nil dimensions must not flag: %+v", got)
	}
}

func TestClassifyAllowExempts(t *testing.T) {
	procs := []Proc{{PID: 5, Name: "BigBuild", Threads: ip(9999)}}
	got := Classify(procs, Thresholds{MaxThreads: 2000}, nil, []string{"bigbuild"})
	if len(got) != 0 {
		t.Fatalf("allow-listed name must be exempt: %+v", got)
	}
}

func TestClassifyProtectedBit(t *testing.T) {
	procs := []Proc{{PID: 4, Name: "System", Threads: ip(9999)}}
	got := Classify(procs, Thresholds{MaxThreads: 2000}, nil, nil)
	if len(got) != 1 || !got[0].Protected {
		t.Fatalf("System breach must be flagged AND protected: %+v", got)
	}
}

func TestCPUDimensionFlagsCorePin(t *testing.T) {
	procs := []Proc{
		{PID: 20, Name: "spin", Threads: ip(2), CPUPct: fp(99)},  // single-threaded core pin
		{PID: 21, Name: "idle", Threads: ip(2), CPUPct: fp(3)},
	}
	th := Thresholds{MaxThreads: 2000, MaxCPUPct: 90}
	got := Classify(procs, th, nil, nil)
	if len(got) != 1 || got[0].PID != 20 {
		t.Fatalf("want only the core-pin flagged: %+v", got)
	}
	if got[0].CPUPct == nil || *got[0].CPUPct != 99 {
		t.Fatalf("cpu_pct must ride the finding: %+v", got[0])
	}
}

func TestClassifyOrphans(t *testing.T) {
	live := map[int]bool{100: true} // owner 999 is gone
	procs := []Proc{
		{PID: 30, Name: "python", PPID: ip(999), Cmdline: "python -m dos_mcp.server"},
		{PID: 31, Name: "python", PPID: ip(100), Cmdline: "python -m dos_mcp.server"}, // owner alive -> spared
		{PID: 32, Name: "pwsh", PPID: ip(1), AgeSec: ip(3600)},                        // idle shell, 0 kids, aged
	}
	counts := ChildCounts(procs)
	got := ClassifyOrphans(procs, live, counts, DefaultOrphanPatterns, DefaultIdleShellNames, 1800, true, nil, nil)
	flaggedPIDs := map[int]string{}
	for _, r := range got {
		flaggedPIDs[r.PID] = r.Kind
	}
	if k, ok := flaggedPIDs[30]; !ok || k != "orphan-helper" {
		t.Fatalf("orphaned helper (owner gone) must flag as orphan-helper: %+v", got)
	}
	if _, ok := flaggedPIDs[31]; ok {
		t.Fatalf("helper with live owner must be spared: %+v", got)
	}
	if k, ok := flaggedPIDs[32]; !ok || k != "idle-shell" {
		t.Fatalf("aged idle shell with 0 children must flag as idle-shell: %+v", got)
	}
}

func TestOwnerAlivePIDReuseSafe(t *testing.T) {
	// A reused parent pid that now names a live process reads as alive -> spared.
	live := map[int]bool{500: true}
	procs := []Proc{{PID: 40, Name: "python", PPID: ip(500), Cmdline: "python -m dos_mcp.server"}}
	got := ClassifyOrphans(procs, live, ChildCounts(procs), DefaultOrphanPatterns, nil, 0, false, nil, nil)
	if len(got) != 0 {
		t.Fatalf("reused-pid owner reads alive; helper must be spared: %+v", got)
	}
}

func TestMergeUnionsReasons(t *testing.T) {
	resource := []Finding{{PID: 50, Name: "x", Reasons: []string{"threads 3000 > 2000"}, Protected: false}}
	orphan := []Finding{{PID: 50, Name: "x", Reasons: []string{"orphaned helper: owner pid 9 not alive"}, Kind: "orphan-helper", Protected: false}}
	merged := mergeFlagged(resource, orphan)
	if len(merged) != 1 {
		t.Fatalf("same pid must collapse to one row: %+v", merged)
	}
	if len(merged[0].Reasons) != 2 {
		t.Fatalf("reasons must be unioned: %+v", merged[0].Reasons)
	}
	if merged[0].Kind != "orphan-helper" {
		t.Fatalf("kind must carry over: %+v", merged[0])
	}
}

func TestBuildProtectedDoesNotFlipOK(t *testing.T) {
	// A protected-only breach is listed but must NOT make ok=false (control-pane
	// must not perpetually ACTION on a transient System thread-pool spike).
	procs := []Proc{{PID: 4, Name: "System", Threads: ip(9999)}}
	p := Build(procs, Options{Thresholds: Thresholds{MaxThreads: 2000}, Platform: "test"})
	if !p.OK {
		t.Fatalf("protected-only breach must keep ok=true: %+v", p)
	}
	if p.FlaggedCount != 1 || p.ActionableFlaggedCount != 0 {
		t.Fatalf("want flagged=1 actionable=0, got flagged=%d actionable=%d", p.FlaggedCount, p.ActionableFlaggedCount)
	}
}

func TestBuildActionableFlipsOK(t *testing.T) {
	procs := []Proc{{PID: 60, Name: "llama-cli", Threads: ip(99999)}}
	p := Build(procs, Options{Thresholds: Thresholds{MaxThreads: 2000}, Platform: "test"})
	if p.OK {
		t.Fatalf("actionable runaway must set ok=false")
	}
}

func TestBuildCollectErrorIsAction(t *testing.T) {
	p := Build(nil, Options{Thresholds: DefaultThresholds(), CollectError: "ps: boom", Platform: "test"})
	if p.OK {
		t.Fatalf("a failed collector must not report a clean host")
	}
	if p.NextAction == "" {
		t.Fatalf("collect error must carry a next_action")
	}
}

func TestEnactReapsRunawayButSkipsProtected(t *testing.T) {
	killed := map[int]bool{}
	killer := func(pid int) (bool, string) { killed[pid] = true; return true, "ok" }
	procs := []Proc{
		{PID: 70, Name: "llama-cli", Threads: ip(99999)},
		{PID: 4, Name: "System", Threads: ip(99999)},
	}
	p := Build(procs, Options{
		Thresholds: Thresholds{MaxThreads: 2000}, Enact: true, Killer: killer, Platform: "test",
	})
	if !killed[70] {
		t.Fatalf("non-protected runaway must be reaped")
	}
	if killed[4] {
		t.Fatalf("protected System must NEVER be reaped")
	}
	for _, r := range p.Flagged {
		if r.PID == 4 && r.Action != "protected-skip" {
			t.Fatalf("System must be protected-skip, got %q", r.Action)
		}
	}
}

func TestEnactCPUOnlyGatedByStreak(t *testing.T) {
	killer := func(int) (bool, string) { return true, "ok" }
	procs := []Proc{{PID: 80, Name: "spin", Threads: ip(2), CPUPct: fp(99), Start: "2026-01-01T00:00:00Z"}}
	th := Thresholds{MaxThreads: 2000, MaxCPUPct: 90}

	// First run: streak becomes 1, confirm=2 -> NOT reaped, surfaced as cpu-unconfirmed.
	p1 := Build(procs, Options{Thresholds: th, Enact: true, Killer: killer, CPUReapConfirm: 2, Platform: "test"})
	if p1.Flagged[0].Action != "cpu-unconfirmed" {
		t.Fatalf("CPU-only pin below confirm must be cpu-unconfirmed, got %q", p1.Flagged[0].Action)
	}
	if len(p1.Enacted) != 0 {
		t.Fatalf("nothing should be reaped yet: %+v", p1.Enacted)
	}

	// Second run carrying the prior streak: now confirmed -> reaped.
	p2 := Build(procs, Options{Thresholds: th, Enact: true, Killer: killer, CPUReapConfirm: 2, CPUStreaksPrev: p1.CPUStreaks, Platform: "test"})
	if p2.Flagged[0].Action != "killed" {
		t.Fatalf("confirmed CPU pin must be killed on 2nd run, got %q", p2.Flagged[0].Action)
	}
}

func TestStreakKeyReuseSafe(t *testing.T) {
	// Same pid, different start time -> different key -> fresh streak.
	if StreakKey(99, "A") == StreakKey(99, "B") {
		t.Fatalf("a recycled pid (different start) must get a distinct streak key")
	}
	prev := map[string]int{StreakKey(99, "A"): 5}
	bumped := bumpStreaks(prev, []string{StreakKey(99, "B")})
	if bumped[StreakKey(99, "B")] != 1 {
		t.Fatalf("reused pid must start from 1, not inherit 5: %+v", bumped)
	}
	if _, ok := bumped[StreakKey(99, "A")]; ok {
		t.Fatalf("a key not flagged this run must be dropped")
	}
}

func TestCPUPctSustainedTakesMinWindow(t *testing.T) {
	// pid 1 saturates one core every window (pin); pid 2 bursts then idles.
	// dt=1s: cumulative cpu-seconds delta of 1.0 == 100%/core.
	samples := []map[int]float64{
		{1: 0.0, 2: 0.0},
		{1: 1.0, 2: 1.0}, // both 100% this window
		{1: 2.0, 2: 1.0}, // pid1 100%, pid2 0% (idle)
	}
	got := CPUPctSustained(samples, 1.0)
	if got[1] != 100 {
		t.Fatalf("sustained pin must read 100%%, got %v", got[1])
	}
	if got[2] != 0 {
		t.Fatalf("a burst then idle must read its MIN window (0%%), got %v", got[2])
	}
}

func TestCPUPctDeltaBackwardsCounterSkipped(t *testing.T) {
	// A counter that went backwards (PID reuse) must yield no flag, never a wrong one.
	if _, ok := CPUPctDelta(5.0, 1.0, 1.0, true, true); ok {
		t.Fatalf("backward CPU counter must be skipped (missed flag, never wrong)")
	}
	if _, ok := CPUPctDelta(1.0, 2.0, 1.0, false, true); ok {
		t.Fatalf("missing before-sample must be skipped")
	}
	pct, ok := CPUPctDelta(1.0, 3.0, 2.0, true, true)
	if !ok || pct != 100 {
		t.Fatalf("2 cpu-seconds over 2s window == 100%%/core, got %v ok=%v", pct, ok)
	}
}

func TestNextActionBranches(t *testing.T) {
	if got := NextAction(nil, false, ""); got != "no runaway or orphaned process; no action" {
		t.Fatalf("clean next_action wrong: %q", got)
	}
	if got := NextAction(nil, false, "ps boom"); got == "" || got[:13] != "process scan " {
		t.Fatalf("collect-error next_action wrong: %q", got)
	}
	sprawl := []Finding{{PID: 1, Name: "python", Kind: "orphan-helper", Reasons: []string{"orphaned helper"}}}
	if got := NextAction(sprawl, false, ""); got == "" || !containsSub(got, "orphaned sprawl") {
		t.Fatalf("sprawl-only hint expected: %q", got)
	}
	runaway := []Finding{{PID: 2, Name: "llama", Reasons: []string{"threads 9000 > 2000"}}}
	if got := NextAction(runaway, false, ""); !containsSub(got, "--enact to kill") {
		t.Fatalf("runaway hint expected: %q", got)
	}
}

// TestJSONContractShape proves the emitted JSON matches the Python contract's
// load-bearing keys (schema, ok, thresholds, flagged rows, next_action) so the
// native command is a drop-in for the control pane that folds it.
func TestJSONContractShape(t *testing.T) {
	procs := []Proc{{PID: 90, Name: "llama-cli", Threads: ip(99999)}}
	p := Build(procs, Options{Thresholds: Thresholds{MaxThreads: 2000}, Platform: "linux"})
	raw, err := json.Marshal(p)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	for _, key := range []string{
		"schema", "ok", "platform", "thresholds", "cpu_reap_confirm", "cpu_streaks",
		"scanned", "flagged_count", "actionable_flagged_count", "flagged", "enacted",
		"enact", "next_action",
	} {
		if _, ok := m[key]; !ok {
			t.Fatalf("JSON contract missing key %q in %s", key, raw)
		}
	}
	if m["schema"] != Schema {
		t.Fatalf("schema must be %q, got %v", Schema, m["schema"])
	}
	th, ok := m["thresholds"].(map[string]any)
	if !ok {
		t.Fatalf("thresholds must be an object")
	}
	for _, k := range []string{"max_threads", "max_handles", "max_ws_mb", "max_cpu_pct"} {
		if _, ok := th[k]; !ok {
			t.Fatalf("thresholds missing %q", k)
		}
	}
	flagged, ok := m["flagged"].([]any)
	if !ok || len(flagged) != 1 {
		t.Fatalf("want one flagged row, got %v", m["flagged"])
	}
	row := flagged[0].(map[string]any)
	for _, k := range []string{"pid", "name", "reasons", "protected"} {
		if _, ok := row[k]; !ok {
			t.Fatalf("flagged row missing %q", k)
		}
	}
}

func containsSub(s, sub string) bool {
	return len(s) >= len(sub) && (func() bool {
		for i := 0; i+len(sub) <= len(s); i++ {
			if s[i:i+len(sub)] == sub {
				return true
			}
		}
		return false
	})()
}
