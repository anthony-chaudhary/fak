package loopfleet

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

// classify is the liveness state machine: live within one cadence, stale within
// DarkMultiple cadences, dark beyond that or never-ticked, unknown without a
// usable cadence+tick. These cases pin every branch and the boundary edges.
func TestClassifyStateMachine(t *testing.T) {
	now := time.Date(2026, 6, 30, 12, 0, 0, 0, time.UTC)
	th := loopmgr.HealthThresholds{DefaultCadenceSeconds: 3600, DarkMultiple: 2}
	cadence := int64(3600)
	at := func(secsAgo int64) int64 { return now.UnixNano() - secsAgo*int64(time.Second) }

	cases := []struct {
		name     string
		lastTick int64
		cadence  int64
		want     loopmgr.HealthState
	}{
		{"never ticked with cadence is dark", 0, cadence, loopmgr.HealthDark},
		{"never ticked no cadence is unknown", 0, 0, loopmgr.HealthUnknown},
		{"ticked but no cadence is unknown", at(10), 0, loopmgr.HealthUnknown},
		{"within cadence is live", at(1800), cadence, loopmgr.HealthLive},
		{"exactly at cadence is live", at(3600), cadence, loopmgr.HealthLive},
		{"just past cadence is stale", at(3601), cadence, loopmgr.HealthStale},
		{"within dark window is stale", at(7000), cadence, loopmgr.HealthStale},
		{"exactly at dark boundary is stale", at(7200), cadence, loopmgr.HealthStale},
		{"past dark window is dark", at(7201), cadence, loopmgr.HealthDark},
		{"future tick clamps to live", now.UnixNano() + int64(time.Hour), cadence, loopmgr.HealthLive},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := classify(c.lastTick, c.cadence, now, th); got != c.want {
				t.Errorf("classify = %q, want %q", got, c.want)
			}
		})
	}
}

// deriveRow draws the line between "never fired" (DARK) and "ran but I can't
// place it on the freshness timeline" (UNKNOWN). A loop with recorded runs but
// no usable last tick must NOT be slandered as DARK — classify alone would do
// exactly that (lastTick<=0, known cadence => DARK), which reads as "registered
// but never ticked" and would trip a scheduler into reviving a loop that
// demonstrably ran. deriveRow lifts that case to UNKNOWN; a genuinely empty loop
// (runs==0, no tick) stays DARK.
func TestDeriveRowRanButNoTickIsUnknownNotDark(t *testing.T) {
	now := time.Date(2026, 6, 30, 12, 0, 0, 0, time.UTC)
	th := loopmgr.HealthThresholds{DefaultCadenceSeconds: 3600, DarkMultiple: 2}
	cadence := int64(3600)

	// classify alone, given the same (no tick, known cadence), calls it DARK —
	// proving the correction lives in deriveRow, not classify.
	if got := classify(0, cadence, now, th); got != loopmgr.HealthDark {
		t.Fatalf("precondition: classify(no tick, cadence) = %q, want DARK", got)
	}

	// A loop that ran 5 times but whose rows carried no parseable timestamp:
	// runs>0, lastTick==0. Honest verdict is UNKNOWN, and Dark must be false so a
	// `--json` consumer gating on Dark does not revive a loop that ran.
	ran := deriveRow("dispatch", rawLoop{kind: "dispatch", runs: 5, keep: 2}, cadence, now, th)
	if ran.State != loopmgr.HealthUnknown {
		t.Errorf("ran-but-no-tick State = %q, want UNKNOWN", ran.State)
	}
	if ran.Dark {
		t.Error("ran-but-no-tick Dark = true, want false (it demonstrably ran)")
	}

	// A genuinely empty loop (no runs, no tick) is still DARK — that one really
	// never fired, so the correction must not over-reach.
	empty := deriveRow("dispatch", rawLoop{kind: "dispatch", runs: 0}, cadence, now, th)
	if empty.State != loopmgr.HealthDark || !empty.Dark {
		t.Errorf("empty loop State/Dark = %q/%v, want DARK/true", empty.State, empty.Dark)
	}

	// A loop that ran AND has a fresh tick is classified by age, not forced to
	// UNKNOWN — the correction only fires when the tick is missing.
	fresh := deriveRow("dispatch", rawLoop{
		kind:             "dispatch",
		runs:             3,
		lastTickUnixNano: now.Add(-30 * time.Minute).UnixNano(),
	}, cadence, now, th)
	if fresh.State != loopmgr.HealthLive {
		t.Errorf("ran-and-fresh State = %q, want LIVE", fresh.State)
	}
}

// Fold drives real JSONL through the adapter path. A cadence ledger with a fresh
// row must read LIVE; a dispatch ledger whose rows parse but carry blank
// timestamps (runs counted, no tick) must read UNKNOWN — not DARK — proving the
// "ran but no usable tick" correction holds end to end, not just in deriveRow.
func TestFoldRealLedgersDrawTheRightLine(t *testing.T) {
	now := time.Date(2026, 6, 30, 12, 0, 0, 0, time.UTC)
	th := loopmgr.HealthThresholds{DefaultCadenceSeconds: 3600, DarkMultiple: 2}
	root := t.TempDir()

	write := func(rel, body string) {
		p := filepath.Join(root, rel)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	// cadence ledger: one fresh OK run 10 minutes ago, within the daily cadence.
	freshStamp := now.Add(-10 * time.Minute).UTC().Format(time.RFC3339)
	write(filepath.Join("docs", "cadence", "history.jsonl"),
		`{"generated_at":"`+freshStamp+`","verdict":"OK","commit":"abc123"}`+"\n")

	// dispatch ledger: two rows that PARSE (so runs==2) but carry no usable tick —
	// the field is blank. Pre-fix this folds to DARK; the fix makes it UNKNOWN.
	write(filepath.Join(".dispatch-runs", "progress.jsonl"),
		`{"utc":"","ok":true,"closed_now":1}`+"\n"+
			`{"utc":"","ok":false}`+"\n")

	rep := Fold(root, now, th)

	byKind := map[string]LoopHealth{}
	for _, l := range rep.Loops {
		byKind[l.Kind] = l
	}

	cad, ok := byKind["cadence"]
	if !ok {
		t.Fatalf("cadence loop missing from fold; loops=%+v", rep.Loops)
	}
	if cad.State != loopmgr.HealthLive {
		t.Errorf("fresh cadence loop State = %q, want LIVE", cad.State)
	}

	dis, ok := byKind["dispatch"]
	if !ok {
		t.Fatalf("dispatch loop missing from fold; loops=%+v", rep.Loops)
	}
	if dis.Runs != 2 {
		t.Errorf("dispatch Runs = %d, want 2 (rows parsed)", dis.Runs)
	}
	if dis.State != loopmgr.HealthUnknown {
		t.Errorf("dispatch loop (ran, no tick) State = %q, want UNKNOWN (not DARK)", dis.State)
	}
	if dis.Dark {
		t.Error("dispatch loop Dark = true, want false (it ran twice)")
	}
}

// darkMultiple / defaultCadence fall back to the loopmgr defaults when the
// threshold leaves them zero (or sub-1 for the multiple).
func TestThresholdFallbacks(t *testing.T) {
	if got := darkMultiple(loopmgr.HealthThresholds{}); got != loopmgr.DefaultDarkMultiple {
		t.Errorf("darkMultiple(zero) = %d, want %d", got, loopmgr.DefaultDarkMultiple)
	}
	if got := darkMultiple(loopmgr.HealthThresholds{DarkMultiple: 5}); got != 5 {
		t.Errorf("darkMultiple(5) = %d, want 5", got)
	}
	if got := defaultCadence(loopmgr.HealthThresholds{}); got != loopmgr.DefaultHealthCadenceSeconds {
		t.Errorf("defaultCadence(zero) = %d, want %d", got, loopmgr.DefaultHealthCadenceSeconds)
	}
}

// rollup tallies states and counts DISTINCT ledgers (two loops sharing a ledger
// count once).
func TestRollupCountsStatesAndDistinctLedgers(t *testing.T) {
	loops := []LoopHealth{
		{Ledger: "a", State: loopmgr.HealthLive},
		{Ledger: "a", State: loopmgr.HealthStale},
		{Ledger: "b", State: loopmgr.HealthDark},
		{Ledger: "c", State: loopmgr.HealthUnknown},
	}
	r := rollup(loops, 2)
	if r.Loops != 4 || r.Skipped != 2 {
		t.Errorf("Loops/Skipped = %d/%d, want 4/2", r.Loops, r.Skipped)
	}
	if r.Live != 1 || r.Stale != 1 || r.Dark != 1 || r.Unknown != 1 {
		t.Errorf("state tally = live:%d stale:%d dark:%d unknown:%d, want 1/1/1/1", r.Live, r.Stale, r.Dark, r.Unknown)
	}
	if r.Ledgers != 3 {
		t.Errorf("distinct Ledgers = %d, want 3 (a,b,c)", r.Ledgers)
	}
}

func TestIsSuccess(t *testing.T) {
	for _, ok := range []string{"ok", "success", "SUCCEEDED", " Passed ", "pass"} {
		if !isSuccess(ok) {
			t.Errorf("isSuccess(%q) = false, want true", ok)
		}
	}
	for _, bad := range []string{"", "fail", "error", "timeout", "skipped"} {
		if isSuccess(bad) {
			t.Errorf("isSuccess(%q) = true, want false", bad)
		}
	}
}

func TestParseRFC3339(t *testing.T) {
	ns, ok := parseRFC3339("2026-06-30T12:00:00Z")
	if !ok {
		t.Fatal("parseRFC3339 of a valid stamp returned ok=false")
	}
	if want := time.Date(2026, 6, 30, 12, 0, 0, 0, time.UTC).UnixNano(); ns != want {
		t.Errorf("parseRFC3339 ns = %d, want %d", ns, want)
	}
	if _, ok := parseRFC3339("not-a-time"); ok {
		t.Error("parseRFC3339 of garbage returned ok=true")
	}
}

func TestRound3(t *testing.T) {
	cases := map[float64]float64{
		1.23456: 1.235,
		0.0:     0.0,
		2.0:     2.0,
		1.2344:  1.234,
	}
	for in, want := range cases {
		if got := round3(in); got != want {
			t.Errorf("round3(%v) = %v, want %v", in, got, want)
		}
	}
}
